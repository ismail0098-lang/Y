// ============================================================
//  Y  —  IR Analysis & Dependency Grapher
//  ir_grapher.rs
//
//  Walks the validated AST and classifies each computational
//  node into one of two hardware pipelines:
//
//    1. RT Core Pipeline  — sparse routing, tree traversals,
//       nearest-neighbor search, attention masking via BVH.
//    2. Tensor Core Pipeline — dense GEMM, mixed-precision
//       matrix fragments via mma.sync.
//
//  Produces an IrGraph (DAG) consumed by the CoprocessorScheduler
//  to partition work across both accelerators within a single SM.
// ============================================================

#![allow(dead_code)]

use crate::ast::*;

// ── Pipeline Classification ─────────────────────────────────

/// Which hardware accelerator a node is assigned to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pipeline {
    /// BVH traversal / ray-triangle intersection hardware
    RtCore,
    /// Matrix multiply-accumulate (mma.sync) hardware
    TensorCore,
    /// Standard CUDA/SM ALU — neither accelerator
    ScalarAlu,
    /// Synchronization barrier between pipelines
    SyncPoint,
}

/// The mathematical abstraction an RT Core node implements.
#[derive(Debug, Clone, PartialEq)]
pub enum RtCoreMapping {
    /// Matrix rows encoded as rays, columns as planes → dot-product GEMM
    GemmViaRayPlane {
        rows: u32,
        cols: u32,
        precision: Precision,
    },
    /// BVH-accelerated nearest-neighbor / kNN search
    NearestNeighbor {
        dimensions: u32,
        k: u32,
    },
    /// Sparse attention routing — tokens mapped to BVH leaves
    SparseAttentionRoute {
        num_tokens: u32,
        sparsity_ratio: f32,
    },
    /// Generic BVH tree traversal for graph pruning
    TreePrune {
        estimated_nodes: u32,
    },
}

/// The fragment layout a Tensor Core node uses.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorCoreMapping {
    /// Standard dense mma.sync with specified fragment shape
    MmaSync {
        m: u32,
        n: u32,
        k: u32,
        input_precision: Precision,
        accumulator_precision: Precision,
    },
    /// Quantized low-precision GEMM (INT4/INT8)
    QuantizedGemm {
        m: u32,
        n: u32,
        k: u32,
        quant_bits: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    FP32,
    FP16,
    BF16,
    TF32,
    FP8,
    FP4,
    INT8,
    INT4,
}

// ── IR Graph Nodes ──────────────────────────────────────────

/// Unique identifier for a node in the IR dependency graph.
pub type NodeId = usize;

/// A single node in the co-processing IR graph.
#[derive(Debug, Clone)]
pub struct IrNode {
    pub id: NodeId,
    pub pipeline: Pipeline,
    pub rt_mapping: Option<RtCoreMapping>,
    pub tensor_mapping: Option<TensorCoreMapping>,
    /// Estimated execution latency in cycles (from HardwareProfile).
    pub estimated_cycles: f64,
    /// Human-readable label for debug output.
    pub label: String,
    /// Source span for diagnostics.
    pub span: Span,
    /// Output precision of this node.
    pub output_precision: Precision,
    /// Whether this node requires a quantization pass on its output.
    pub needs_quantization: bool,
    /// Shared memory bytes this node requires.
    pub smem_bytes: u32,
}

/// A directed edge in the dependency DAG.
#[derive(Debug, Clone)]
pub struct IrEdge {
    pub from: NodeId,
    pub to: NodeId,
    /// If the edge crosses pipelines, a sync point is required.
    pub crosses_pipeline: bool,
    /// Bytes transferred across the edge (through shared memory).
    pub transfer_bytes: u32,
}

/// The complete dependency graph for dual-accelerator scheduling.
#[derive(Debug, Clone)]
pub struct IrGraph {
    pub nodes: Vec<IrNode>,
    pub edges: Vec<IrEdge>,
    next_id: NodeId,
}

impl IrGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            next_id: 0,
        }
    }

    fn alloc_node(
        &mut self,
        pipeline: Pipeline,
        label: String,
        span: Span,
        estimated_cycles: f64,
    ) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(IrNode {
            id,
            pipeline,
            rt_mapping: None,
            tensor_mapping: None,
            estimated_cycles,
            label,
            span,
            output_precision: Precision::FP32,
            needs_quantization: false,
            smem_bytes: 0,
        });
        id
    }

    fn add_edge(&mut self, from: NodeId, to: NodeId, transfer_bytes: u32) {
        let from_pipe = self.nodes[from].pipeline;
        let to_pipe = self.nodes[to].pipeline;
        let crosses = from_pipe != to_pipe
            && from_pipe != Pipeline::SyncPoint
            && to_pipe != Pipeline::SyncPoint
            && from_pipe != Pipeline::ScalarAlu
            && to_pipe != Pipeline::ScalarAlu;

        self.edges.push(IrEdge {
            from,
            to,
            crosses_pipeline: crosses,
            transfer_bytes,
        });
    }

    /// Returns all nodes assigned to the RT Core pipeline.
    pub fn rt_core_nodes(&self) -> Vec<&IrNode> {
        self.nodes
            .iter()
            .filter(|n| n.pipeline == Pipeline::RtCore)
            .collect()
    }

    /// Returns all nodes assigned to the Tensor Core pipeline.
    pub fn tensor_core_nodes(&self) -> Vec<&IrNode> {
        self.nodes
            .iter()
            .filter(|n| n.pipeline == Pipeline::TensorCore)
            .collect()
    }

    /// Returns edges that cross pipeline boundaries (requiring sync).
    pub fn cross_pipeline_edges(&self) -> Vec<&IrEdge> {
        self.edges.iter().filter(|e| e.crosses_pipeline).collect()
    }

    /// Total estimated shared memory pressure in bytes.
    pub fn total_smem_bytes(&self) -> u32 {
        self.nodes.iter().map(|n| n.smem_bytes).sum()
    }

    /// Returns the critical path length in estimated cycles.
    pub fn critical_path_cycles(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        // Topological longest-path via dynamic programming
        let n = self.nodes.len();
        let mut dist = vec![0.0f64; n];

        // Initialize with node costs
        for node in &self.nodes {
            dist[node.id] = node.estimated_cycles;
        }

        // Relax edges (assumes DAG — nodes are in topological order by construction)
        for edge in &self.edges {
            let new_dist = dist[edge.from] + self.nodes[edge.to].estimated_cycles;
            if new_dist > dist[edge.to] {
                dist[edge.to] = new_dist;
            }
        }

        dist.iter().cloned().fold(0.0, f64::max)
    }
}

// ── AST → IR Graph Lowering ─────────────────────────────────

/// The dependency grapher that walks the AST and builds the IrGraph.
pub struct DependencyGrapher {
    pub graph: IrGraph,
    /// Maps variable names to their producing node IDs.
    var_producers: std::collections::HashMap<String, NodeId>,
}

impl DependencyGrapher {
    pub fn new() -> Self {
        Self {
            graph: IrGraph::new(),
            var_producers: std::collections::HashMap::new(),
        }
    }

    /// Analyze the full program and produce the co-processing IR graph.
    pub fn analyze_program(&mut self, program: &Program) -> &IrGraph {
        for item in &program.items {
            match item {
                Item::Kernel(k) => self.analyze_kernel(k),
                Item::Func(f) => self.analyze_func(f),
                _ => {}
            }
        }
        &self.graph
    }

    fn analyze_kernel(&mut self, kernel: &KernelDecl) {
        self.analyze_block(&kernel.body);
    }

    fn analyze_func(&mut self, func: &FuncDecl) {
        self.analyze_block(&func.body);
    }

    fn analyze_block(&mut self, block: &Block) {
        for stmt in &block.stmts {
            self.analyze_stmt(stmt);
        }
    }

    fn analyze_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let {
                name, init, span, ..
            } => {
                if let Some(expr) = init {
                    let (pipeline, label) = self.classify_expr(expr);
                    let cycles = self.estimate_expr_cycles(expr, pipeline);
                    let node_id =
                        self.graph
                            .alloc_node(pipeline, label, span.clone(), cycles);

                    // Apply RT/Tensor mappings based on classification
                    self.apply_mappings(node_id, expr);

                    // Wire up data dependencies
                    self.wire_expr_dependencies(node_id, expr);

                    self.var_producers.insert(name.clone(), node_id);
                }
            }
            Stmt::Assign { target, value, span, .. } => {
                let (pipeline, label) = self.classify_expr(value);
                let cycles = self.estimate_expr_cycles(value, pipeline);
                let node_id = self.graph.alloc_node(pipeline, label, span.clone(), cycles);
                self.apply_mappings(node_id, value);
                self.wire_expr_dependencies(node_id, value);

                if let Expr::Ident(name, _) = target {
                    self.var_producers.insert(name.clone(), node_id);
                }
            }
            Stmt::For { body, .. } => self.analyze_block(body),
            Stmt::While { body, .. } => self.analyze_block(body),
            Stmt::If {
                then_block,
                else_block,
                ..
            } => {
                self.analyze_block(then_block);
                if let Some(eb) = else_block {
                    self.analyze_block(eb);
                }
            }
            Stmt::SafeBlock(block, _) => self.analyze_block(block),
            Stmt::GhostBlock(block, _) => self.analyze_block(block),
            Stmt::Chisel(block, _) => self.analyze_block(block),
            Stmt::Expr(expr) => {
                let (pipeline, label) = self.classify_expr(expr);
                let cycles = self.estimate_expr_cycles(expr, pipeline);
                let node_id =
                    self.graph
                        .alloc_node(pipeline, label, expr.span().clone(), cycles);
                self.wire_expr_dependencies(node_id, expr);
            }
            _ => {}
        }
    }

    /// Classify an expression into its target pipeline.
    fn classify_expr(&self, expr: &Expr) -> (Pipeline, String) {
        match expr {
            // ── Tensor Core indicators ──
            Expr::Call { func, .. } => match &**func {
                Expr::Ident(name, _) => match name.as_str() {
                    "mma_sync" | "wmma_mma" | "hmma" => {
                        (Pipeline::TensorCore, format!("tensor_mma:{}", name))
                    }
                    "ldmatrix" => {
                        (Pipeline::TensorCore, "tensor_ldmatrix".into())
                    }
                    // ── RT Core indicators ──
                    "bvh_traverse" | "rt_trace" | "optix_trace" => {
                        (Pipeline::RtCore, format!("rt_traverse:{}", name))
                    }
                    "rt_nearest_neighbor" | "nns_query" => {
                        (Pipeline::RtCore, format!("rt_nns:{}", name))
                    }
                    "sparse_route" | "attention_mask_bvh" => {
                        (Pipeline::RtCore, format!("rt_sparse:{}", name))
                    }
                    // ── Synchronization ──
                    "cp_async" => (Pipeline::ScalarAlu, "async_copy".into()),
                    "barrier_sync" => (Pipeline::SyncPoint, "barrier".into()),
                    _ => (Pipeline::ScalarAlu, format!("call:{}", name)),
                },
                Expr::Path {
                    namespace, member, ..
                } => {
                    if namespace == "barrier" && member == "sync" {
                        (Pipeline::SyncPoint, "barrier::sync".into())
                    } else if namespace == "RtCore" {
                        (Pipeline::RtCore, format!("rt:{}::{}", namespace, member))
                    } else if namespace == "TensorCore" || namespace == "Fragment" {
                        (
                            Pipeline::TensorCore,
                            format!("tensor:{}::{}", namespace, member),
                        )
                    } else {
                        (Pipeline::ScalarAlu, format!("{}::{}", namespace, member))
                    }
                }
                _ => (Pipeline::ScalarAlu, "call:unknown".into()),
            },
            Expr::Path {
                namespace, member, ..
            } => {
                if namespace == "Fragment" && member == "zero" {
                    (Pipeline::TensorCore, "tensor:fragment_zero".into())
                } else if namespace == "barrier" && member == "sync" {
                    (Pipeline::SyncPoint, "barrier::sync".into())
                } else {
                    (Pipeline::ScalarAlu, format!("path:{}::{}", namespace, member))
                }
            }
            // Binary ops are scalar ALU unless part of a larger tensor/rt pattern
            Expr::BinaryOp { .. } => (Pipeline::ScalarAlu, "alu:binop".into()),
            _ => (Pipeline::ScalarAlu, "scalar".into()),
        }
    }

    /// Rough cycle estimate for an expression based on pipeline assignment.
    fn estimate_expr_cycles(&self, expr: &Expr, pipeline: Pipeline) -> f64 {
        match pipeline {
            Pipeline::RtCore => {
                // RT core traversals are high-latency but overlappable
                match expr {
                    Expr::Call { func, .. } => match &**func {
                        Expr::Ident(name, _) if name == "bvh_traverse" => 200.0,
                        Expr::Ident(name, _) if name == "rt_nearest_neighbor" => 180.0,
                        Expr::Ident(name, _) if name == "sparse_route" => 150.0,
                        _ => 100.0,
                    },
                    _ => 100.0,
                }
            }
            Pipeline::TensorCore => {
                // mma.sync latency from HW profile (~42 cycles for F16)
                match expr {
                    Expr::Call { func, .. } => match &**func {
                        Expr::Ident(name, _) if name == "mma_sync" => 42.0,
                        Expr::Ident(name, _) if name == "ldmatrix" => 28.0,
                        _ => 42.0,
                    },
                    _ => 42.0,
                }
            }
            Pipeline::SyncPoint => 35.0, // bar.sync latency
            Pipeline::ScalarAlu => {
                match expr {
                    Expr::BinaryOp { op, .. } => match op {
                        BinaryOp::Mul => 4.5,
                        BinaryOp::Div => 17.0,
                        _ => 4.0,
                    },
                    _ => 4.0,
                }
            }
        }
    }

    /// Apply RT/Tensor core mappings to a node based on its expression.
    fn apply_mappings(&mut self, node_id: NodeId, expr: &Expr) {
        if let Expr::Call { func, args, .. } = expr {
            match &**func {
                Expr::Ident(name, _) => match name.as_str() {
                    "mma_sync" | "wmma_mma" => {
                        self.graph.nodes[node_id].tensor_mapping =
                            Some(TensorCoreMapping::MmaSync {
                                m: 16,
                                n: 8,
                                k: 16,
                                input_precision: Precision::FP16,
                                accumulator_precision: Precision::FP32,
                            });
                        self.graph.nodes[node_id].output_precision = Precision::FP32;
                        self.graph.nodes[node_id].smem_bytes = 16 * 16 * 2; // fragment staging
                    }
                    "bvh_traverse" | "rt_trace" => {
                        let rows = self.extract_int_arg(args, 0).unwrap_or(64) as u32;
                        let cols = self.extract_int_arg(args, 1).unwrap_or(64) as u32;
                        self.graph.nodes[node_id].rt_mapping =
                            Some(RtCoreMapping::GemmViaRayPlane {
                                rows,
                                cols,
                                precision: Precision::FP32,
                            });
                        self.graph.nodes[node_id].output_precision = Precision::FP32;
                        self.graph.nodes[node_id].needs_quantization = true;
                        self.graph.nodes[node_id].smem_bytes = rows * cols * 4;
                    }
                    "rt_nearest_neighbor" | "nns_query" => {
                        let dims = self.extract_int_arg(args, 0).unwrap_or(128) as u32;
                        let k = self.extract_int_arg(args, 1).unwrap_or(8) as u32;
                        self.graph.nodes[node_id].rt_mapping =
                            Some(RtCoreMapping::NearestNeighbor {
                                dimensions: dims,
                                k,
                            });
                        self.graph.nodes[node_id].smem_bytes = dims * k * 4;
                    }
                    "sparse_route" | "attention_mask_bvh" => {
                        let tokens = self.extract_int_arg(args, 0).unwrap_or(512) as u32;
                        self.graph.nodes[node_id].rt_mapping =
                            Some(RtCoreMapping::SparseAttentionRoute {
                                num_tokens: tokens,
                                sparsity_ratio: 0.9,
                            });
                        self.graph.nodes[node_id].smem_bytes = tokens * 4;
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }

    /// Wire data-dependency edges from producer variables to a consumer node.
    fn wire_expr_dependencies(&mut self, consumer: NodeId, expr: &Expr) {
        match expr {
            Expr::Ident(name, _) => {
                if let Some(&producer) = self.var_producers.get(name) {
                    let transfer = self.estimate_transfer_bytes(producer);
                    self.graph.add_edge(producer, consumer, transfer);
                }
            }
            Expr::Call { args, .. } => {
                for arg in args {
                    self.wire_expr_dependencies(consumer, arg);
                }
            }
            Expr::BinaryOp { left, right, .. } => {
                self.wire_expr_dependencies(consumer, left);
                self.wire_expr_dependencies(consumer, right);
            }
            Expr::Index { base, index, .. } => {
                self.wire_expr_dependencies(consumer, base);
                self.wire_expr_dependencies(consumer, index);
            }
            Expr::UnaryOp { operand, .. } => {
                self.wire_expr_dependencies(consumer, operand);
            }
            _ => {}
        }
    }

    fn extract_int_arg(&self, args: &[Expr], idx: usize) -> Option<i64> {
        args.get(idx).and_then(|e| {
            if let Expr::IntLit(v, _) = e {
                Some(*v)
            } else {
                None
            }
        })
    }

    fn estimate_transfer_bytes(&self, producer: NodeId) -> u32 {
        if producer < self.graph.nodes.len() {
            let node = &self.graph.nodes[producer];
            match node.output_precision {
                Precision::FP32 => 4,
                Precision::FP16 | Precision::BF16 => 2,
                Precision::INT8 | Precision::FP8 => 1,
                Precision::INT4 | Precision::FP4 => 1,
                _ => 4,
            }
        } else {
            4
        }
    }
}
