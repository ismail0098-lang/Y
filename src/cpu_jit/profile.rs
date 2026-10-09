//! Measured outcomes of conditional branches in the original, unoptimized IR.
//!
//! Training sessions have an atomic counter for each outcome. Profile-use
//! sessions only receive LLVM branch metadata, and have no profiling counters.

use sha2::{Digest, Sha256};

/// One conditional branch, identified before LLVM optimization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchSite {
    pub function: String,
    pub block: String,
    pub true_count: u64,
    pub false_count: u64,
}

/// An owned snapshot tied to the exact original IR that was instrumented.
///
/// Counters wrap after 2^64 observations of an individual outcome. Concurrent
/// native calls can update counters during a snapshot: each counter is read
/// atomically, but the snapshot is not a single instant across all counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchProfile {
    ir_hash: [u8; 32],
    sites: Vec<BranchSite>,
}

impl BranchProfile {
    pub fn sites(&self) -> &[BranchSite] {
        &self.sites
    }

    /// SHA-256 of the original IR, including language lowering options.
    pub fn fingerprint(&self) -> &[u8; 32] {
        &self.ir_hash
    }

    pub fn total_observations(&self) -> u128 {
        self.sites
            .iter()
            .map(|site| u128::from(site.true_count) + u128::from(site.false_count))
            .sum()
    }
}

fn fingerprint(ir: &str) -> [u8; 32] {
    Sha256::digest(ir.as_bytes()).into()
}

/// Fit measured ratios into LLVM's two i32 branch weights. Keep zero edges
/// zero, and keep observed edges nonzero when the finite metadata width rounds
/// a very small ratio down. There are no guessed or added observations.
fn weights(true_count: u64, false_count: u64) -> Option<[u32; 2]> {
    let maximum = true_count.max(false_count);
    if maximum == 0 {
        return None;
    }
    if maximum <= u64::from(u32::MAX) {
        return Some([true_count as u32, false_count as u32]);
    }
    let scale = |value: u64| {
        if value == 0 {
            0
        } else {
            ((u128::from(value) * u128::from(u32::MAX) / u128::from(maximum)) as u32).max(1)
        }
    };
    Some([scale(true_count), scale(false_count)])
}

/// Identify only conditional natural-loop headers and direct backedge latches.
/// Work inside the loop, including conditional breaks, keeps its profile.
/// Unreachable and irreducible regions receive no special treatment.
#[cfg(any(test, all(target_os = "linux", target_arch = "x86_64")))]
fn natural_loop_controls(successors: &[Vec<usize>]) -> Vec<bool> {
    use std::collections::{HashMap, HashSet};

    let count = successors.len();
    let mut controls = vec![false; count];
    if count == 0 {
        return controls;
    }
    let mut predecessors = vec![Vec::new(); count];
    for (block, edges) in successors.iter().enumerate() {
        for &next in edges {
            predecessors[next].push(block);
        }
    }
    let mut reachable = vec![false; count];
    let mut pending = vec![0];
    while let Some(block) = pending.pop() {
        if !reachable[block] {
            reachable[block] = true;
            pending.extend(successors[block].iter().copied());
        }
    }

    // Bitsets keep the fixed-point dominance calculation compact even for
    // functions containing many source branches.
    let words = count.div_ceil(64);
    let mut all = vec![0u64; words];
    for (block, &live) in reachable.iter().enumerate() {
        if live {
            all[block / 64] |= 1u64 << (block % 64);
        }
    }
    let mut dominators = vec![all.clone(); count];
    dominators[0].fill(0);
    dominators[0][0] = 1;
    loop {
        let mut changed = false;
        for block in 1..count {
            if !reachable[block] {
                continue;
            }
            let mut next = all.clone();
            for &pred in &predecessors[block] {
                if reachable[pred] {
                    for (word, &pred_word) in next.iter_mut().zip(&dominators[pred]) {
                        *word &= pred_word;
                    }
                }
            }
            next[block / 64] |= 1u64 << (block % 64);
            if next != dominators[block] {
                dominators[block] = next;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let dominates = |header: usize, block: usize| {
        reachable[block] && dominators[block][header / 64] & (1u64 << (header % 64)) != 0
    };
    let mut loops: HashMap<usize, HashSet<usize>> = HashMap::new();
    let mut backedges = Vec::new();
    for (latch, edges) in successors.iter().enumerate() {
        if !reachable[latch] {
            continue;
        }
        for &header in edges {
            if !dominates(header, latch) {
                continue;
            }
            backedges.push((latch, header));
            let members = loops
                .entry(header)
                .or_insert_with(|| HashSet::from([header]));
            let mut pending = vec![latch];
            while let Some(block) = pending.pop() {
                if block != header && dominates(header, block) && members.insert(block) {
                    pending.extend(predecessors[block].iter().copied());
                }
            }
        }
    }
    for (&header, members) in &loops {
        if let [a, b] = successors[header].as_slice() {
            controls[header] = members.contains(a) != members.contains(b);
        }
    }
    for (latch, header) in backedges {
        if let [a, b] = successors[latch].as_slice() {
            let members = &loops[&header];
            if (*a == header && !members.contains(b)) || (*b == header && !members.contains(a)) {
                controls[latch] = true;
            }
        }
    }
    controls
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub(super) use native::{apply, instrument, Instrumentation};

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod native {
    use super::{fingerprint, natural_loop_controls, weights, BranchProfile, BranchSite};
    use crate::cpu_jit::llvm::{Api, Ref};
    use crate::cpu_jit::JitError;
    use std::collections::HashMap;
    use std::ffi::{CStr, CString};
    use std::sync::atomic::{AtomicU64, Ordering};

    pub(crate) struct Instrumentation {
        ir_hash: [u8; 32],
        sites: Vec<BranchSite>,
        counter_symbol: String,
    }

    impl Instrumentation {
        pub(crate) fn symbol(&self) -> &str {
            &self.counter_symbol
        }

        /// `address` must be this training session's live, aligned counter
        /// global, obtained from LLVM ORC. Dropping the session invalidates it.
        pub(crate) unsafe fn snapshot(&self, address: usize) -> BranchProfile {
            let counters = address as *const AtomicU64;
            let sites = self
                .sites
                .iter()
                .enumerate()
                .map(|(index, identity)| BranchSite {
                    function: identity.function.clone(),
                    block: identity.block.clone(),
                    true_count: (*counters.add(2 * index)).load(Ordering::Relaxed),
                    false_count: (*counters.add(2 * index + 1)).load(Ordering::Relaxed),
                })
                .collect();
            BranchProfile {
                ir_hash: self.ir_hash,
                sites,
            }
        }
    }

    struct Branch {
        identity: BranchSite,
        terminator: Ref,
        loop_control: bool,
    }

    unsafe fn name(api: &Api, value: Ref) -> String {
        let mut length = 0;
        let bytes = (api.LLVMGetValueName2)(value, &mut length);
        if length == 0 {
            String::new()
        } else {
            String::from_utf8_lossy(std::slice::from_raw_parts(bytes.cast(), length)).into_owned()
        }
    }

    /// LLVM 23 split the branch opcode. The old branch predicates are retained
    /// in its C API and also work with the LLVM 17 through 22 opcode layout.
    unsafe fn branches(api: &Api, module: Ref) -> Vec<Branch> {
        let mut result = Vec::new();
        let mut function = (api.LLVMGetFirstFunction)(module);
        while !function.is_null() {
            let function_name = name(api, function);
            let mut blocks = Vec::new();
            let mut block = (api.LLVMGetFirstBasicBlock)(function);
            while !block.is_null() {
                blocks.push(block);
                block = (api.LLVMGetNextBasicBlock)(block);
            }
            let indices: HashMap<_, _> = blocks.iter().copied().zip(0..blocks.len()).collect();
            let successors: Vec<Vec<usize>> = blocks
                .iter()
                .map(|&block| {
                    let terminator = (api.LLVMGetBasicBlockTerminator)(block);
                    if terminator.is_null() {
                        Vec::new()
                    } else {
                        (0..(api.LLVMGetNumSuccessors)(terminator))
                            .map(|edge| indices[&(api.LLVMGetSuccessor)(terminator, edge)])
                            .collect()
                    }
                })
                .collect();
            let controls = natural_loop_controls(&successors);
            for (ordinal, &block) in blocks.iter().enumerate() {
                let terminator = (api.LLVMGetBasicBlockTerminator)(block);
                if !terminator.is_null()
                    && !(api.LLVMIsABranchInst)(terminator).is_null()
                    && (api.LLVMIsConditional)(terminator) != 0
                {
                    let block_name = (api.LLVMGetBasicBlockName)(block);
                    let block_name = if block_name.is_null() {
                        String::new()
                    } else {
                        CStr::from_ptr(block_name).to_string_lossy().into_owned()
                    };
                    result.push(Branch {
                        identity: BranchSite {
                            function: function_name.clone(),
                            block: if block_name.is_empty() {
                                format!("#block{ordinal}")
                            } else {
                                block_name
                            },
                            true_count: 0,
                            false_count: 0,
                        },
                        terminator,
                        loop_control: controls[ordinal],
                    });
                }
            }
            function = (api.LLVMGetNextFunction)(function);
        }
        result
    }

    /// Counter writes change memory effects, including through nested calls.
    /// Remove contradictory promises conservatively from all definitions and
    /// call sites in a training module; preserve unrelated ABI/target attrs.
    unsafe fn strip_effect_attributes(api: &Api, module: Ref) {
        let kinds: Vec<_> = [
            "memory",
            "readnone",
            "readonly",
            "writeonly",
            "argmemonly",
            "inaccessiblememonly",
            "inaccessiblemem_or_argmemonly",
            "nosync",
            "speculatable",
        ]
        .iter()
        .map(|attribute| {
            (api.LLVMGetEnumAttributeKindForName)(attribute.as_ptr().cast(), attribute.len())
        })
        .filter(|kind| *kind != 0)
        .collect();
        // The function attribute index is ~0U in the LLVM C API.
        const FUNCTION_INDEX: u32 = u32::MAX;
        let mut function = (api.LLVMGetFirstFunction)(module);
        while !function.is_null() {
            if (api.LLVMIsDeclaration)(function) == 0 {
                for kind in &kinds {
                    (api.LLVMRemoveEnumAttributeAtIndex)(function, FUNCTION_INDEX, *kind);
                }
            }
            let mut block = (api.LLVMGetFirstBasicBlock)(function);
            while !block.is_null() {
                let mut instruction = (api.LLVMGetFirstInstruction)(block);
                while !instruction.is_null() {
                    // Invoke, Call and CallBr have stable opcode numbers.
                    if matches!((api.LLVMGetInstructionOpcode)(instruction), 5 | 45 | 67) {
                        for kind in &kinds {
                            (api.LLVMRemoveCallSiteEnumAttribute)(
                                instruction,
                                FUNCTION_INDEX,
                                *kind,
                            );
                        }
                    }
                    instruction = (api.LLVMGetNextInstruction)(instruction);
                }
                block = (api.LLVMGetNextBasicBlock)(block);
            }
            function = (api.LLVMGetNextFunction)(function);
        }
    }

    /// Insert one atomic increment for each original branch observation.
    /// Edge mode gives each outcome a fixed pointer in a dedicated trampoline.
    /// PHI successors keep the selected-pointer increment before the branch:
    /// their incoming blocks, values, flags and metadata remain untouched.
    /// Neither form reevaluates source conditions or successor expressions.
    /// Loop-only mode restricts fixed addresses to original natural-loop controls.
    pub(crate) unsafe fn instrument(
        api: &Api,
        context: Ref,
        module: Ref,
        original_ir: &str,
        edge_counters: bool,
        loop_edge_counters: bool,
    ) -> Result<Instrumentation, JitError> {
        let branches = branches(api, module);
        let counter_count = branches
            .len()
            .checked_mul(2)
            .and_then(|count| u32::try_from(count.max(1)).ok())
            .ok_or_else(|| JitError::new("too many branches for CPU JIT profiling"))?;
        let ir_hash = fingerprint(original_ir);
        let suffix = ir_hash[..12]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let symbol = CString::new(format!("__y_jit_profile_{suffix}")).unwrap();
        let i64_type = (api.LLVMInt64TypeInContext)(context);
        let array_type = (api.LLVMArrayType)(i64_type, counter_count);
        let global = (api.LLVMAddGlobal)(module, array_type, symbol.as_ptr());
        (api.LLVMSetInitializer)(global, (api.LLVMConstNull)(array_type));
        (api.LLVMSetAlignment)(global, 8);
        // LLVM uniquifies global names if the source defines the same symbol.
        let counter_symbol = name(api, global);
        strip_effect_attributes(api, module);

        let builder = (api.LLVMCreateBuilderInContext)(context);
        let zero = (api.LLVMConstInt)(i64_type, 0, 0);
        let one = (api.LLVMConstInt)(i64_type, 1, 0);
        for (index, branch) in branches.iter().enumerate() {
            let destinations = [
                (api.LLVMGetSuccessor)(branch.terminator, 0),
                (api.LLVMGetSuccessor)(branch.terminator, 1),
            ];
            let can_split = (edge_counters || (loop_edge_counters && branch.loop_control))
                && destinations.iter().all(|&destination| {
                    let first = (api.LLVMGetFirstInstruction)(destination);
                    first.is_null() || (api.LLVMIsAPHINode)(first).is_null()
                });
            if can_split {
                let parent = (api.LLVMGetBasicBlockParent)((api.LLVMGetInstructionParent)(
                    branch.terminator,
                ));
                for (outcome, &destination) in destinations.iter().enumerate() {
                    let edge = (api.LLVMAppendBasicBlockInContext)(
                        context,
                        parent,
                        if outcome == 0 {
                            c"__y_profile_true".as_ptr()
                        } else {
                            c"__y_profile_false".as_ptr()
                        },
                    );
                    (api.LLVMPositionBuilderAtEnd)(builder, edge);
                    let mut indices = [
                        zero,
                        (api.LLVMConstInt)(i64_type, (2 * index + outcome) as u64, 0),
                    ];
                    let pointer = (api.LLVMBuildInBoundsGEP2)(
                        builder,
                        array_type,
                        global,
                        indices.as_mut_ptr(),
                        2,
                        c"".as_ptr(),
                    );
                    let increment = (api.LLVMBuildAtomicRMW)(builder, 1, pointer, one, 2, 0);
                    (api.LLVMSetAlignment)(increment, 8);
                    (api.LLVMBuildBr)(builder, destination);
                    (api.LLVMSetSuccessor)(branch.terminator, outcome as u32, edge);
                }
                continue;
            }
            (api.LLVMPositionBuilderBefore)(builder, branch.terminator);
            let mut true_indices = [zero, (api.LLVMConstInt)(i64_type, 2 * index as u64, 0)];
            let mut false_indices = [zero, (api.LLVMConstInt)(i64_type, 2 * index as u64 + 1, 0)];
            let true_pointer = (api.LLVMBuildInBoundsGEP2)(
                builder,
                array_type,
                global,
                true_indices.as_mut_ptr(),
                2,
                c"".as_ptr(),
            );
            let false_pointer = (api.LLVMBuildInBoundsGEP2)(
                builder,
                array_type,
                global,
                false_indices.as_mut_ptr(),
                2,
                c"".as_ptr(),
            );
            let pointer = (api.LLVMBuildSelect)(
                builder,
                (api.LLVMGetCondition)(branch.terminator),
                true_pointer,
                false_pointer,
                c"".as_ptr(),
            );
            // LLVMAtomicRMWBinOpAdd = 1, LLVMAtomicOrderingMonotonic = 2.
            // Cross-thread monotonic increments pair with AtomicU64 loads.
            let increment = (api.LLVMBuildAtomicRMW)(builder, 1, pointer, one, 2, 0);
            (api.LLVMSetAlignment)(increment, 8);
        }
        (api.LLVMDisposeBuilder)(builder);
        Ok(Instrumentation {
            ir_hash,
            sites: branches.into_iter().map(|branch| branch.identity).collect(),
            counter_symbol,
        })
    }

    /// Apply only measured profiles that match the original IR and every site.
    /// Validate the entire profile before changing the module.
    pub(crate) unsafe fn apply(
        api: &Api,
        context: Ref,
        module: Ref,
        original_ir: &str,
        profile: &BranchProfile,
        include_loop_controls: bool,
    ) -> Result<usize, JitError> {
        if profile.ir_hash != fingerprint(original_ir) {
            return Err(JitError::new(
                "branch profile does not match the original IR",
            ));
        }
        let branches = branches(api, module);
        if branches.len() != profile.sites.len()
            || branches.iter().zip(&profile.sites).any(|(branch, site)| {
                branch.identity.function != site.function || branch.identity.block != site.block
            })
        {
            return Err(JitError::new(
                "branch profile sites do not match the original IR",
            ));
        }
        let prof_kind = (api.LLVMGetMDKindIDInContext)(context, c"prof".as_ptr(), 4);
        let i32_type = (api.LLVMInt32TypeInContext)(context);
        let tag = (api.LLVMMDStringInContext2)(context, c"branch_weights".as_ptr(), 14);
        let mut applied = 0;
        for (branch, site) in branches.iter().zip(&profile.sites) {
            // Iteration counts can become inaccurate path probabilities after
            // loop rotation/vectorization introduces unmeasured fallback checks.
            // Preserve LLVM's default loop policy unless explicitly requested.
            if branch.loop_control && !include_loop_controls {
                continue;
            }
            let Some([true_weight, false_weight]) = weights(site.true_count, site.false_count)
            else {
                continue;
            };
            let mut operands = [
                tag,
                (api.LLVMValueAsMetadata)((api.LLVMConstInt)(i32_type, u64::from(true_weight), 0)),
                (api.LLVMValueAsMetadata)((api.LLVMConstInt)(i32_type, u64::from(false_weight), 0)),
            ];
            let node = (api.LLVMMDNodeInContext2)(context, operands.as_mut_ptr(), operands.len());
            (api.LLVMSetMetadata)(
                branch.terminator,
                prof_kind,
                (api.LLVMMetadataAsValue)(context, node),
            );
            applied += 1;
        }
        Ok(applied)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn natural_loop_controls_preserve_work_breaks_and_multiple_backedges() {
        // A while header with an interior conditional break.
        assert_eq!(
            natural_loop_controls(&[vec![1], vec![2, 5], vec![3, 4], vec![5], vec![1], vec![],]),
            [false, true, false, false, false, false]
        );
        // A direct conditional latch, with block order independent of CFG order.
        assert_eq!(
            natural_loop_controls(&[vec![2], vec![2, 3], vec![1], vec![]]),
            [false, true, false, false]
        );
        // Inner and outer headers; the inner break is still conditional work.
        assert_eq!(
            natural_loop_controls(&[
                vec![1],
                vec![2, 7],
                vec![3],
                vec![4, 6],
                vec![5, 6],
                vec![3],
                vec![1],
                vec![],
            ]),
            [false, true, false, true, false, false, false, false]
        );
        // Merge all backedges before deciding membership for the header.
        assert_eq!(
            natural_loop_controls(&[
                vec![1],
                vec![2, 6],
                vec![3, 4],
                vec![1],
                vec![5],
                vec![1],
                vec![],
            ]),
            [false, true, false, false, false, false, false]
        );
    }

    #[test]
    fn unreachable_and_irreducible_cycles_do_not_receive_loop_policy() {
        assert_eq!(natural_loop_controls(&[]), Vec::<bool>::new());
        assert_eq!(
            natural_loop_controls(&[vec![1], vec![], vec![3, 4], vec![2], vec![]]),
            [false; 5]
        );
        // Neither entry into this cycle dominates the other.
        assert_eq!(
            natural_loop_controls(&[vec![1, 2], vec![2, 3], vec![1, 3], vec![]]),
            [false; 4]
        );
        assert_eq!(
            natural_loop_controls(&[vec![1, 2], vec![], vec![]]),
            [false; 3]
        );
    }

    #[test]
    fn dominance_bitsets_handle_loops_across_word_boundaries() {
        let mut graph: Vec<_> = (0..128).map(|block| vec![block + 1]).collect();
        graph.push(vec![]);
        graph[65] = vec![66, 128];
        graph[127] = vec![65];
        let controls = natural_loop_controls(&graph);
        assert_eq!(controls.iter().filter(|&&control| control).count(), 1);
        assert!(controls[65]);
    }

    #[test]
    fn weights_preserve_measured_ratios_and_zero_outcomes() {
        assert_eq!(weights(0, 0), None);
        assert_eq!(weights(100, 2), Some([100, 2]));
        assert_eq!(weights(0, 987), Some([0, 987]));
        assert_eq!(weights(987, 0), Some([987, 0]));
        assert_eq!(weights(u64::from(u32::MAX), 1), Some([u32::MAX, 1]));
        assert_eq!(weights(u64::MAX, 0), Some([u32::MAX, 0]));
        assert_eq!(weights(0, u64::MAX), Some([0, u32::MAX]));
        assert_eq!(weights(u64::MAX, u64::MAX), Some([u32::MAX, u32::MAX]));
        assert_eq!(weights(u64::MAX, 1), Some([u32::MAX, 1]));
        assert_eq!(
            weights(8_000_000_000, 4_000_000_000),
            Some([u32::MAX, u32::MAX / 2])
        );
    }

    #[test]
    fn fingerprints_include_all_original_ir_bytes() {
        let ir = "define i64 @test() { ret i64 1 }";
        assert_eq!(fingerprint(ir), fingerprint(ir));
        assert_ne!(
            fingerprint(ir),
            fingerprint("define i64 @test() { ret i64 2 }")
        );
        assert_ne!(fingerprint(ir), fingerprint(&format!("{ir}\n")));
    }

    #[test]
    fn snapshots_own_counts_and_total_does_not_overflow_u64() {
        let profile = BranchProfile {
            ir_hash: fingerprint("original IR"),
            sites: vec![BranchSite {
                function: "test".into(),
                block: "entry".into(),
                true_count: u64::MAX,
                false_count: u64::MAX,
            }],
        };
        assert_eq!(profile.fingerprint(), &fingerprint("original IR"));
        assert_eq!(profile.sites()[0].block, "entry");
        assert_eq!(profile.total_observations(), 2 * u128::from(u64::MAX));
        assert_eq!(profile.clone(), profile);
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn instrumentation_and_profile_use_verify_with_effect_attributes() {
        use crate::cpu_jit::llvm::{self, Ref};
        use std::sync::atomic::AtomicU64;

        struct Module {
            api: &'static llvm::Api,
            thread_context: Ref,
            context: Ref,
            module: Ref,
        }
        impl Module {
            unsafe fn parse(ir: &str) -> Self {
                let api = llvm::api().expect("CPU JIT tests require LLVM 17+");
                let (thread_context, context) = if let Some(adopt) = api.context_from_llvm {
                    let context = (api.LLVMContextCreate)();
                    (adopt(context), context)
                } else {
                    let thread_context = (api.LLVMOrcCreateNewThreadSafeContext)();
                    (
                        thread_context,
                        api.context_get_llvm.unwrap()(thread_context),
                    )
                };
                let mut result = Self {
                    api,
                    thread_context,
                    context,
                    module: std::ptr::null_mut(),
                };
                let buffer = (api.LLVMCreateMemoryBufferWithMemoryRangeCopy)(
                    ir.as_ptr().cast(),
                    ir.len(),
                    c"profile-unit-test".as_ptr(),
                );
                let mut message = std::ptr::null_mut();
                let status =
                    (api.LLVMParseIRInContext)(context, buffer, &mut result.module, &mut message);
                assert_eq!(status, 0, "{}", api.message(message));
                result
            }

            unsafe fn verify(&self) -> String {
                let mut message = std::ptr::null_mut();
                let status = (self.api.LLVMVerifyModule)(self.module, 2, &mut message);
                assert_eq!(status, 0, "{}", self.api.message(message));
                self.api
                    .message((self.api.LLVMPrintModuleToString)(self.module))
            }
        }
        impl Drop for Module {
            fn drop(&mut self) {
                unsafe {
                    if !self.module.is_null() {
                        (self.api.LLVMDisposeModule)(self.module);
                    }
                    (self.api.LLVMOrcDisposeThreadSafeContext)(self.thread_context);
                }
            }
        }
        let ir = r#"
            define i64 @leaf(i1 %condition) memory(none) speculatable {
            entry:
                br i1 %condition, label %yes, label %no
            yes:
                ret i64 1
            no:
                ret i64 2
            }
            define i64 @wrapper(i1 %condition) memory(none) {
            entry:
                %value = call i64 @leaf(i1 %condition) memory(none)
                ret i64 %value
            }
        "#;
        unsafe {
            let training = Module::parse(ir);
            let instrumentation = instrument(
                training.api,
                training.context,
                training.module,
                ir,
                false,
                false,
            )
            .unwrap();
            let text = training.verify();
            assert!(text.contains("atomicrmw add"), "{text}");
            assert!(text.contains("select i1 %condition"), "{text}");
            assert!(!text.contains("memory(none)"), "{text}");
            assert!(!text.contains("speculatable"), "{text}");
            let counters = [AtomicU64::new(37), AtomicU64::new(4)];
            let profile = instrumentation.snapshot(counters.as_ptr() as usize);
            assert_eq!(profile.total_observations(), 41);
            assert_eq!(profile.sites()[0].function, "leaf");
            assert_eq!(profile.sites()[0].block, "entry");

            let final_module = Module::parse(ir);
            assert_eq!(
                apply(
                    final_module.api,
                    final_module.context,
                    final_module.module,
                    ir,
                    &profile,
                    false
                )
                .unwrap(),
                1
            );
            let text = final_module.verify();
            assert!(
                text.contains("!\"branch_weights\", i32 37, i32 4"),
                "{text}"
            );
            assert!(!text.contains("atomicrmw"), "{text}");
            assert!(!text.contains("__y_jit_profile_"), "{text}");
            assert!(text.contains("memory(none)"), "{text}");

            let unobserved =
                instrumentation.snapshot([AtomicU64::new(0), AtomicU64::new(0)].as_ptr() as usize);
            let untouched = Module::parse(ir);
            assert_eq!(
                apply(
                    untouched.api,
                    untouched.context,
                    untouched.module,
                    ir,
                    &unobserved,
                    false
                )
                .unwrap(),
                0
            );
            assert!(!untouched.verify().contains("branch_weights"));

            let mut invalid = profile.clone();
            invalid.sites[0].block = "different".into();
            assert!(apply(
                untouched.api,
                untouched.context,
                untouched.module,
                ir,
                &invalid,
                false
            )
            .is_err());
            assert!(apply(
                untouched.api,
                untouched.context,
                untouched.module,
                "changed IR",
                &profile,
                false
            )
            .is_err());
            assert!(!untouched.verify().contains("branch_weights"));

            // The requested counter name may already name a source function.
            // The module symbol table must preserve it and give the counters
            // a distinct exported name, which the ORC lookup then uses.
            let collision_key = "counter-symbol-collision";
            let suffix = fingerprint(collision_key)[..12]
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            let requested = format!("__y_jit_profile_{suffix}");
            let collision = Module::parse(&format!("define i64 @{requested}() {{ ret i64 7 }}"));
            let collision_counters = instrument(
                collision.api,
                collision.context,
                collision.module,
                collision_key,
                false,
                false,
            )
            .unwrap();
            assert_ne!(collision_counters.symbol(), requested);
            let text = collision.verify();
            assert!(
                text.contains(&format!("define i64 @{requested}()")),
                "{text}"
            );
            assert!(
                text.contains(&format!(
                    "@{} = global [1 x i64]",
                    collision_counters.symbol()
                )),
                "{text}"
            );

            // Fixed edge addresses retain the original identities and effects
            // policy. LLVM's uniquifier also protects source block names.
            let edge_training = Module::parse(ir);
            let edge_instrumentation = instrument(
                edge_training.api,
                edge_training.context,
                edge_training.module,
                ir,
                true,
                false,
            )
            .unwrap();
            let text = edge_training.verify();
            assert_eq!(text.matches("atomicrmw add").count(), 2, "{text}");
            assert!(!text.contains("select i1"), "{text}");
            assert!(!text.contains("memory(none)"), "{text}");
            assert!(!text.contains("speculatable"), "{text}");
            assert_eq!(
                edge_instrumentation.snapshot(counters.as_ptr() as usize),
                profile
            );

            // A same-destination branch still observes its true/false condition
            // separately; non-PHI self loops also accept dedicated edge blocks.
            // PHIs (including duplicated incoming edges and cyclic values) must
            // retain their exact incoming blocks, flags and metadata via fallback.
            for (fixture, atomic_count, select_count) in [
                ("define void @same(i1 %c) { entry: br i1 %c, label %end, label %end end: ret void }", 2, 0),
                ("define void @self(i1 %c) { entry: br label %loop loop: br i1 %c, label %loop, label %end end: ret void }", 2, 0),
                ("define i64 @same_phi(i1 %c) { entry: br i1 %c, label %end, label %end end: %v = phi i64 [7, %entry], [7, %entry] ret i64 %v }", 1, 1),
                ("define i64 @loop_phi(i1 %c) { entry: br label %loop loop: %v = phi i64 [0, %entry], [%next, %loop], !annotation !0 %next = add i64 %v, 1 br i1 %c, label %loop, label %end end: ret i64 %v }\n!0 = !{!\"phi-preserved\"}", 1, 1),
                ("define i64 @mixed(i1 %c, i1 %d) { entry: br i1 %c, label %left, label %right left: br i1 %d, label %merge, label %right right: br label %merge merge: %v = phi i64 [1, %left], [2, %right] ret i64 %v }", 3, 1),
                ("define void @names(i1 %c) { entry: br i1 %c, label %__y_profile_true, label %__y_profile_false __y_profile_true: ret void __y_profile_false: ret void }", 2, 0),
            ] {
                let module = Module::parse(fixture);
                let original = module.verify();
                let original_phis: Vec<_> = original.lines().filter(|line| line.contains(" = phi ")).collect();
                let instrumentation = instrument(module.api, module.context, module.module, fixture, true, false).unwrap();
                let text = module.verify();
                assert_eq!(text.matches("atomicrmw add").count(), atomic_count, "{text}");
                assert_eq!(text.matches("select i1").count(), select_count, "{text}");
                let zeros: Vec<_> = (0..8).map(|_| AtomicU64::new(0)).collect();
                assert_eq!(instrumentation.snapshot(zeros.as_ptr() as usize).fingerprint(), &fingerprint(fixture));
                let final_phis: Vec<_> = text.lines().filter(|line| line.contains(" = phi ")).collect();
                assert_eq!(final_phis, original_phis, "{text}");
            }

            // Loop-only mode splits iteration decisions without turning an
            // interior work decision into a counter-dependent branch. Flags
            // affect instrumentation only, leaving profile identities intact.
            for (fixture, loop_atomics, loop_selects) in [
                ("define void @same(i1 %c) { entry: br i1 %c, label %end, label %end end: ret void }", 1, 1),
                ("define void @self(i1 %c) { entry: br label %loop loop: br i1 %c, label %loop, label %end end: ret void }", 2, 0),
                ("define i64 @loop_phi(i1 %c) { entry: br label %loop loop: %v = phi i64 [0, %entry], [%next, %loop] %next = add i64 %v, 1 br i1 %c, label %loop, label %end end: ret i64 %v }", 1, 1),
                ("define void @three(i1 %enter, i1 %more, i1 %work) { entry: br i1 %enter, label %loop, label %end loop: br i1 %more, label %body, label %end body: br i1 %work, label %left, label %right left: br label %loop right: br label %loop end: ret void }", 4, 2),
            ] {
                let module = Module::parse(fixture);
                module.verify();
                let loop_instrumentation = instrument(module.api, module.context, module.module, fixture, false, true).unwrap();
                let text = module.verify();
                assert_eq!(text.matches("atomicrmw add").count(), loop_atomics, "{text}");
                assert_eq!(text.matches("select i1").count(), loop_selects, "{text}");
                let synthetic: Vec<_> = (0..8).map(|index| AtomicU64::new(index)).collect();
                let loop_profile = loop_instrumentation.snapshot(synthetic.as_ptr() as usize);
                let selected = Module::parse(fixture);
                let selected_instrumentation = instrument(selected.api, selected.context, selected.module, fixture, false, false).unwrap();
                selected.verify();
                assert_eq!(loop_profile, selected_instrumentation.snapshot(synthetic.as_ptr() as usize));
                let all = Module::parse(fixture);
                instrument(all.api, all.context, all.module, fixture, true, false).unwrap();
                let both = Module::parse(fixture);
                instrument(both.api, both.context, both.module, fixture, true, true).unwrap();
                assert_eq!(all.verify(), both.verify(), "all-edge flag takes precedence");
            }
        }
    }
}
