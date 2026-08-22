//! Does the co-processor dependency graph see every statement that can hold an
//! op, and every read that can create an edge?
//!
//! `ir_grapher.rs` feeds `coprocessor_scheduler.rs`, whose whole purpose is to
//! OVERLAP the RT Core and Tensor Core pipelines. A node missing from this
//! graph is not scheduled; a missing *edge* is worse, because it tells the
//! scheduler two ops are independent when one consumes the other's output.
//!
//! `analyze_stmt` handled 10 of the 17 `Stmt` variants and ended in `_ => {}`,
//! and `wire_expr_dependencies` handled 5 of the 17 `Expr` variants the same
//! way. Measured on `--emit-coprocessor` before the fix:
//!
//! ```text
//!   b = bvh_traverse(a);   RT 2  Tensor 1  cross-pipe 1  412 cycles
//!   b += bvh_traverse(a);  RT 1  Tensor 1  cross-pipe 0  212 cycles
//! ```
//!
//! Zero cross-pipe edges is the scheduler being told the tensor consumer does
//! not depend on the RT producer. This file's central assertion is therefore
//! not a node count but an EQUIVALENCE: `x += e` is `x = x + e`, so the two
//! must produce the same graph. A count would have to be updated whenever the
//! cost model changes; the equivalence holds for as long as the language does.
//!
//! Neither `_ =>` arm remains, so a new AST variant is a compile error here
//! rather than a silently dropped dependency.

use y::ir_grapher::DependencyGrapher;

/// (rt_nodes, tensor_nodes, cross_pipeline_edges, total_nodes)
fn graph_of(source: &str) -> (usize, usize, usize, usize) {
    let tokens = y::lexer::Lexer::new(source).tokenize();
    let prog = y::parser::Parser::new(tokens)
        .parse_program()
        .expect("probe did not parse");
    let mut g = DependencyGrapher::new();
    let ir = g.analyze_program(&prog);
    (
        ir.rt_core_nodes().len(),
        ir.tensor_core_nodes().len(),
        ir.cross_pipeline_edges().len(),
        ir.nodes.len(),
    )
}

/// The plain form every other probe is compared against: two RT ops, the
/// second consuming the first.
const PLAIN: &str = r#"
@unsafe
fn main() {
    let a: I32 = rt_nearest_neighbor(128, 8);
    let b: I32 = bvh_traverse(a);
}
"#;

#[test]
fn the_control_is_not_vacuous() {
    // Everything below asserts "the same as PLAIN". If PLAIN itself graphed
    // nothing, every one of those would pass with the pass deleted.
    let (rt, _, _, nodes) = graph_of(PLAIN);
    assert_eq!(rt, 2, "the plain form must put both RT ops in the graph");
    assert!(nodes >= 2, "the graph must have nodes at all");
}

#[test]
fn a_compound_assign_graphs_exactly_like_its_desugaring() {
    // `x += e` IS `x = x + e`. This is the assertion the whole file is for:
    // before the fix the `+=` form dropped the statement outright, taking the
    // node AND the cross-pipeline edge with it.
    let assign = r#"
@unsafe
fn main() {
    let a: I32 = rt_nearest_neighbor(128, 8);
    let b: I32 = 0;
    b = bvh_traverse(a);
    let frag_A: Fragment<MMA_m16n8k16, A, F16> = ldmatrix(b);
}
"#;
    let compound = assign.replace("b = bvh_traverse(a);", "b += bvh_traverse(a);");

    let want = graph_of(assign);
    let got = graph_of(&compound);
    assert_eq!(
        got, want,
        "`b += bvh_traverse(a)` must graph like `b = bvh_traverse(a)`; \
         got {got:?} against {want:?}"
    );
    // And say out loud what the edge is, so a future change that makes BOTH
    // arms wrong in the same way still fails.
    assert_eq!(want.2, 1, "the RT producer -> Tensor consumer edge must exist");
    assert_eq!(want.0, 2, "both RT ops must be nodes");
}

#[test]
fn every_statement_that_can_carry_an_op_is_walked() {
    // One case per `Stmt` variant that owns a block or an expression and was
    // reached by the old `_ => {}`. Each must graph the RT op that the plain
    // form graphs.
    let cases: &[(&str, &str)] = &[
        (
            "match arm",
            r#"
@unsafe
fn main() {
    let a: I32 = rt_nearest_neighbor(128, 8);
    match a {
        _ => bvh_traverse(a)
    }
}
"#,
        ),
        (
            "return expression",
            r#"
@unsafe
fn main() {
    let a: I32 = rt_nearest_neighbor(128, 8);
    return bvh_traverse(a);
}
"#,
        ),
        (
            "@clock_domain body",
            r#"
@unsafe
fn main() {
    let a: I32 = rt_nearest_neighbor(128, 8);
    @clock_domain(fast) {
        let b: I32 = bvh_traverse(a);
    }
}
"#,
        ),
    ];
    let (want_rt, ..) = graph_of(PLAIN);
    for (what, src) in cases {
        let (rt, ..) = graph_of(src);
        assert_eq!(
            rt, want_rt,
            "an RT op inside a {what} is missing from the dependency graph \
             ({rt} nodes against {want_rt} for the same op written plainly)"
        );
    }
}

#[test]
fn a_read_reaching_a_consumer_indirectly_is_still_a_dependency() {
    // `wire_expr_dependencies` dropped `StructLit`, `MemberAccess`,
    // `GenericCall` and `BlockExpr`. A produced value reaching a consumer
    // through any of them is a real read, and a dropped read is a missing
    // edge -- the consumer is then free to be scheduled before its producer.
    //
    // `StructLit` is the same variant that hid the `takes_reference` gap in
    // the type checker, which is why it leads here.
    //
    // The FIRST version of this test was named for the struct field and never
    // built one -- it asserted on a plain `bvh_traverse(a)` and passed with
    // the `StructLit` arm mutated back to `=> {}`. Caught by mutation, not by
    // review, which is the whole argument for running the mutations.
    fn edges(src: &str) -> usize {
        let tokens = y::lexer::Lexer::new(src).tokenize();
        let prog = y::parser::Parser::new(tokens)
            .parse_program()
            .expect("probe did not parse");
        let mut g = DependencyGrapher::new();
        g.analyze_program(&prog).edges.len()
    }

    // Control: a direct read makes exactly one edge. Without this, "the
    // indirect form makes an edge" could pass on a graph that edges
    // everything to everything.
    let direct = r#"
@unsafe
fn main() {
    let a: I32 = rt_nearest_neighbor(128, 8);
    let b: I32 = a;
}
"#;
    assert_eq!(edges(direct), 1, "a direct read must make exactly one edge");

    let through_struct = r#"
struct P {
    x: I32,
}

@unsafe
fn main() {
    let a: I32 = rt_nearest_neighbor(128, 8);
    let s: P = P { x: a };
}
"#;
    assert_eq!(
        edges(through_struct),
        1,
        "`P {{ x: a }}` reads `a`, so the struct literal's node must depend on \
         a's producer; a dropped edge lets the consumer be scheduled first"
    );

    let through_member = r#"
struct P {
    x: I32,
}

@unsafe
fn main() {
    let a: P = ZeroInit();
    let b: I32 = a.x;
}
"#;
    assert_eq!(
        edges(through_member),
        1,
        "`a.x` reads `a`, so member access must wire an edge"
    );
}
