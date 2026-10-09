//! Conservative outer-loop unrolling policy for the original lowered CFG.
//! Loop IDs change only where there is no existing loop metadata.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, PartialEq, Eq)]
struct NaturalLoop {
    header: usize,
    members: BTreeSet<usize>,
    latches: BTreeSet<usize>,
}

/// Reachable dominance backedges form natural loops, merging all latches of
/// one header. This does not classify arbitrary irreducible cycles as loops.
fn natural_loops(successors: &[Vec<usize>]) -> Vec<NaturalLoop> {
    let count = successors.len();
    if count == 0 || successors.iter().flatten().any(|&next| next >= count) {
        return Vec::new();
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
    let words = count.div_ceil(64);
    let mut all = vec![0_u64; words];
    for (block, &live) in reachable.iter().enumerate() {
        if live {
            all[block / 64] |= 1_u64 << (block % 64);
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
            next[block / 64] |= 1_u64 << (block % 64);
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
        reachable[block] && dominators[block][header / 64] & (1_u64 << (header % 64)) != 0
    };
    let mut loops: BTreeMap<usize, NaturalLoop> = BTreeMap::new();
    for (latch, edges) in successors.iter().enumerate() {
        if !reachable[latch] {
            continue;
        }
        for &header in edges {
            if !dominates(header, latch) {
                continue;
            }
            let found = loops.entry(header).or_insert_with(|| NaturalLoop {
                header,
                members: BTreeSet::from([header]),
                latches: BTreeSet::new(),
            });
            found.latches.insert(latch);
            let mut pending = vec![latch];
            while let Some(block) = pending.pop() {
                if block != header && dominates(header, block) && found.members.insert(block) {
                    pending.extend(predecessors[block].iter().copied());
                }
            }
        }
    }
    loops.into_values().collect()
}

fn outer_loops(loops: &[NaturalLoop]) -> Vec<&NaturalLoop> {
    loops
        .iter()
        .filter(|outer| {
            loops.iter().any(|inner| {
                inner.members.len() < outer.members.len() && inner.members.is_subset(&outer.members)
            })
        })
        .filter(|outer| {
            // A loop cannot own an instruction's sole LoopID when that latch
            // branches back to another natural-loop header as well.
            outer.latches.iter().all(|latch| {
                loops
                    .iter()
                    .filter(|other| other.latches.contains(latch))
                    .count()
                    == 1
            })
        })
        .filter(|outer| {
            // Preserve overlapping regions that have no strict loop nesting.
            loops.iter().all(|other| {
                other.members.is_disjoint(&outer.members)
                    || other.members.is_subset(&outer.members)
                    || outer.members.is_subset(&other.members)
            })
        })
        .collect()
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub(super) unsafe fn disable_outer_unrolling(
    api: &super::llvm::Api,
    context: super::llvm::Ref,
    module: super::llvm::Ref,
) -> usize {
    use std::collections::HashMap;
    use std::ptr::null_mut;

    let kind = (api.LLVMGetMDKindIDInContext)(context, c"llvm.loop".as_ptr(), 9);
    let tag = (api.LLVMMDStringInContext2)(context, c"llvm.loop.unroll.disable".as_ptr(), 24);
    let mut disable_operands = [tag];
    let disable = (api.LLVMMDNodeInContext2)(
        context,
        disable_operands.as_mut_ptr(),
        disable_operands.len(),
    );
    let mut annotations = 0;
    let mut function = (api.LLVMGetFirstFunction)(module);
    while !function.is_null() {
        let mut blocks = Vec::new();
        let mut block = (api.LLVMGetFirstBasicBlock)(function);
        while !block.is_null() {
            blocks.push(block);
            block = (api.LLVMGetNextBasicBlock)(block);
        }
        let indices: HashMap<_, _> = blocks.iter().copied().zip(0..blocks.len()).collect();
        let terminators: Vec<_> = blocks
            .iter()
            .map(|&block| (api.LLVMGetBasicBlockTerminator)(block))
            .collect();
        let successors: Vec<Vec<usize>> = terminators
            .iter()
            .map(|&terminator| {
                if terminator.is_null() {
                    Vec::new()
                } else {
                    (0..(api.LLVMGetNumSuccessors)(terminator))
                        .map(|edge| indices[&(api.LLVMGetSuccessor)(terminator, edge)])
                        .collect()
                }
            })
            .collect();
        let loops = natural_loops(&successors);
        for outer in outer_loops(&loops) {
            if outer.latches.iter().any(|&latch| {
                (api.LLVMIsABranchInst)(terminators[latch]).is_null()
                    || !(api.LLVMGetMetadata)(terminators[latch], kind).is_null()
            }) {
                // All latches of one loop must share one ID. Do not overwrite
                // any original ID or references to it (e.g. parallel accesses).
                continue;
            }
            let temporary = (api.LLVMTemporaryMDNode)(context, null_mut(), 0);
            let mut operands = [temporary, disable];
            let loop_id =
                (api.LLVMMDNodeInContext2)(context, operands.as_mut_ptr(), operands.len());
            // This C API consumes/deletes its temporary. A direct self-reference
            // resolves into a distinct node; do not dispose temporary afterward.
            (api.LLVMMetadataReplaceAllUsesWith)(temporary, loop_id);
            let value = (api.LLVMMetadataAsValue)(context, loop_id);
            for &latch in &outer.latches {
                (api.LLVMSetMetadata)(terminators[latch], kind, value);
            }
            annotations += 1;
        }
        function = (api.LLVMGetNextFunction)(function);
    }
    annotations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_loops_merge_latches_and_select_only_strict_outers() {
        // entry -> outer -> middle -> inner -> middle latch -> outer latch
        let loops = natural_loops(&[
            vec![1],
            vec![2, 8],
            vec![3, 7],
            vec![4, 6],
            vec![5],
            vec![3],
            vec![2],
            vec![1],
            vec![],
        ]);
        assert_eq!(
            loops.iter().map(|item| item.header).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(
            outer_loops(&loops)
                .iter()
                .map(|item| item.header)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        let siblings = natural_loops(&[
            vec![1],
            vec![2, 8],
            vec![3, 4],
            vec![2],
            vec![5],
            vec![6, 7],
            vec![5],
            vec![1],
            vec![],
        ]);
        assert_eq!(outer_loops(&siblings).len(), 1);
        assert_eq!(outer_loops(&siblings)[0].header, 1);
        let multi = natural_loops(&[vec![1], vec![2, 5], vec![2, 3], vec![1, 4], vec![1], vec![]]);
        assert_eq!(multi[0].latches, BTreeSet::from([3, 4]));
        assert_eq!(multi[1].latches, BTreeSet::from([2]));
        assert_eq!(outer_loops(&multi).len(), 1);
    }

    #[test]
    fn unreachable_irreducible_shared_and_overlapping_regions_preserve_policy() {
        assert!(natural_loops(&[]).is_empty());
        assert!(natural_loops(&[vec![1], vec![]]).is_empty());
        assert!(natural_loops(&[vec![99]]).is_empty());
        assert!(natural_loops(&[vec![1], vec![], vec![2]]).is_empty());
        assert!(natural_loops(&[vec![1, 2], vec![2, 3], vec![1, 3], vec![]]).is_empty());
        let shared = natural_loops(&[vec![1], vec![2, 4], vec![3], vec![1, 2], vec![]]);
        assert!(outer_loops(&shared).is_empty());
        let overlapping = [
            NaturalLoop {
                header: 1,
                members: BTreeSet::from([1, 2, 3, 4]),
                latches: BTreeSet::from([4]),
            },
            NaturalLoop {
                header: 2,
                members: BTreeSet::from([2, 3]),
                latches: BTreeSet::from([3]),
            },
            NaturalLoop {
                header: 4,
                members: BTreeSet::from([3, 4, 5]),
                latches: BTreeSet::from([5]),
            },
        ];
        assert!(outer_loops(&overlapping).is_empty());
    }

    #[test]
    fn nested_dominance_spans_bitset_words() {
        let mut graph: Vec<Vec<usize>> = (0..140).map(|i| vec![i + 1]).collect();
        graph.push(Vec::new());
        graph[1] = vec![2, 140];
        graph[66] = vec![67, 138];
        graph[137] = vec![66];
        graph[139] = vec![1];
        let loops = natural_loops(&graph);
        assert_eq!(outer_loops(&loops).len(), 1);
        assert_eq!(outer_loops(&loops)[0].header, 1);
        assert_eq!(outer_loops(&loops)[0].latches, BTreeSet::from([139]));
    }
}

#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
mod native_tests {
    use super::disable_outer_unrolling;
    use crate::cpu_jit::llvm::{self, Ref};
    use std::ffi::CStr;
    use std::ptr::null_mut;

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
                module: null_mut(),
            };
            let buffer = (api.LLVMCreateMemoryBufferWithMemoryRangeCopy)(
                ir.as_ptr().cast(),
                ir.len(),
                c"outer-loop-unit".as_ptr(),
            );
            let mut message = null_mut();
            let status =
                (api.LLVMParseIRInContext)(context, buffer, &mut result.module, &mut message);
            assert_eq!(status, 0, "{}", api.message(message));
            result
        }

        unsafe fn verify(&self) -> String {
            let mut message = null_mut();
            let status = (self.api.LLVMVerifyModule)(self.module, 2, &mut message);
            assert_eq!(status, 0, "{}", self.api.message(message));
            self.api
                .message((self.api.LLVMPrintModuleToString)(self.module))
        }

        unsafe fn metadata(&self, function_name: &str, block_name: &str, kind: &str) -> Ref {
            let mut function = (self.api.LLVMGetFirstFunction)(self.module);
            while !function.is_null() {
                let mut len = 0;
                let name = (self.api.LLVMGetValueName2)(function, &mut len);
                let name =
                    std::str::from_utf8(std::slice::from_raw_parts(name.cast(), len)).unwrap();
                if name == function_name {
                    let mut block = (self.api.LLVMGetFirstBasicBlock)(function);
                    while !block.is_null() {
                        if CStr::from_ptr((self.api.LLVMGetBasicBlockName)(block))
                            .to_str()
                            .unwrap()
                            == block_name
                        {
                            let id = (self.api.LLVMGetMDKindIDInContext)(
                                self.context,
                                kind.as_ptr().cast(),
                                kind.len() as u32,
                            );
                            return (self.api.LLVMGetMetadata)(
                                (self.api.LLVMGetBasicBlockTerminator)(block),
                                id,
                            );
                        }
                        block = (self.api.LLVMGetNextBasicBlock)(block);
                    }
                    panic!("missing block {block_name}");
                }
                function = (self.api.LLVMGetNextFunction)(function);
            }
            panic!("missing function {function_name}");
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

    const MULTI: &str = r#"
define void @multi(i1 %a, i1 %b, i1 %c) {
entry: br label %outer
outer: br i1 %a, label %inner, label %exit, !prof !0
inner: br i1 %b, label %body, label %choose
body: br label %inner
choose: br i1 %c, label %left, label %right
left: br label %outer
right: br label %outer
exit: ret void
}
!0 = !{!"branch_weights", i32 3, i32 1}
"#;

    #[test]
    fn distinct_loop_ids_share_all_latches_and_preserve_inner_loops_and_profiles() {
        let additional = r#"
define void @triple(i1 %a, i1 %b, i1 %c) {
entry: br label %outer
outer: br i1 %a, label %middle, label %exit
middle: br i1 %b, label %inner, label %outer_latch
inner: br i1 %c, label %inner_latch, label %middle_latch
inner_latch: br label %inner
middle_latch: br label %middle
outer_latch: br label %outer
exit: ret void
}
define void @siblings(i1 %a, i1 %b, i1 %c) {
entry: br i1 %a, label %left_outer, label %right_outer
left_outer: br i1 %b, label %left_inner, label %exit
left_inner: br i1 %c, label %left_body, label %left_latch
left_body: br label %left_inner
left_latch: br label %left_outer
right_outer: br i1 %b, label %right_inner, label %exit
right_inner: br i1 %c, label %right_body, label %right_latch
right_body: br label %right_inner
right_latch: br label %right_outer
exit: ret void
}
define void @single(i1 %a) {
entry: br label %loop
loop: br i1 %a, label %loop, label %exit
exit: ret void
}
"#;
        unsafe {
            let module = Module::parse(&format!("{MULTI}\n{additional}"));
            module.verify();
            let profile = module.metadata("multi", "outer", "prof");
            assert!(!profile.is_null());
            assert_eq!(
                disable_outer_unrolling(module.api, module.context, module.module),
                5
            );
            let printed = module.verify();
            let multi = module.metadata("multi", "left", "llvm.loop");
            assert!(!multi.is_null());
            assert_eq!(multi, module.metadata("multi", "right", "llvm.loop"));
            assert_eq!(profile, module.metadata("multi", "outer", "prof"));
            let ids = [
                multi,
                module.metadata("triple", "middle_latch", "llvm.loop"),
                module.metadata("triple", "outer_latch", "llvm.loop"),
                module.metadata("siblings", "left_latch", "llvm.loop"),
                module.metadata("siblings", "right_latch", "llvm.loop"),
            ];
            for (i, &id) in ids.iter().enumerate() {
                assert!(!id.is_null());
                assert!(ids[..i].iter().all(|&prior| prior != id));
            }
            for (function, block) in [
                ("multi", "body"),
                ("triple", "inner_latch"),
                ("siblings", "left_body"),
                ("siblings", "right_body"),
                ("single", "loop"),
            ] {
                assert!(module.metadata(function, block, "llvm.loop").is_null());
            }
            assert_eq!(
                printed
                    .lines()
                    .filter(|line| line.contains(" = distinct !{!"))
                    .count(),
                5
            );
            assert!(printed.contains("!\"llvm.loop.unroll.disable\""));
            assert_eq!(
                disable_outer_unrolling(module.api, module.context, module.module),
                0
            );
            assert_eq!(
                module.verify(),
                printed,
                "second application preserves all existing IDs"
            );
        }
    }

    #[test]
    fn existing_metadata_nonbranch_latches_and_ambiguous_cfgs_are_unchanged() {
        let existing = MULTI.replace(
            "left: br label %outer",
            "left: br label %outer, !llvm.loop !1",
        )
            + "\n!1 = distinct !{!1, !2}\n!2 = !{!\"llvm.loop.vectorize.enable\", i1 true}\n";
        let switch = MULTI.replace("i1 %c)", "i1 %c, i32 %n)").replace(
            "left: br label %outer",
            "left: switch i32 %n, label %outer [ i32 0, label %outer ]",
        );
        let ambiguous = r#"
define void @shared(i1 %a, i1 %b) {
entry: br label %outer
outer: br i1 %a, label %inner, label %exit
inner: br label %latch
latch: br i1 %b, label %outer, label %inner
exit: ret void
}
define void @irreducible(i1 %a, i1 %b) {
entry: br i1 %a, label %left, label %right
left: br i1 %b, label %right, label %exit
right: br i1 %b, label %left, label %exit
exit: ret void
}
define void @unreachable(i1 %a, i1 %b) {
entry: ret void
outer: br i1 %a, label %inner, label %exit
inner: br i1 %b, label %body, label %latch
body: br label %inner
latch: br label %outer
exit: ret void
}
"#;
        for ir in [existing.as_str(), switch.as_str(), ambiguous] {
            unsafe {
                let module = Module::parse(ir);
                let before = module.verify();
                assert_eq!(
                    disable_outer_unrolling(module.api, module.context, module.module),
                    0
                );
                assert_eq!(module.verify(), before);
            }
        }
    }
}
