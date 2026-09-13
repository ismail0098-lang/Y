//! Names and scalar results of compiler-provided functions.
//!
//! Paths use the same `Namespace_member` spelling as LLVM call lowering.
//! Recognition does not promise that every backend implements a function:
//! known unsupported GPU operations must reach the backend's specific error.
//! Argument and shape checks belong to the checker/lowering that models them.

use crate::llvm_emitter::{LIBC_SYMBOLS, RUNTIME_SYMBOLS};

pub fn is_known_function(name: &str) -> bool {
    scalar_return_type(name).is_some()
        || RUNTIME_SYMBOLS.contains(&name)
        || LIBC_SYMBOLS.contains(&name)
        || matches!(
            name,
            // LLVM host allocation APIs and CPU filesystem lowering.
            "malloc"
                | "File_read"
                | "File_read_to_string"
                | "String_new"
                | "String_clone"
                | "Vec_new"
                // Generic construction and typed/shape-dependent memory APIs.
                | "Pipeline_init"
                | "Fragment_zero"
                | "SharedMemory_alloc"
                | "GlobalMemory_load"
                | "GlobalMemory_load_v4"
                | "GlobalMemory_ld_v4"
                | "BlockTile_load"
                | "load"
                | "ldmatrix"
                | "cp_async"
                | "mma_sync"
                | "block_tile_load"
                | "tile_load"
                | "make_block_ptr2d"
                | "make_block_ptr3d"
                | "block_ptr2d_load"
                | "block_ptr2d_load_v4"
                | "block_ptr2d_advance"
                | "block_ptr3d_load"
                | "block_ptr3d_load_v4"
                | "block_ptr3d_advance"
                | "ld_global_v4_f32"
                | "load_v4"
                | "shared_alloc_u32"
                | "shared_load_v4"
                // A shuffle moves raw 32-bit data; its source determines the
                // scalar interpretation, including for the `_b32` spelling.
                | "shfl_sync_bfly"
                | "shfl_sync_bfly_b32"
                // Exact names with coprocessor mappings in IrGrapher. These
                // mappings stage structured results in shared memory, so this
                // scalar-only table cannot supply their result types.
                | "bvh_traverse"
                | "rt_trace"
                | "rt_nearest_neighbor"
                | "nns_query"
                | "sparse_route"
                | "attention_mask_bvh"
                | "wmma_mma"
                // PTX recognizes these and reports why their lowering is absent.
                | "cp_async_bulk"
                | "tma_load"
                | "tma_load_2d"
                | "wgmma_async"
                | "wgmma_mma_async"
                | "mbarrier_init"
                | "mbarrier_arrive"
                | "mbarrier_try_wait"
                // The R1CS backend derives this result from its active field.
                | "poseidon_hash"
        )
}

/// A result fixed by the builtin's contract, independent of arguments/shapes.
/// `None` means this table cannot supply a scalar type; it does not mean void.
pub fn scalar_return_type(name: &str) -> Option<&'static str> {
    Some(match name {
        "thread_id" | "global_thread_id" | "thread_idx" | "thread_idx_x" | "thread_idx_y"
        | "thread_idx_z" | "block_idx_x" | "block_idx_y" | "block_idx_z" | "block_dim_x"
        | "block_dim_y" | "block_dim_z" | "grid_dim_x" | "grid_dim_y" | "grid_dim_z"
        | "block_arange" | "block_cdiv" => "I32",
        "mul_wide_u32" => "U64",
        "mul_wide_s32" => "I64",
        "u64_lo32" | "u64_hi32" | "add_cc_u32" | "addc_u32" | "addc_cc_u32" | "sub_cc_u32"
        | "subc_u32" | "subc_cc_u32" | "mad_lo_cc_u32" | "madc_lo_u32" | "madc_lo_cc_u32"
        | "mad_hi_cc_u32" | "madc_hi_u32" | "madc_hi_cc_u32" => "U32",
        "sqrtf" | "math_sqrt" | "math_fmin" | "math_fmax" | "warp_reduce_sum"
        | "warp_reduce_max" => "F32",
        "String_eq" | "String_eq_cstr" | "ystr_eq" | "ystr_eq_cstr" => "bool",
        "String_len" | "Vec_len" | "ystr_len" | "yvec_len" | "str_to_i64" => "I64",
        "String_char_at" | "Vec_get_char" | "ystr_char_at" | "yvec_get_char" => "char",
        "printf" | "ychar_to_ascii" | "usleep" => "I32",
        "print"
        | "print_int"
        | "println"
        | "free"
        | "exit"
        | "File_write"
        | "yfile_write"
        | "String_push"
        | "String_push_str"
        | "String_free"
        | "Vec_push"
        | "Vec_free"
        | "ystr_push"
        | "ystr_push_str"
        | "ystr_free"
        | "yvec_push"
        | "yvec_free"
        | "store"
        | "st_global_v4_f32"
        | "store_v4"
        | "GlobalMemory_store_v4"
        | "GlobalMemory_st_v4"
        | "BlockTile_store"
        | "block_tile_store"
        | "tile_store"
        | "block_ptr2d_store"
        | "block_ptr2d_store_v4"
        | "block_ptr3d_store"
        | "block_ptr3d_store_v4"
        | "shared_store_v4"
        | "atomic_add"
        | "atomic_max"
        | "barrier_sync"
        | "membar"
        | "vec_add_v4"
        | "vector_add_v4"
        | "vec_add_unrolled4"
        | "rmsnorm_fast"
        | "rmsnorm_v4"
        | "swiglu_fast"
        | "swiglu_v4" => "void",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_named_dispatch_entries_have_frontend_names() {
        // Read the producer rather than repeating its list in a second test
        // table. Adding a PTX dispatch arm must keep name resolution in sync.
        let source = include_str!("ptx_emitter.rs");
        for tail in source.split("fname == \"").skip(1) {
            let name = tail.split('"').next().unwrap();
            assert!(is_known_function(name), "PTX dispatch missing: {name}");
        }
    }

    #[test]
    fn coprocessor_mapping_entries_have_frontend_names() {
        let mapping = include_str!("ir_grapher.rs")
            .split("fn apply_mappings(")
            .nth(1)
            .expect("coprocessor mapping dispatch")
            .split("fn ")
            .next()
            .unwrap();
        for line in mapping.lines().map(str::trim) {
            if line.starts_with('"') && line.contains("=> {") {
                for name in line.split('"').skip(1).step_by(2) {
                    assert!(
                        is_known_function(name),
                        "coprocessor mapping missing: {name}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_known_namespace_does_not_admit_arbitrary_functions() {
        for name in [
            "String_typo",
            "Vec_typo",
            "File_typo",
            "GlobalMemory_typo",
            "Pipeline_typo",
            "made_up_function",
        ] {
            assert!(
                !is_known_function(name),
                "unknown function admitted: {name}"
            );
        }
    }

    #[test]
    fn shape_dependent_results_are_not_reported_as_scalars_or_void() {
        for name in [
            "load",
            "BlockTile_load",
            "block_ptr2d_load_v4",
            "cp_async",
            "shfl_sync_bfly",
            "shfl_sync_bfly_b32",
            "rt_nearest_neighbor",
        ] {
            assert!(is_known_function(name));
            assert_eq!(scalar_return_type(name), None);
        }
        assert_eq!(scalar_return_type("store"), Some("void"));
        assert_eq!(scalar_return_type("mul_wide_u32"), Some("U64"));
        assert_eq!(scalar_return_type("String_eq"), Some("bool"));
    }
}
