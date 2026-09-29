import unittest
import os
from pathlib import Path
import y_lang

class TestGpuArchitectFeatures(unittest.TestCase):
    def setUp(self):
        y_lang.clear_disk_cache()

    def tearDown(self):
        y_lang.clear_disk_cache()

    def test_disk_binary_cache(self):
        kernel_src = """
        kernel test_cache_kernel(A: GlobalMemory<F32>, B: GlobalMemory<F32>) {
            @invariant(true)
            for i in 0..1024 step 4 {
                st_global_v4_f32(A, 1.0);
            }
        }
        """
        # First compilation -> Miss
        ptx1 = y_lang.compile_to_ptx(kernel_src, target_sm="sm_90a")
        stats1 = y_lang.get_cache_stats()
        self.assertEqual(stats1["misses"], 1)
        self.assertEqual(stats1["disk_hits"], 0)
        self.assertEqual(stats1["cached_files_count"], 1)

        # Second compilation (new python session simulation) -> Disk Hit
        # Clear in-memory cache to force disk hit
        y_lang.compiler._JIT_CACHE.clear()
        ptx2 = y_lang.compile_to_ptx(kernel_src, target_sm="sm_90a")
        stats2 = y_lang.get_cache_stats()
        self.assertEqual(ptx1, ptx2)
        self.assertEqual(stats2["disk_hits"], 1)

        # Third compilation -> Memory Hit
        ptx3 = y_lang.compile_to_ptx(kernel_src, target_sm="sm_90a")
        stats3 = y_lang.get_cache_stats()
        self.assertEqual(stats3["mem_hits"], 1)

    def test_block_tile_boundary_masking_ptx(self):
        kernel_src = """
        kernel test_block_tile(A: GlobalMemory<F32>, B: GlobalMemory<F32>) {
            let tile: BlockTile<F32, 128> = {};
            let val: F32 = block_tile_load(A, 10, 128);
            block_tile_store(B, 10, val, 128);
        }
        """
        ptx = y_lang.compile_to_ptx(kernel_src, target_sm="sm_90a")
        self.assertIn("Y BLOCK TILE LOAD - AUTOMATIC BOUNDARY MASKING", ptx)
        self.assertIn("Y BLOCK TILE STORE - AUTOMATIC BOUNDARY MASKING", ptx)
        self.assertIn("setp.lt.u32", ptx)
        self.assertIn("@%", ptx)

    def test_automated_vectorizing_loop_pass_ptx(self):
        kernel_src = """
        kernel test_vector_loop(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>) {
            @invariant(true)
            for i in 0..1024 step 4 {
                vec_add_unrolled4(A, B, C);
            }
        }
        """
        ptx = y_lang.compile_to_ptx(kernel_src, target_sm="sm_90a")
        self.assertIn("Y AUTOMATED VECTORIZING PASS", ptx)
        self.assertIn("ld.global.cs.v4.f32", ptx)
        self.assertIn("st.global.cs.v4.f32", ptx)

if __name__ == "__main__":
    unittest.main()
