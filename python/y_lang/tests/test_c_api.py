import unittest
import sys
from pathlib import Path

# Add python folder to sys.path
pkg_dir = Path(__file__).resolve().parent.parent.parent
if str(pkg_dir) not in sys.path:
    sys.path.insert(0, str(pkg_dir))

import y_lang

class TestYLangPythonAPI(unittest.TestCase):

    def test_compile_simple_ysu(self):
        source = """
        kernel main(x: GlobalMemory<F16>, y: GlobalMemory<F32>) {
            let val: F32 = 1.0;
        }
        """
        ptx = y_lang.compile_to_ptx(source, target_sm="sm_80")
        self.assertIn(".version", ptx)
        self.assertIn(".target sm_80", ptx)
        self.assertIn(".visible .entry main", ptx)

    def test_autotune_search_space(self):
        candidates = y_lang.generate_autotune_search_space(1024, 1024, 1024)
        self.assertTrue(len(candidates) > 0)
        self.assertIn("cta_m", candidates[0])
        self.assertIn("num_warps", candidates[0])

    def test_jit_decorator(self):
        @y_lang.jit(target_sm="sm_89")
        def my_kernel():
            return """
            kernel matmul(a: GlobalMemory<F16>, b: GlobalMemory<F32>) {
                let x: F32 = 1.0;
            }
            """

        self.assertTrue(hasattr(my_kernel, "ptx"))
        self.assertIn(".target sm_89", my_kernel.ptx)
        self.assertIn(".visible .entry matmul", my_kernel.ptx)

if __name__ == "__main__":
    unittest.main()
