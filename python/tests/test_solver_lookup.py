"""The SMT solver beside liby.so is found from any working directory.

The compiler's relative solver candidates (`venv/bin/z3`, ...) resolve against
the process's working directory, and inside Python that is wherever the
script started - so a repository's own `venv/bin/z3` was not found and every
`@invariant` was refused as unverifiable. The library now also looks beside
its own file (`type_checker::z3_candidates`); `current_exe()` would name the
interpreter here, so it reads which file this code was mapped from.

Each run copies the built library to `<root>/target/release/liby.so`, puts a
stub solver at `<root>/venv/bin/z3` that records that it ran and answers
`unsat`, and compiles from an unrelated directory with `PATH` and `HOME`
stripped. The control is the same copy without the stub.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from y_lang import compiler

PROGRAM = '''
import y_lang
src = """kernel k(A: GlobalMemory<F32>) {
    @invariant(i >= 0)
    for i in 0..16 {
        A[i] = 1.0;
    }
}
"""
try:
    y_lang.compile_to_ptx(src, target_sm="sm_80")
    print("COMPILED")
except Exception as error:
    print("REFUSED", " ".join(str(error).split()))
'''

PROFILE = "SM_VERSION=8.0\nCOMPUTE_CAPABILITY=8.0\nGPU_NAME=PinnedByTest\nSM_COUNT=66\n"


@unittest.skipUnless(sys.platform.startswith("linux"), "the stub solver is a POSIX shell script")
class TestSolverBesideTheLibrary(unittest.TestCase):
    def setUp(self):
        try:
            self.library = Path(compiler._find_liby()).resolve()
        except (OSError, RuntimeError) as error:
            self.skipTest(f"no built liby.so to copy: {error}")
        self.directory = tempfile.TemporaryDirectory(prefix="y-solver-lookup-")
        self.addCleanup(self.directory.cleanup)
        self.scratch = Path(self.directory.name)

    def compile_beside(self, tag, with_stub):
        root = self.scratch / tag
        lib_dir = root / "target" / "release"
        lib_dir.mkdir(parents=True)
        library = shutil.copy(self.library, lib_dir / "liby.so")
        marker = root / "stub_ran"
        if with_stub:
            stub = root / "venv" / "bin" / "z3"
            stub.parent.mkdir(parents=True)
            # Shell builtins only: PATH is gone. The query is read to EOF so
            # the compiler's write never meets a closed pipe.
            stub.write_text(f"#!/bin/sh\n: > '{marker}'\nwhile read -r _q; do :; done\necho unsat\n")
            stub.chmod(0o755)
        cwd = self.scratch / f"{tag}_cwd"
        cwd.mkdir()
        (cwd / ".ysu_hw_profile").write_text(PROFILE)
        env = {
            "PYTHONPATH": str(Path(compiler.__file__).resolve().parents[1]),
            "Y_LIB_PATH": str(library),
            "PATH": "/nonexistent-path",
            "HOME": "/nonexistent-home",
            "YSU_CACHE_DIR": str(cwd / "cache"),
        }
        result = subprocess.run(
            [sys.executable, "-c", PROGRAM], cwd=cwd, env=env, capture_output=True, text=True, timeout=300
        )
        return marker.exists(), result.stdout + result.stderr

    def test_a_solver_beside_the_library_is_found(self):
        ran, out = self.compile_beside("beside", with_stub=True)
        self.assertTrue(ran, f"the solver beside liby.so was not run:\n{out}")
        self.assertIn("COMPILED", out)

    def test_without_one_the_invariant_is_refused(self):
        ran, out = self.compile_beside("none", with_stub=False)
        self.assertFalse(ran)
        self.assertIn("REFUSED", out)
        self.assertIn("SMT solver could not be run", out)


if __name__ == "__main__":
    unittest.main()
