import os
import unittest
import y_lang


def _fresh_pick(n: int) -> tuple:
    """A brand-new decorated stub's pick for (n, n, n).

    Deliberately NOT reusing one decorator: `AutotunedKernel` memoises per
    shape, so calling one instance five times proves only that a dict works.
    """
    @y_lang.autotune()
    def stub(config, a, b, c):
        return config

    return stub(n, n, n).tile()

class TestPythonJITAndDebugMode(unittest.TestCase):
    def test_jit_decorator_grid_launch(self):
        @y_lang.jit
        def simple_add():
            return """
            kernel add_kernel(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>) {
                let mut i: I32 = 0;
                while i < 1024 {
                    C[i] = A[i] + B[i];
                    i = i + 1;
                }
            }
            """

        src_str = simple_add.source() if callable(simple_add.source) else simple_add.source
        self.assertTrue(callable(simple_add))
        self.assertIn("add_kernel", src_str)
        # Test launcher indexing syntax simple_add[grid](args)
        launcher = simple_add[128]
        self.assertTrue(callable(launcher))

    def test_autotune_decorator(self):
        """A decorated stub must return the MODEL's pick, deterministically.

        This used to assert `cta_m >= 64` while the decorator ranked ~30
        candidates by timing a function that returns instantly - so the winner
        was whichever candidate the scheduler happened to favour, and the
        documented `python -m unittest` command failed about one run in four
        (observed cta_m of 128, 128, 32, 256, 16, ... over 12 runs). Ranking by
        noise is not a weak test, it is a wrong autotuner.
        """
        @y_lang.autotune()
        def dummy_matmul(config, a, b, c):
            return config

        picks = {_fresh_pick(1024) for _ in range(5)}
        self.assertEqual(
            len(picks), 1, f"autotune is not deterministic on a stub: {picks}"
        )

        conf = dummy_matmul(1024, 1024, 1024)
        self.assertIsNotNone(conf)
        self.assertTrue(hasattr(conf, "cta_m"))
        self.assertGreaterEqual(conf.cta_m, 64)

    def test_autotune_still_follows_a_real_measurement(self):
        """The control: falling back to the model must not become ignoring
        measurement entirely.

        "Always return the analytic pick" passes the test above and deletes the
        whole point of the decorator, the same way "refuse everything" passes a
        refusal census. A candidate that is genuinely much faster must win.
        """
        import time
        from y_lang.autotune_decorator import AutotuneConfig

        # Busy-wait, not `time.sleep`. Sleeping for 0.2 ms overshoots by
        # whatever the scheduler feels like, which inflates the fast
        # candidate's own dispersion past the margin it has to clear - this
        # control failed about 4 runs in 40 on sleep, for a reason that had
        # nothing to do with the code under test.
        def spin(seconds):
            end = time.perf_counter() + seconds
            while time.perf_counter() < end:
                pass

        fast = AutotuneConfig(64, 256, 64, 2, 4, 3, 8)
        others = [AutotuneConfig(128, 128, 32), AutotuneConfig(256, 128, 32)]

        @y_lang.autotune(configs=[others[0], fast, others[1]])
        def measured(config, a, b, c):
            spin(0.0004 if config.tile() == fast.tile() else 0.008)
            return config

        got = measured(1024, 1024, 1024)
        self.assertEqual(
            got.tile(), fast.tile(),
            f"a 20x faster candidate was not selected (got {got})",
        )

    def test_microsecond_differences_do_not_decide_a_tile(self):
        """A consistent but sub-launch-cost difference must NOT pick a tile.

        `between > within` alone is not enough, and this is the case that shows
        it: a candidate that is reliably 4x "faster" at 5 microseconds is not
        faster at anything - a GPU launch plus a synchronise costs tens of
        microseconds, so a decorated function finishing that quickly never
        launched. Without the floor this test fails deterministically; with
        only the determinism check above, the same mutation slips through 28
        runs in 30.
        """
        import time
        from y_lang.autotune_decorator import AutotuneConfig, MEASUREMENT_FLOOR_MS

        def spin(seconds):
            end = time.perf_counter() + seconds
            while time.perf_counter() < end:
                pass

        quick = AutotuneConfig(16, 32, 32, 1, 1, 2, 1)

        @y_lang.autotune(configs=[AutotuneConfig(256, 128, 32), quick])
        def tiny(config, a, b, c):
            # 9x apart, and the slower arm is still under the floor - so the
            # margin test cannot be what rejects this, only the floor can.
            spin(0.000005 if config.tile() == quick.tile() else 0.000045)
            return config

        got = tiny(1024, 1024, 1024)
        self.assertLess(0.000045 * 1000.0, MEASUREMENT_FLOOR_MS)
        self.assertNotEqual(
            got.tile(), quick.tile(),
            "a 4x difference below the launch-cost floor selected a tile; the "
            "harness was timing itself, not a kernel",
        )
        self.assertEqual(got.cta_m, 128, f"expected the model's pick, got {got}")

    def test_autotune_is_shape_sensitive(self):
        """...and it must not collapse to one answer for every shape."""
        @y_lang.autotune()
        def dummy(config, a, b, c):
            return config

        small = dummy(64, 64, 64)
        large = dummy(4096, 4096, 4096)
        self.assertLess(
            small.cta_m, large.cta_m,
            f"same tile for 64^3 and 4096^3: {small} vs {large}",
        )

    def test_cpu_debug_interpreter_mode(self):
        os.environ["Y_INTERPRETER"] = "1"
        self.assertTrue(y_lang.is_interpreter_enabled())

        @y_lang.jit
        def debug_kernel():
            return """
            kernel debug_k(A: GlobalMemory<F32>) {
                store(A, 1.0);
            }
            """

        # Under Y_INTERPRETER=1, launch prints debug diagnostics without throwing CUDA errors
        debug_kernel[1]([1.0, 2.0, 3.0])

        os.environ["Y_INTERPRETER"] = "0"
        self.assertFalse(y_lang.is_interpreter_enabled())

if __name__ == "__main__":
    unittest.main()
