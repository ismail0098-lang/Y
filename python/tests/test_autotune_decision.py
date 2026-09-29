"""The autotune acceptance rule, pinned with exact numbers rather than a clock.

`AutotunedKernel` used to rank ~30 tile candidates by a single timing pass over
a function that, in the documented unit-test suite, returns instantly. The
winner was therefore whichever candidate the scheduler happened to favour, and
`python -m unittest` failed about one run in four. Ranking by noise is not a
weak test - it is an autotuner that answers at random.

The replacement prefers the compiler's own analytic pick unless a measurement
beats it decisively. That "decisively" is two rules, and both were verified by
mutation; testing them through real timings detected a dropped floor in only
about 4 runs out of 10, because microsecond-scale repeatability is not
something a loaded machine offers. A pure predicate needs no clock.
"""

import unittest

from y_lang.autotune_decorator import MEASUREMENT_FLOOR_MS, measurement_beats_model


class TestAutotuneDecision(unittest.TestCase):
    def test_a_clear_win_above_the_floor_is_accepted(self):
        """The control. Without this, "always return the model" passes."""
        self.assertTrue(measurement_beats_model(1.0, 20.0, within=0.05))
        self.assertTrue(measurement_beats_model(0.5, 8.0, within=0.30))

    def test_a_win_narrower_than_the_run_s_own_noise_is_rejected(self):
        # 10% faster, but some candidate's own repeats varied by 30%.
        self.assertFalse(measurement_beats_model(1.0, 1.1, within=0.30))
        # ...and the same margin IS accepted once the run is quiet.
        self.assertTrue(measurement_beats_model(1.0, 1.1, within=0.05))

    def test_nothing_below_the_launch_cost_floor_can_win(self):
        """The rule that a consistent microsecond difference is not a result.

        A GPU launch plus a synchronise costs tens of microseconds, so a
        decorated function finishing faster than that never launched anything.
        This holds however large and however repeatable the difference looks -
        which is exactly the case `between > within` style reasoning gets
        wrong, since a stub's differences can be perfectly consistent.
        """
        tiny = MEASUREMENT_FLOOR_MS / 10.0
        self.assertFalse(measurement_beats_model(tiny, tiny * 100.0, within=0.0))
        self.assertFalse(measurement_beats_model(tiny, 50.0, within=0.0))

    def test_the_floor_is_a_boundary_not_a_slope(self):
        just_under = MEASUREMENT_FLOOR_MS * 0.999
        just_over = MEASUREMENT_FLOOR_MS * 1.001
        self.assertFalse(measurement_beats_model(just_under, 100.0, within=0.0))
        self.assertTrue(measurement_beats_model(just_over, 100.0, within=0.0))

    def test_the_floor_is_a_plausible_launch_cost(self):
        """A floor of 0 disables the rule; a huge one disables measurement."""
        self.assertGreater(MEASUREMENT_FLOOR_MS, 0.0)
        self.assertLess(MEASUREMENT_FLOOR_MS, 1.0)


if __name__ == "__main__":
    unittest.main()
