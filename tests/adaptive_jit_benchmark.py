"""CPU-only checks for interpreting benchmark evidence, not GPU performance tests."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "benchmark_adaptive_jit", Path(__file__).resolve().parents[1] / "tools/benchmark_adaptive_jit.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def measurements(repeats=3, tier="Tuned", paired_savings=None):
    result = []
    for repeat in range(repeats):
        for mode in benchmark.MODES:
            pairs = [] if mode == "baseline" else [
                {"baseline_us": 10.0, "selected_us": 10.0 - saving}
                for saving in (paired_savings if paired_savings is not None else [2.0] * 15)
            ]
            result.append({
                "mode": mode, "m": 256, "n": 256, "k": 256, "weight_copies": 1,
                "calls": 100000, "setup_s": 0.001, "first32_s": 0.0003,
                "tier": "Baseline" if mode == "baseline" else tier,
                "cache_hit": mode == "cached", "tuning_attempts": int(mode == "adaptive"),
                "maintenance_s": 3.0 if mode == "adaptive" else 0.0,
                "tuning_s": 3.0 if mode == "adaptive" else 0.0,
                "candidates_measured": 8 if mode == "adaptive" else 0,
                "checkpoints": [{"calls": 1000, "elapsed_s": 0.02}, {
                    "calls": 100000, "elapsed_s": 5.0 if mode == "adaptive" else 2.0}],
                "pairs": pairs, "baseline_rel_l2": 1e-6, "selected_rel_l2": 2e-6,
                "device": "test device",
            })
    return result


class SummaryTests(unittest.TestCase):
    def test_projection_includes_measured_calls_and_host_lifecycle_disadvantage(self):
        summary = benchmark.summarize_shape(measurements())
        self.assertEqual(summary["break_even"]["status"], "projected")
        self.assertEqual(summary["break_even"]["additional_calls"], 1500000)
        self.assertEqual(summary["break_even"]["total_calls"], 1600000)
        self.assertEqual(summary["maintenance_s"], 3.0)
        self.assertEqual(summary["cache_hits"], 3)

    def test_retained_baseline_cannot_claim_payoff_from_lucky_timing(self):
        summary = benchmark.summarize_shape(measurements(tier="RetainedBaseline"))
        self.assertEqual(summary["break_even"]["status"], "no_kernel_improvement")
        self.assertIsNone(summary["break_even"]["total_calls"])

    def test_positive_median_with_negative_lower_tail_is_unresolved(self):
        summary = benchmark.summarize_shape(measurements(paired_savings=[-1.0] * 3 + [2.0] * 12))
        self.assertGreater(summary["steady"]["adaptive"]["gain_percent"], 5)
        self.assertEqual(summary["break_even"]["status"], "unresolved")

    def test_small_stable_gain_below_five_percent_is_unresolved(self):
        summary = benchmark.summarize_shape(measurements(paired_savings=[0.1] * 15))
        self.assertGreater(summary["steady"]["adaptive"]["saving_p10_min_us"], 0)
        self.assertEqual(summary["break_even"]["status"], "unresolved")

    def test_one_bad_trial_or_mixed_decisions_blocks_projection(self):
        for field, value in (("tier", "RetainedBaseline"),
                             ("pairs", [{"baseline_us": 10, "selected_us": 11}] * 15)):
            records = measurements()
            records[1][field] = value
            summary = benchmark.summarize_shape(records)
            self.assertEqual(summary["break_even"]["status"], "unresolved")

    def test_single_trial_is_exploratory(self):
        summary = benchmark.summarize_shape(measurements(repeats=1))
        self.assertEqual(summary["break_even"]["status"], "insufficient_repeats")

    def test_current_measurements_ignore_historical_tuning_fields(self):
        records = measurements()
        for record in records:
            record.update(baseline_us=50000, selected_us=1)
        summary = benchmark.summarize_shape(records)
        self.assertEqual(summary["steady"]["adaptive"]["saving_us"], 2)
        self.assertEqual(summary["steady"]["cached"]["saving_us"], 2)

    def test_worker_rejects_accidental_cache_miss(self):
        record = deepcopy(measurements()[2])
        record["cache_hit"] = False
        with self.assertRaisesRegex(ValueError, "restore a decision"):
            benchmark.validate_worker(record, "cached", (256, 256, 256, 1), 100000)

    def test_existing_output_directory_is_not_modified(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            sentinel = directory / "untouched.txt"
            sentinel.write_text("existing results", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                benchmark.main(["--output", str(directory)])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing results")
            self.assertEqual(list(directory.iterdir()), [sentinel])


if __name__ == "__main__":
    unittest.main()
