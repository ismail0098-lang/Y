#!/usr/bin/env python3
"""Measure adaptive GEMM lifecycle cost and separate-process decision reuse.

Uses the Rust core directly. No Python bindings or extra Python packages are needed.
Every invocation creates a new output directory and preserves raw measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHAPES = [(256, 256, 256, 1), (512, 512, 512, 1),
                  (1024, 1024, 1024, 1), (1, 4096, 4096, 6)]
MODES = ("baseline", "adaptive", "cached")
SOURCE_PATHS = [
    "Cargo.toml", "Cargo.lock", "examples/adaptive_jit_bench.rs",
    "tools/benchmark_adaptive_jit.py", "src/adaptive_jit.rs",
    "src/adaptive_jit/cache.rs", "src/cuda_runtime.rs", "src/empirical_autotune.rs",
    "src/autotuner.rs", "src/ptx_emitter.rs", "src/sentinel.rs",
]


def quantile(values, fraction):
    """Linearly interpolated empirical quantile; descriptive, not a confidence CI."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_summary(records):
    trials = []
    for record in records:
        pairs = record["pairs"]
        if not pairs:
            continue
        differences = [p["baseline_us"] - p["selected_us"] for p in pairs]
        trials.append({
            "baseline_us": statistics.median(p["baseline_us"] for p in pairs),
            "selected_us": statistics.median(p["selected_us"] for p in pairs),
            "saving_us": statistics.median(differences),
            "saving_p10_us": quantile(differences, 0.1),
            "saving_p90_us": quantile(differences, 0.9),
        })
    if not trials:
        return None
    baseline = statistics.median(t["baseline_us"] for t in trials)
    saving = statistics.median(t["saving_us"] for t in trials)
    return {
        "baseline_us": baseline,
        "selected_us": statistics.median(t["selected_us"] for t in trials),
        "saving_us": saving,
        "gain_percent": 100.0 * saving / baseline,
        "saving_p10_min_us": min(t["saving_p10_us"] for t in trials),
        "saving_p90_max_us": max(t["saving_p90_us"] for t in trials),
        "trials": trials,
    }


def summarize_shape(records):
    by_mode = {mode: [r for r in records if r["mode"] == mode] for mode in MODES}
    if any(not runs for runs in by_mode.values()):
        raise ValueError("each shape needs baseline, adaptive, and cached measurements")
    counts = set.intersection(*[
        {c["calls"] for c in record["checkpoints"]} for record in records
    ])
    totals = {}
    for calls in sorted(counts):
        totals[str(calls)] = {
            mode: statistics.median(
                next(c["elapsed_s"] for c in r["checkpoints"] if c["calls"] == calls)
                for r in runs
            ) for mode, runs in by_mode.items()
        }
    if not totals:
        raise ValueError("measurements have no common lifecycle checkpoint")
    steady = {mode: paired_summary(by_mode[mode]) for mode in ("adaptive", "cached")}
    adaptive = by_mode["adaptive"]
    tiers = Counter(r["tier"] for r in adaptive)
    calls = max(counts)
    disadvantage = totals[str(calls)]["adaptive"] - totals[str(calls)]["baseline"]
    estimate = {
        "status": "unresolved", "measured_calls": calls,
        "lifecycle_disadvantage_s": disadvantage,
        "additional_calls": None, "total_calls": None,
    }
    gain = steady["adaptive"]
    if set(tiers) == {"RetainedBaseline"}:
        estimate["status"] = "no_kernel_improvement"
    elif len(adaptive) < 3:
        estimate["status"] = "insufficient_repeats"
    elif (set(tiers) == {"Tuned"} and gain is not None
          and len(gain["trials"]) == len(adaptive)
          and gain["saving_p10_min_us"] > 0 and gain["gain_percent"] >= 5.0):
        extra = max(0, math.ceil(disadvantage * 1e6 / gain["saving_us"]))
        estimate.update(status="projected", additional_calls=extra, total_calls=calls + extra)
    first = records[0]
    return {
        "shape": [first["m"], first["n"], first["k"]],
        "weight_copies": first["weight_copies"],
        "weight_bytes": first["n"] * first["k"] * 2 * first["weight_copies"],
        "repeats": len(adaptive), "totals_s": totals, "steady": steady,
        "adaptive_tiers": dict(tiers),
        "cached_tiers": dict(Counter(r["tier"] for r in by_mode["cached"])),
        "cache_hits": sum(bool(r["cache_hit"]) for r in by_mode["cached"]),
        "maintenance_s": statistics.median(r["maintenance_s"] for r in adaptive),
        "candidates_measured": [r["candidates_measured"] for r in adaptive],
        "setup_s": {mode: statistics.median(r["setup_s"] for r in runs)
                    for mode, runs in by_mode.items()},
        "maximum_relative_l2": max(
            r[field] for r in records for field in ("baseline_rel_l2", "selected_rel_l2")
        ),
        "break_even": estimate,
    }


def command_metadata(command):
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except OSError as error:
        return {"command": command, "error": str(error)}


def gpu_metadata():
    return command_metadata([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,clocks.sm,"
        "clocks.mem,pstate,temperature.gpu,power.draw,utilization.gpu",
        "--format=csv",
    ])


def metadata(args):
    return {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {"output": str(args.output), "repeats": args.repeats,
                      "calls": args.calls, "shapes": args.shapes},
        "python": sys.version,
        "rustc": command_metadata(["rustc", "-Vv"]),
        "git_head": command_metadata(["git", "rev-parse", "HEAD"]),
        "git_status": command_metadata(["git", "status", "--short"]),
        "gpu_before": gpu_metadata(),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in SOURCE_PATHS if (ROOT / name).is_file()},
        "environment": {name: os.environ[name] for name in (
            "CUDA_VISIBLE_DEVICES", "CUDA_CACHE_DISABLE", "CUDA_CACHE_PATH",
            "Y_SMEM_PAD", "RUSTFLAGS", "CARGO_BUILD_TARGET"
        ) if name in os.environ},
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def shape_label(summary):
    return "×".join(map(str, summary["shape"]))


def render_report(summaries, records, run_metadata):
    devices = ", ".join(sorted({r["device"] for r in records}))
    lines = [
        "# Adaptive GPU JIT benchmark", "",
        f"Measured on {devices}. Started {run_metadata['started_utc']}.", "",
        "These are synthetic FP16-input/F32-output GEMMs through the Rust adaptive runtime. "
        "They compare the analytic baseline with explicit tuning and persisted decision reuse; "
        "they do not establish an application-wide speedup or compare against cuBLAS.", "",
        "## Method", "",
        "Each measurement runs in a separate process. Baseline uses Disabled policy; first-run "
        "adaptive uses Deferred policy with a fresh decision-cache directory; cached reuses that "
        "trial's successful decision. All use the same inputs, eight-candidate budget, and "
        "32-call tuning threshold. Baseline/adaptive/cached ordering alternates by trial.", "",
        "Every process performs three seconds of identical baseline GPU warmup before measurement. "
        "Context creation, allocations, input upload, this common warmup, correctness readback, "
        "and later paired timing are excluded. Lifecycle totals start before runtime construction "
        "and include preparation, dispatch, synchronization at checkpoints, and the maintenance "
        "call after 32 launches. Any clock ramp and final search performed by adaptive tuning remain included. "
        "Maintenance wall time includes decision-cache writing.", "",
        "The GPU and NVIDIA driver code cache are warm; only the application runtime and adaptive "
        "decision cache are fresh. CUDA compilation-cache behavior follows the recorded environment. "
        "Clocks are not locked and the GPU is not reserved exclusively; existing desktop and "
        "application GPU processes remain running. Counterbalanced process order and paired "
        "measurements reduce drift but do not remove contention or clock noise.", "",
        "Square cases reuse one weight matrix. Decode rotates six weight allocations by default "
        "(192 MiB total). Cache residency depends on the target GPU's L2 size. These are distinct memory workloads. "
        "The tuner uses its own synthetic weight rotation, which can differ from shared-weight "
        "application execution.", "",
        "After the lifecycle measurement, a second Disabled runtime provides the reference for "
        "15 paired rounds of 1,000 launches. Baseline/selected order alternates; each batch ends "
        "with synchronization. These current host microseconds per call include dispatch overhead. "
        "Statistics below aggregate per-process medians, not historical cached timing estimates. "
        "The displayed saving range runs from the lowest trial p10 to the highest trial p90; "
        "it is descriptive dispersion, not a confidence interval.", "",
        "## Measured lifecycle totals", "",
        "Values are medians across fresh processes, in milliseconds. Each column executes the "
        "same number of application calls; tuning scratch launches are additional work.", "",
        "| Shape M×N×K | Calls | Baseline ms | First adaptive ms | Cached ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        for calls, values in summary["totals_s"].items():
            lines.append(f"| {shape_label(summary)} | {int(calls):,} | " + " | ".join(
                f"{values[mode] * 1000:.3f}" for mode in MODES) + " |")
    lines += ["", "## Current steady execution", "",
              "| Shape | Decision | Baseline µs | Selected µs | Paired gain | Saving p10–p90 range µs |",
              "|---|---|---:|---:|---:|---:|"]
    for summary in summaries:
        for mode in ("adaptive", "cached"):
            values = summary["steady"][mode]
            if values:
                lines.append(
                    f"| {shape_label(summary)} | {mode} | {values['baseline_us']:.3f} | "
                    f"{values['selected_us']:.3f} | {values['gain_percent']:.2f}% | "
                    f"{values['saving_p10_min_us']:.3f} to {values['saving_p90_max_us']:.3f} |"
                )
    lines += ["", "## Tuning cost and payoff", "",
              "| Shape | Weight copies / MiB | Fresh decisions | Maintenance ms | Cache hits | Projected break-even total calls |",
              "|---|---:|---|---:|---:|---|"]
    for summary in summaries:
        estimate = summary["break_even"]
        if estimate["status"] == "projected":
            outcome = f"{estimate['total_calls']:,} ({estimate['additional_calls']:,} beyond measured)"
        elif estimate["status"] == "no_kernel_improvement":
            outcome = "No kernel improvement to repay tuning"
        elif estimate["status"] == "insufficient_repeats":
            outcome = "Exploratory: fewer than 3 trials"
        else:
            outcome = "Not resolved: inconsistent promotion or insufficient measured gain"
        tiers = ", ".join(f"{tier}: {count}" for tier, count in summary["adaptive_tiers"].items())
        lines.append(
            f"| {shape_label(summary)} | {summary['weight_copies']} / "
            f"{summary['weight_bytes'] / 2**20:.1f} | {tiers} | "
            f"{summary['maintenance_s'] * 1000:.3f} | "
            f"{summary['cache_hits']}/{summary['repeats']} | {outcome} |"
        )
    lines += ["",
              "Break-even is an extrapolation, not an observed crossover. Starting at the largest "
              "measured call count N, let D be the median first-adaptive lifecycle time minus the "
              "median baseline time, and s the median paired saving in seconds per call. "
              "The projection is N + max(0, ceil(D/s)). It is shown only with at least three "
              "trials, promotion in every trial, positive p10 paired savings in every trial, "
              "and at least 5% aggregate measured gain. Retaining the baseline offers no kernel "
              "improvement to repay tuning; small timing differences for that case are noise. "
              "Mixed promotion decisions and unresolved gains receive no numeric projection.", "",
              "Persisted runs avoid empirical tuning but still regenerate/load code and perform "
              "cache validation. Their total cost is measured directly above; a cache hit alone "
              "does not establish faster execution than the analytic baseline.", "",
              "## Validation and artifacts", ""]
    for summary in summaries:
        lines.append(
            f"- {shape_label(summary)}: maximum sampled CPU-reference relative L2 error "
            f"{summary['maximum_relative_l2']:.3e}; "
            f"fresh candidate counts {summary['candidates_measured']}."
        )
    lines += ["",
              "Correctness checks sample outputs; they are not exhaustive equivalence proofs. "
              "All worker JSON records, stderr logs, paired observations, metadata, source hashes, "
              "and aggregate calculations are preserved alongside this report. "
              "See `raw_records.json`, `summary.json`, `metadata.json`, and `workers/`.", ""]
    return "\n".join(lines)


def parse_shape(value):
    try:
        m, n, k, copies = map(int, value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be M,N,K,WEIGHT_COPIES") from error
    if not (1 <= m <= 16384 and 16 <= n <= 16384 and n % 16 == 0
            and 16 <= k <= 16384 and k % 16 == 0 and 1 <= copies <= 8):
        raise argparse.ArgumentTypeError(
            "M must be 1..16384; N/K multiples of 16 in 16..16384; copies 1..8")
    return m, n, k, copies


def validate_worker(record, mode, shape, calls):
    if (record["mode"] != mode or tuple(record[name] for name in ("m", "n", "k", "weight_copies"))
            != shape or record["calls"] != calls):
        raise ValueError("worker result does not match requested workload")
    if mode == "adaptive" and (record["cache_hit"] or record["tuning_attempts"] != 1):
        raise ValueError("fresh adaptive process did not perform exactly one fresh tuning attempt")
    if mode == "cached" and (not record["cache_hit"] or record["tuning_attempts"] != 0):
        raise ValueError("cached process did not restore a decision without tuning")
    if mode == "baseline" and (record["cache_hit"] or record["tuning_attempts"] != 0):
        raise ValueError("baseline unexpectedly used adaptive tuning/cache")
    if record["tier"] in ("TuningFailed", "RejectedBaseline"):
        raise ValueError(f"worker reported {record['tier']}")
    if mode != "baseline" and len(record["pairs"]) != 15:
        raise ValueError("worker must return 15 paired rounds")
    values = [record[name] for name in ("setup_s", "first32_s", "maintenance_s", "tuning_s",
                                       "baseline_rel_l2", "selected_rel_l2")]
    values += [c["elapsed_s"] for c in record["checkpoints"]]
    values += [p[name] for p in record["pairs"] for name in ("baseline_us", "selected_us")]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("worker contains nonfinite or negative measurements")
    if not any(c["calls"] == calls for c in record["checkpoints"]):
        raise ValueError("worker omitted its final lifecycle checkpoint")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new output directory; existing paths are never overwritten")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--calls", type=int, default=100000)
    parser.add_argument("--shape", type=parse_shape, action="append", dest="shapes",
                        help="M,N,K,WEIGHT_COPIES; repeat to override default four shapes")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.calls < 1000:
        parser.error("--repeats must be positive and --calls must be at least 1000")
    args.shapes = args.shapes or DEFAULT_SHAPES
    if len(set(args.shapes)) != len(args.shapes):
        parser.error("duplicate --shape values are not allowed")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    workers = args.output / "workers"
    workers.mkdir()
    caches = args.output / "caches"
    caches.mkdir()
    run_metadata = metadata(args)
    write_json(args.output / "metadata.json", run_metadata)
    records = []
    try:
        print("Building release benchmark worker...", flush=True)
        with (args.output / "build.log").open("w", encoding="utf-8") as log:
            subprocess.run(["cargo", "build", "--offline", "--release", "--example",
                            "adaptive_jit_bench"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           text=True, check=True)
        binary = ROOT / "target/release/examples/adaptive_jit_bench"
        run_metadata["worker_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        sequence = 0
        for shape in args.shapes:
            name = "x".join(map(str, shape[:3])) + f"-b{shape[3]}"
            for repeat in range(args.repeats):
                cache = caches / f"{name}-r{repeat + 1}"
                # The adaptive worker creates this leaf exclusively; baseline ignores it.
                order = MODES if repeat % 2 == 0 else ("adaptive", "cached", "baseline")
                for mode in order:
                    sequence += 1
                    stem = f"{sequence:03d}-{name}-r{repeat + 1}-{mode}"
                    command = [str(binary), mode, *map(str, shape), str(args.calls), str(cache)]
                    print(f"[{sequence}/{len(args.shapes) * args.repeats * 3}] "
                          f"{name}, trial {repeat + 1}, {mode}", flush=True)
                    started = time.monotonic()
                    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                            check=False)
                    (workers / f"{stem}.stdout.log").write_text(result.stdout, encoding="utf-8")
                    (workers / f"{stem}.stderr.log").write_text(result.stderr, encoding="utf-8")
                    if result.returncode:
                        raise RuntimeError(f"worker failed ({result.returncode}); see workers/{stem}.stderr.log")
                    record = json.loads(result.stdout)
                    validate_worker(record, mode, shape, args.calls)
                    record.update(repeat=repeat + 1, sequence=sequence,
                                  process_wall_s=time.monotonic() - started)
                    write_json(workers / f"{stem}.json", record)
                    records.append(record)
                    write_json(args.output / "raw_records.json", records)
                    print(f"  {record['tier']}, cache_hit={record['cache_hit']}, "
                          f"lifecycle={record['checkpoints'][-1]['elapsed_s']:.3f}s", flush=True)
        summaries = [summarize_shape([
            r for r in records if tuple(r[name] for name in ("m", "n", "k", "weight_copies")) == shape
        ]) for shape in args.shapes]
        write_json(args.output / "summary.json", summaries)
        (args.output / "report.md").write_text(render_report(summaries, records, run_metadata), encoding="utf-8")
        run_metadata["status"] = "complete"
        print(f"Report: {args.output / 'report.md'}", flush=True)
    except Exception as error:
        run_metadata.update(status="failed", error=str(error))
        raise
    finally:
        run_metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        run_metadata["gpu_after"] = gpu_metadata()
        write_json(args.output / "metadata.json", run_metadata)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        sys.exit(1)
