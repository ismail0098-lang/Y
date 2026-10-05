#!/usr/bin/env python3
"""Compare preserved before/after resident GEMM dispatch workers.

Uses only Python's standard library. Build and preserve both Rust release workers
before running; this script neither builds binaries nor modifies GPU settings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ((1, 16, 16), (256, 256, 256), (1024, 1024, 1024))
VARIANTS = ("before", "after")


def parse_shape(value):
    try:
        shape = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be M,N,K integers") from error
    if (len(shape) != 3 or not all(1 <= dim <= 16384 for dim in shape)
            or shape[1] % 16 != 0 or shape[2] % 16 != 0):
        raise argparse.ArgumentTypeError(
            "shape requires M in 1..16384 and N/K multiples of 16 in 16..16384")
    return shape


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def command_metadata(command):
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                timeout=30, check=False)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def gpu_metadata():
    return command_metadata([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,clocks.sm,"
        "clocks.mem,pstate,temperature.gpu,power.draw,utilization.gpu", "--format=csv",
    ])


def positive_finite(value, name, allow_zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")


def validate_worker(worker, shape, policy):
    if worker["shape"] != list(shape) or worker["policy"] != policy:
        raise ValueError("worker shape or policy does not match its requested arguments")
    for name in ("calls_per_round", "audit_calls", "audit_rust_allocations"):
        if type(worker[name]) is not int or worker[name] < (0 if name.endswith("allocations") else 1):
            raise ValueError(f"invalid worker counter: {name}")
    if worker["calls_per_round"] != 1000 or worker["audit_calls"] != 1000:
        raise ValueError("unexpected worker batch size; update the runner protocol deliberately")
    if not isinstance(worker["rounds"], list) or len(worker["rounds"]) != 41:
        raise ValueError("expected 41 timing rounds from each worker")
    for index, sample in enumerate(worker["rounds"]):
        for metric in ("enqueue_us", "completed_us"):
            positive_finite(sample[metric], f"round {index} {metric}")
        if sample["completed_us"] < sample["enqueue_us"]:
            raise ValueError("completed time is shorter than enqueue time")
    for name in ("initial_rel_l2", "final_rel_l2"):
        positive_finite(worker[name], name, allow_zero=True)
        if worker[name] > 0.002:
            raise ValueError(f"worker failed CPU-reference correctness threshold: {name}")
    if not isinstance(worker["device"], str) or not worker["device"]:
        raise ValueError("missing worker device name")


def summarize(records, trials, shapes=SHAPES):
    summaries = []
    for shape in shapes:
        selected = [record for record in records if record["shape"] == list(shape)]
        paired = []
        for trial in range(trials):
            runs = {variant: [record for record in selected
                              if record["trial"] == trial and record["variant"] == variant]
                    for variant in VARIANTS}
            if any(len(value) != 1 for value in runs.values()):
                raise ValueError(f"shape {shape}, trial {trial} does not have exactly one pair")
            medians = {variant: {
                metric: statistics.median(sample[metric] for sample in runs[variant][0]["rounds"])
                for metric in ("enqueue_us", "completed_us")
            } for variant in VARIANTS}
            paired.append({
                "trial": trial, "process_medians": medians,
                "gain_percent": {metric: 100.0 * (medians["before"][metric]
                                                  - medians["after"][metric])
                                 / medians["before"][metric]
                                 for metric in ("enqueue_us", "completed_us")},
            })
        metrics = {}
        for metric in ("enqueue_us", "completed_us"):
            gains = [pair["gain_percent"][metric] for pair in paired]
            metrics[metric] = {
                "before_median": statistics.median(pair["process_medians"]["before"][metric]
                                                    for pair in paired),
                "after_median": statistics.median(pair["process_medians"]["after"][metric]
                                                   for pair in paired),
                "gain_percent_median": statistics.median(gains),
                "gain_percent_min": min(gains), "gain_percent_max": max(gains),
            }
        summaries.append({
            "shape": list(shape), "trials": trials, "metrics": metrics, "pairs": paired,
            "rust_allocations_per_call": {
                variant: sorted({record["audit_rust_allocations"] / record["audit_calls"]
                                 for record in selected if record["variant"] == variant})
                for variant in VARIANTS},
            "maximum_relative_l2": max(record[name] for record in selected
                                       for name in ("initial_rel_l2", "final_rel_l2")),
        })
    return summaries


def report(summaries, records, metadata):
    devices = ", ".join(sorted({record["device"] for record in records}))
    affinity = metadata["worker_affinity"]
    affinity_text = (f"Every worker is pinned to logical CPU {affinity['cpu']} using taskset. "
                     if affinity["cpu"] is not None else "Workers use the inherited CPU affinity. ")
    lines = [
        "# Resident adaptive GEMM dispatch comparison", "",
        f"Measured on {devices}; {metadata['arguments']['trials']} separate-process pairs per shape, "
        f"using {metadata['arguments']['policy']} policy. Started {metadata['started_utc']}.", "",
        "Each worker prepares exactly one resident shape and reuses its inputs and baseline GEMM kernel. "
        "Compilation, tuning, buffer setup, "
        "correctness readback, and three seconds of warmup are outside timing. No tuning maintenance "
        "or persistent decision cache is used. This measures resident dispatch changes; it does not "
        "establish a better kernel, an adaptive tuning benefit, or application-wide speedup.", "",
        "The preserved worker, PTX-emitter, empirical-tuner, and Cargo.lock hashes match between "
        "binaries. Among captured source hashes, changes are limited to the runtime files "
        "`src/adaptive_jit.rs` and `src/cuda_runtime.rs`; source snapshots and actual binary hashes "
        "are checked before measurement.", "",
        "Each process records 41 batches of 1,000 launches. Tables aggregate process medians; "
        "the 41 rounds are not treated as independent process trials. Before/after order reverses "
        "on alternate trials and shape order rotates. Each pair's gain is "
        "100 × (before − after) / before, using that pair's process medians. The reported minimum "
        "and maximum gains describe observed variation, not a confidence interval.", "",
        affinity_text + "The GPU is not reserved exclusively and clocks are not locked. "
        "Existing desktop/application processes remain running; counterbalanced order cannot "
        "remove contention, scheduling, or clock noise. A positive gain range in this small "
        "synthetic sample still does not guarantee the same result in another workload.", "",
        "Completed time includes final CUDA synchronization. Enqueue time ends after submitting "
        "the batch and may include driver queue blocking; it is not isolated CPU overhead. "
        "Both binaries have the same allocator wrapper, with counting disabled during timing. "
        "The separate untimed audit counts Rust allocations/reallocations and excludes CUDA "
        "driver internal allocations. Even disabled, the allocator wrapper adds a flag load per "
        "allocation in the old path; the measured enqueue gain can include that instrumentation "
        "cost and is not an exact production saving.", "",
        "| Shape M×N×K | Before completed µs/call | After completed µs/call | Median paired gain | Pair gain range |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summaries:
        metric = item["metrics"]["completed_us"]
        label = "×".join(map(str, item["shape"]))
        lines.append(f"| {label} | {metric['before_median']:.4f} | {metric['after_median']:.4f} | "
                     f"{metric['gain_percent_median']:+.2f}% | "
                     f"{metric['gain_percent_min']:+.2f}% to {metric['gain_percent_max']:+.2f}% |")
    lines += ["", "| Shape | Before enqueue µs/call | After enqueue µs/call | Median enqueue gain (range) | Rust allocations/call before → after |",
              "|---|---:|---:|---:|---:|"]
    for item in summaries:
        metric = item["metrics"]["enqueue_us"]
        label = "×".join(map(str, item["shape"]))
        allocations = {variant: ", ".join(f"{value:g}" for value in item["rust_allocations_per_call"][variant])
                       for variant in VARIANTS}
        lines.append(f"| {label} | {metric['before_median']:.4f} | {metric['after_median']:.4f} | "
                     f"{metric['gain_percent_median']:+.2f}% "
                     f"({metric['gain_percent_min']:+.2f}% to {metric['gain_percent_max']:+.2f}%) | "
                     f"{allocations['before']} → {allocations['after']} |")
    maximum = max(item["maximum_relative_l2"] for item in summaries)
    lines += ["", f"Every worker passed its initial and final CPU-reference sample checks; maximum "
              f"relative L2 error was {maximum:.8g} (limit 0.002). Workers sample 64 output positions; "
              "positions repeat for the smallest shape. This is sampled correctness, not exhaustive validation. "
              "Workers also verify launch-count increments and the absence of tuning/cache activity.", "",
              "`raw_records.json` preserves every round, audit, process order, and worker command. "
              "`summary.json` preserves every paired process median. `metadata.json` records binary hashes, "
              "the preserved source-hash snapshots, runner hash, environment, and system/GPU information. "
              "Worker stdout and stderr are retained in `logs/`. Both binary hashes are checked again "
              "after the run. Reproduce with the recorded arguments and preserved binaries from those "
              "source states; rebuilding modified sources does not recreate the original baseline.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--policy", choices=("disabled", "deferred"), default="deferred")
    parser.add_argument("--shape", type=parse_shape, action="append",
                        help="M,N,K; repeat for multiple shapes (default: 1,16,16; 256^3; 1024^3)")
    args = parser.parse_args()
    selected_shapes = tuple(args.shape) if args.shape else SHAPES
    if len(set(selected_shapes)) != len(selected_shapes):
        parser.error("--shape entries must be distinct")
    if args.trials < 1:
        parser.error("--trials must be positive")
    binaries = {variant: getattr(args, variant).resolve() for variant in VARIANTS}
    source_snapshots = {}
    for variant, binary in binaries.items():
        if not binary.is_file() or not os.access(binary, os.X_OK):
            parser.error(f"{variant} worker is not executable: {binary}")
        snapshot = binary.parent / f"{variant}_sources.json"
        if not snapshot.is_file():
            parser.error(f"missing preserved source snapshot: {snapshot}")
        source_snapshots[variant] = {"path": str(snapshot), "sha256": sha256(snapshot),
                                     "contents": json.loads(snapshot.read_text(encoding="utf-8"))}
    binary_hashes = {variant: sha256(binary) for variant, binary in binaries.items()}
    for variant in VARIANTS:
        if source_snapshots[variant]["contents"]["binary_sha256"] != binary_hashes[variant]:
            parser.error(f"{variant} binary does not match its preserved source snapshot")
    before_sources = source_snapshots["before"]["contents"]["sources"]
    after_sources = source_snapshots["after"]["contents"]["sources"]
    required_same = ("examples/adaptive_jit_dispatch_bench.rs", "src/ptx_emitter.rs",
                     "src/empirical_autotune.rs", "Cargo.lock")
    if any(name not in before_sources or name not in after_sources
           or before_sources[name] != after_sources[name] for name in required_same):
        parser.error("worker, emitter, tuner, and dependency hashes must match for this comparison")
    changed_sources = sorted(name for name in before_sources.keys() | after_sources.keys()
                             if before_sources.get(name) != after_sources.get(name))
    if set(changed_sources) - {"src/adaptive_jit.rs", "src/cuda_runtime.rs"}:
        parser.error("captured source changes extend beyond the two dispatch runtime files")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir()
    allowed_cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    taskset = shutil.which("taskset")
    cpu = allowed_cpus[0] if taskset and allowed_cpus else None
    prefix = [taskset, "--cpu-list", str(cpu)] if cpu is not None else []
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {"before": str(binaries["before"]), "after": str(binaries["after"]),
                      "output": str(output), "trials": args.trials, "policy": args.policy,
                      "shapes": selected_shapes},
        "shapes": selected_shapes, "python": sys.version, "platform": platform.platform(),
        "worker_affinity": {"cpu": cpu, "inherited_allowed_cpus": allowed_cpus, "taskset": taskset},
        "binary_sha256": binary_hashes, "changed_captured_sources": changed_sources,
        "preserved_sources": source_snapshots, "runner_sha256": sha256(Path(__file__)),
        "rustc": command_metadata(["rustc", "-Vv"]),
        "cpu": command_metadata(["lscpu"]), "gpu_before": gpu_metadata(),
        "gpu_processes_before": command_metadata(["nvidia-smi"]),
        "environment": {name: os.environ[name] for name in (
            "CUDA_VISIBLE_DEVICES", "CUDA_CACHE_DISABLE", "CUDA_CACHE_PATH", "Y_SMEM_PAD",
            "RUSTFLAGS", "CARGO_BUILD_TARGET") if name in os.environ},
        "gpu_exclusive": False, "gpu_clocks_locked": False,
    }
    write_json(output / "metadata.json", metadata)
    records = []
    try:
        for trial in range(args.trials):
            order = VARIANTS if trial % 2 == 0 else tuple(reversed(VARIANTS))
            offset = trial % len(selected_shapes)
            shapes = selected_shapes[offset:] + selected_shapes[:offset]
            for shape in shapes:
                for pair_position, variant in enumerate(order):
                    stem = f"trial-{trial + 1:02d}-{'x'.join(map(str, shape))}-{variant}"
                    command = prefix + [str(binaries[variant]), *map(str, shape), args.policy]
                    print(f"[{len(records) + 1}/{args.trials * len(selected_shapes) * 2}] {stem}", flush=True)
                    start = time.monotonic()
                    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                            timeout=300, check=False)
                    wall_s = time.monotonic() - start
                    (logs / f"{stem}.stdout").write_text(result.stdout, encoding="utf-8")
                    (logs / f"{stem}.stderr").write_text(result.stderr, encoding="utf-8")
                    if result.returncode != 0:
                        raise RuntimeError(f"{stem} failed with exit {result.returncode}; see {logs}")
                    worker = json.loads(result.stdout)
                    validate_worker(worker, shape, args.policy)
                    records.append({**worker, "variant": variant, "trial": trial,
                                    "pair_position": pair_position, "process_sequence": len(records),
                                    "process_wall_s": wall_s, "command": command})
                    write_json(output / "raw_records.json", records)
        summaries = summarize(records, args.trials, selected_shapes)
        metadata["binary_sha256_after_run"] = {variant: sha256(binary)
                                                for variant, binary in binaries.items()}
        if metadata["binary_sha256"] != metadata["binary_sha256_after_run"]:
            raise RuntimeError("a worker binary changed during the comparison")
        metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["gpu_after"] = gpu_metadata()
        write_json(output / "metadata.json", metadata)
        write_json(output / "summary.json", summaries)
        (output / "report.md").write_text(report(summaries, records, metadata), encoding="utf-8")
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
        metadata["error"] = str(error)
        metadata["failed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "metadata.json", metadata)
        raise
    print(f"Report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
