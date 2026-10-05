#!/usr/bin/env python3
"""Compare preserved compiler PTX for complete paged-attention decodes.

CUDA graph batches measure repeated device work without Python gaps between
partial/reduce launches. Ordinary CuPy launch batches separately report completed
host wall time. Both measurements include the combine kernel for split variants.
Compilation, allocation, references, graph capture, and warmup are excluded.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KERNELS = [
    "paged_decode_attention_split_128_32_8_16_4_8",
    "paged_decode_attention_split_128_32_8_16_16_8",
]
DEFAULT_CASES = [(b, length) for b in (1, 8, 32) for length in (1024, 4096)]
TOLERANCE = 0.003
MAX_BUFFER_BYTES = 2 * 1024**3


def read_artifact(directory, name):
    """Parse launch geometry from compiler metadata, not filename assumptions."""
    if not re.fullmatch(r"paged_decode_attention_[A-Za-z0-9_]+", name):
        raise ValueError("kernel must be a paged_decode_attention_ basename")
    path = Path(directory) / (name + ".ptx")
    data = path.read_bytes()
    text = data.decode("utf-8")
    headers = [line for line in text.splitlines() if "[Y PAGED DECODE ATTENTION]" in line]
    if len(headers) != 1:
        raise ValueError(f"{path}: expected one paged-attention launch header")
    header = headers[0]
    fields = {}
    for label in ("head_dim", "q_heads", "kv_heads", "page_size"):
        match = re.search(r"\b" + label + r"=(\d+)\b", header)
        if match is None:
            raise ValueError(f"{path}: missing {label}")
        fields[label] = int(match[1])
    warps = re.search(r"\|\s*(\d+) warps\b", header)
    splits = re.search(r"\|\s*(\d+) splits,\s*(\d+) q heads/CTA\b", header)
    target = re.search(r"^\.target\s+(sm_\d+)\b", text, re.MULTILINE)
    if warps is None or target is None:
        raise ValueError(f"{path}: missing warps or target")
    fields.update(warps=int(warps[1]), splits=int(splits[1]) if splits else 1,
                  split_kernel=splits is not None, target=target[1])
    hd, nqh, nkvh, ps = (fields[x] for x in ("head_dim", "q_heads", "kv_heads", "page_size"))
    if min(hd, nqh, nkvh, ps, fields["warps"], fields["splits"]) <= 0:
        raise ValueError(f"{path}: nonpositive launch parameter")
    if fields["warps"] > 32 or hd % 32 or nqh % nkvh:
        raise ValueError(f"{path}: invalid warp/head geometry")
    if splits and int(splits[2]) != nqh // nkvh:
        raise ValueError(f"{path}: GQA metadata mismatch")
    entries = set(re.findall(r"\.visible\s+\.entry\s+(\w+)\s*\(", text))
    expected = {name, name + "_reduce"} if splits else {name}
    if entries != expected:
        raise ValueError(f"{path}: unexpected entry names {sorted(entries)}")
    return {"path": str(path.resolve()), "name": name, "sha256": hashlib.sha256(data).hexdigest(),
            "geometry": fields}


def read_pair(before, after, name):
    pair = [read_artifact(before, name), read_artifact(after, name)]
    if pair[0]["geometry"] != pair[1]["geometry"]:
        raise ValueError(f"{name}: before/after launch geometry differs")
    return pair


def load_helpers():
    spec = importlib.util.spec_from_file_location(
        "y_paged_attention_reference", ROOT / "tests/benchmark_y_paged_decode_attention.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case_values(value):
    try:
        batch, length = map(int, value.split(","))
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("--case requires BATCH,CONTEXT") from None
    if batch < 1 or batch > 64 or length < 1 or length > 32768:
        raise argparse.ArgumentTypeError("BATCH must be 1..64 and CONTEXT 1..32768")
    return batch, length


class Runner:
    """Own module, output, scratch, and all zero-copy views until work completes."""
    def __init__(self, artifact, inputs, torch_stream, cupy_stream, torch, cp, np):
        self.artifact = artifact
        self.torch = torch
        self.cp = cp
        self.stream = torch_stream
        self.cp_stream = cupy_stream
        q, kc, vc, pt, sl, max_pages = inputs
        geometry = artifact["geometry"]
        self.module = cp.RawModule(path=artifact["path"])
        self.fn = self.module.get_function(artifact["name"])
        self.reduce = (self.module.get_function(artifact["name"] + "_reduce")
                       if geometry["split_kernel"] else None)
        ns, nqh, hd = q.shape
        self.output = torch.empty_like(q)
        self.scratch = []
        if self.reduce is not None:
            self.scratch = [
                torch.empty((ns, nqh, geometry["splits"], hd), dtype=torch.float32, device=q.device),
                torch.empty((ns, nqh, geometry["splits"], 2), dtype=torch.float32, device=q.device),
            ]
        tensors = [q, kc, vc, pt, sl, self.output] + self.scratch
        self.views = [cp.from_dlpack(tensor) for tensor in tensors]
        if any(view.data.ptr != tensor.data_ptr() for view, tensor in zip(self.views, tensors)):
            raise RuntimeError("Torch/CuPy interoperability unexpectedly copied a tensor")
        self.args = tuple(self.views) + (np.int32(max_pages),)
        self.block = (geometry["warps"] * 32, 1, 1)
        self.grid = ((geometry["kv_heads"], ns, geometry["splits"])
                     if self.reduce is not None else (nqh, ns, 1))
        self.reduce_grid = (nqh, ns, 1)
        self.graph = None

    def __call__(self):
        self.fn(self.grid, self.block, self.args, stream=self.cp_stream)
        if self.reduce is not None:
            self.reduce(self.reduce_grid, (32, 1, 1), self.args, stream=self.cp_stream)

    def poison(self):
        self.output.fill_(float("nan"))
        for tensor in self.scratch:
            tensor.fill_(float("nan"))

    def capture(self, calls):
        # RawModule compilation/lookup and launch initialization are complete
        # before capture. The explicit CuPy stream is the capture stream.
        for _ in range(3):
            self()
        self.stream.synchronize()
        self.graph = self.torch.cuda.CUDAGraph()
        with self.torch.cuda.graph(self.graph, stream=self.stream):
            for _ in range(calls):
                self()
        self.graph.replay()
        self.stream.synchronize()


def validate_output(output, reference, lengths, torch):
    got = output.float()
    if not bool(torch.isfinite(got).all().item()):
        raise AssertionError("nonfinite or unwritten attention output")
    difference = (got - reference).double()
    expected = reference.double()
    denominator = float(expected.norm().item())
    error_norm = float(difference.norm().item())
    relative_l2 = error_norm / denominator if denominator else error_norm
    expected_heads = expected.norm(dim=-1)
    head_errors = difference.norm(dim=-1)
    head_relative = torch.where(expected_heads > 0, head_errors / expected_heads.clamp_min(1e-300), head_errors)
    max_head_l2 = float(head_relative.max().item())
    for index, length in enumerate(lengths):
        if length <= 0 and float(got[index].abs().max().item()) != 0.0:
            raise AssertionError(f"zero-length sequence {index} was not zeroed")
    if not math.isfinite(relative_l2) or relative_l2 > TOLERANCE or max_head_l2 > TOLERANCE:
        raise AssertionError(f"reference mismatch: relative L2 {relative_l2}, worst head {max_head_l2}")
    return {"relative_l2": relative_l2, "max_head_relative_l2": max_head_l2,
            "max_absolute_error": float(difference.abs().max().item()), "all_finite": True,
            "outputs_checked": output.numel()}


def validate_pair(runners, reference, lengths, stream, torch, poison=True):
    result = []
    for runner in runners:
        if poison:
            runner.poison()
        runner()
        stream.synchronize()
        result.append(validate_output(runner.output, reference, lengths, torch))
    delta = (runners[0].output.float() - runners[1].output.float()).abs()
    return {"before": result[0], "after": result[1],
            "bitwise_equal": bool(torch.equal(runners[0].output.view(torch.int16), runners[1].output.view(torch.int16))),
            "before_after_max_absolute_difference": float(delta.max().item())}


def measure_pair(runners, calls, rounds, stream, torch):
    for runner in runners:
        runner.capture(calls)
    events = [torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)]
    for event in events:
        event.record(stream)
    stream.synchronize()
    start = time.perf_counter()
    warm_round = 0
    while time.perf_counter() - start < 3.0:
        for slot in range(2):
            runners[(slot + warm_round) % 2].graph.replay()
            stream.synchronize()
        warm_round += 1
    warmup_seconds = time.perf_counter() - start
    raw = []
    for rnd in range(rounds):
        samples = [None, None]
        for slot in range(2):
            side = (rnd + slot) % 2
            runner = runners[side]
            stream.synchronize()
            events[0].record(stream)
            runner.graph.replay()
            events[1].record(stream)
            events[1].synchronize()
            graph_us = events[0].elapsed_time(events[1]) * 1000 / calls
            stream.synchronize()
            started = time.perf_counter_ns()
            for _ in range(calls):
                runner()
            enqueued = time.perf_counter_ns()
            stream.synchronize()
            completed = time.perf_counter_ns()
            samples[side] = {
                "graph_event_us": graph_us,
                "ordinary_enqueue_us": (enqueued - started) / 1000 / calls,
                "ordinary_completed_wall_us": (completed - started) / 1000 / calls,
            }
        raw.append({"round": rnd, "first": "before" if rnd % 2 == 0 else "after",
                    "before": samples[0], "after": samples[1]})
    summaries = {}
    for metric in ("graph_event_us", "ordinary_enqueue_us", "ordinary_completed_wall_us"):
        before = [row["before"][metric] for row in raw]
        after = [row["after"][metric] for row in raw]
        paired = [100 * (b - a) / b for b, a in zip(before, after)]
        summaries[metric] = {
            "before_median": statistics.median(before), "after_median": statistics.median(after),
            "paired_gain_percent_median": statistics.median(paired),
            "paired_gain_percent_min": min(paired), "paired_gain_percent_max": max(paired),
        }
    return {"calls_per_batch": calls, "rounds": rounds, "warmup_seconds": warmup_seconds,
            "graph_nodes_per_batch": [calls * (2 if runner.reduce is not None else 1) for runner in runners],
            "summary": summaries, "raw": raw}


def gpu_metadata():
    query = ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw",
             "--format=csv,noheader"]
    try:
        result = subprocess.run(query, capture_output=True, text=True, timeout=10)
        return {"query": query, "returncode": result.returncode, "stdout": result.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"unavailable": str(error)}


def run_case(artifact_pair, lengths, seed, helpers, stream, cp_stream, timed, args):
    torch, cp, np = helpers.torch, helpers.cp, helpers.np
    shape = artifact_pair[0]["geometry"]
    hd, nqh, nkvh, ps = (shape[key] for key in ("head_dim", "q_heads", "kv_heads", "page_size"))
    batch = len(lengths)
    required_pages = sum(max(1, (length + ps - 1) // ps) for length in lengths)
    num_pages = max(512, required_pages * 2)
    estimated_bytes = 2 * num_pages * ps * nkvh * hd * 2 + batch * nqh * hd * 6
    if estimated_bytes > MAX_BUFFER_BYTES:
        raise ValueError(f"case needs at least {estimated_bytes} bytes, exceeding the 2 GiB benchmark limit")
    inputs = helpers.make_inputs(hd, nqh, nkvh, ps, lengths, num_pages, seed=seed)
    reference = helpers.reference(*inputs[:5], hd, nqh, nkvh, ps)
    runners = [Runner(artifact, inputs, stream, cp_stream, torch, cp, np) for artifact in artifact_pair]
    checks = [validate_pair(runners, reference, lengths, stream, torch) for _ in range(2)]
    record = {"kernel": artifact_pair[0]["name"], "lengths": lengths, "batch": batch,
              "seed": seed, "num_pages": num_pages, "page_table_sha256":
              hashlib.sha256(inputs[3].cpu().numpy().tobytes()).hexdigest(),
              "before_sha256": artifact_pair[0]["sha256"], "after_sha256": artifact_pair[1]["sha256"],
              "geometry": shape, "correctness": checks}
    if timed:
        record["timing"] = measure_pair(runners, args.calls, args.rounds, stream, torch)
        # Validate the actual buffers produced by the final timed work before
        # launching again; then poison and check another complete decode.
        record["timed_output_correctness"] = [validate_output(r.output, reference, lengths, torch) for r in runners]
        record["post_timing_correctness"] = validate_pair(runners, reference, lengths, stream, torch)
    stream.synchronize()
    return record


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kernel", action="append", help="PTX basename without extension; repeat for several variants")
    parser.add_argument("--case", action="append", type=case_values, metavar="BATCH,CONTEXT")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--calls", type=int, default=50)
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args(argv)
    if args.rounds < 2 or args.rounds > 1000 or args.rounds % 2:
        parser.error("--rounds must be an even number in 2..1000")
    if args.calls < 1 or args.calls > 1000:
        parser.error("--calls must be in 1..1000")
    if args.output.exists():
        parser.error("--output already exists; preserve prior measurements with a new filename")
    names = args.kernel or DEFAULT_KERNELS
    if len(set(names)) != len(names):
        parser.error("duplicate --kernel")
    pairs = [read_pair(args.before, args.after, name.removesuffix(".ptx")) for name in names]
    helpers = load_helpers()
    torch, cp = helpers.torch, helpers.cp
    # The independent reference uses F32 accumulation without TF32 reduction.
    torch.set_float32_matmul_precision("highest")
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream(device=0)
    cp_stream = cp.cuda.ExternalStream(stream.cuda_stream)
    report = {
        "schema_version": 1, "status": "running", "argv": sys.argv if argv is None else argv,
        "python": sys.version, "torch": torch.__version__, "cupy": cp.__version__,
        "numpy": helpers.np.__version__, "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0), "before_metadata": gpu_metadata(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_script_sha256": hashlib.sha256((ROOT / "tests/benchmark_y_paged_decode_attention.py").read_bytes()).hexdigest(),
        "artifacts": pairs, "correctness_tolerance": TOLERANCE,
        "notes": ["One reused Q/K/V/page-table set per paired case; cache reuse is part of the workload.",
                  "Graph event batches include every partial and reduce launch, with fixed calls and no Python gaps between nodes.",
                  "Ordinary wall times include Python/CuPy submission overhead and synchronization.",
                  "Positive paired gain means after is faster. Individual process results do not establish repeatability.",
                  "The independent reference checks every output; outputs and split scratch are poisoned before repeated checks."],
        "correctness_cases": [], "timed_cases": [],
    }
    write_json(args.output, report)
    try:
        with torch.cuda.stream(stream), cp_stream:
            for pair in pairs:
                # Cover empty splits, zero rows, sub-warp lengths, exact pages,
                # partial pages, and long ragged sequences with shuffled pages.
                ps = pair[0]["geometry"]["page_size"]
                cases = [([0], 17), ([1, 3, ps - 1, ps, ps + 1], 29),
                         ([0, 1, 7, 63, 64, 65, 200, 33], 41), ([1000, 4095, 2, 2048], 53)]
                for lengths, seed in cases:
                    report["correctness_cases"].append(run_case(pair, lengths, seed, helpers, stream, cp_stream, False, args))
                    write_json(args.output, report)
                for batch, length in args.case or DEFAULT_CASES:
                    result = run_case(pair, [length] * batch, 101, helpers, stream, cp_stream,
                                      not args.correctness_only, args)
                    report["timed_cases"].append(result)
                    write_json(args.output, report)
                    if "timing" in result:
                        summary = result["timing"]["summary"]["graph_event_us"]
                        print(f"{pair[0]['name']} b{batch} ctx{length}: "
                              f"graph {summary['before_median']:.3f} -> {summary['after_median']:.3f} us; "
                              f"paired median {summary['paired_gain_percent_median']:+.2f}%", flush=True)
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["after_metadata"] = gpu_metadata()
        write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
