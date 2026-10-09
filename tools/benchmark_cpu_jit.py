#!/usr/bin/env python3
"""Compare Y's in-process CPU JIT with optimized .NET 8 on identical kernels.

No Python dependencies or NuGet packages. Raw measurements, source hashes,
compiler/runtime versions and a Markdown report are saved for every run.
"""
from __future__ import annotations

import argparse
import bisect
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_KERNELS = ("integer_branch", "recursive_fib", "float_recurrence", "indexed_memory")
KERNELS = ORIGINAL_KERNELS + ("unsigned_mix", "float_dot", "short_circuit", "binary_search")
RUNTIME_KERNELS = KERNELS + ("string_scan", "vec_scan_append")
COPY_KERNELS = RUNTIME_KERNELS + ("vec_dynamic_byte", "vec_dynamic_i64", "string_bulk_append")
HELPER_KERNELS = COPY_KERNELS + ("string_scan_helper", "vec_scan_append_helper")
CONTROLLED_STAGES = ("verify-each", "codegen", "training-tier", "profile-edges", "profile-loop-edges", "final-unroll", "outer-unroll")
UNSIGNED_SEED = 0xFEDCBA9876543210
MASK64 = (1 << 64) - 1
SOURCE_PATHS = (
    "Cargo.toml", "Cargo.lock", "src/cpu_jit.rs", "src/cpu_jit/llvm.rs", "src/cpu_jit/runtime.rs",
    "src/cpu_jit/call.rs", "src/cpu_jit/cache.rs", "src/llvm_emitter.rs",
    "src/cpu_jit/profile.rs", "src/cpu_jit/loops.rs",
    "src/type_checker.rs", "src/lexer.rs", "src/parser.rs", "src/ast.rs", "src/intrinsics.rs",
    "src/linear_tracker.rs", "src/require.rs", "src/c_api.rs", "src/lib.rs",
    "src/cpu_jit_ffi.rs", "src/main.rs", "python/y_lang/cpu_jit.py", "python/y_lang/__init__.py",
    "examples/cpu_jit_bench.rs", "tools/benchmark_cpu_jit.py",
    "benchmarks/cpu_jit/kernels.ysu", "benchmarks/cpu_jit/kernels_original.ysu", "benchmarks/cpu_jit/kernels_runtime.ysu", "benchmarks/cpu_jit/kernels_copies.ysu", "benchmarks/cpu_jit/Program.cs",
    "benchmarks/cpu_jit/kernels_helpers.ysu", "benchmarks/cpu_jit/Program.Helpers.cs",
    "benchmarks/cpu_jit/CSharpBench.csproj", "benchmarks/cpu_jit/NuGet.Config",
)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def metadata_command(command):
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except OSError as error:
        return {"command": command, "error": str(error)}


def memory_hash(data):
    value = 14695981039346656037
    for element in data:
        value = ((value ^ element) * 1099511628211) & ((1 << 64) - 1)
    return str(value)


def initial_memory():
    return [(i * 13 + 5) & 1023 for i in range(65536)]


def reference_integer(n, seed):
    x, total = seed, 0
    for _ in range(n):
        x = x * 48271 % 2147483647
        total += x if (x & 7) < 3 else -x
    return total


@lru_cache(maxsize=None)
def reference_fib(n):
    if n < 2:
        return n
    return reference_fib(n - 1) + reference_fib(n - 2)


def reference_float(n, seed):
    x = seed
    for _ in range(n):
        x = x * 1.000001 + 0.0000003
        if x > 2.0:
            x -= 1.9999
    return x


def reference_memory(data, n, seed):
    total = 0
    for i in range(n):
        index = (i * 17 + seed) & 65535
        value = (data[index] * 3 + 7) & 65535
        data[index] = value
        total += value
    return total


def reference_unsigned(n, seed):
    x = seed
    for _ in range(n):
        x ^= x >> 12
        x = (x ^ (x << 25)) & MASK64
        x ^= x >> 27
        x = x * 2685821657736338717 & MASK64
        x = ((x << 13) | (x >> 51)) & MASK64
    return x


def dot_inputs():
    return ([(i * 17 + 3 & 1023) / 1024.0 for i in range(65536)],
            [(i * 29 + 7 & 1023) / 2048.0 for i in range(65536)])


def reference_dot(a, b, n, seed):
    total = 0.0
    for i in range(n):
        index = (i + seed) & 65535
        total += a[index] * b[index]
    return total


def reference_logical(counters, n, seed):
    def bump_if(value):
        counters[0] += 1
        return (value & 7) < 3
    x, total = seed, 0
    for _ in range(n):
        x = x * 48271 % 2147483647
        if ((x & 1) == 0) and bump_if(x):
            counters[1] += 1
            total += x
        if ((x & 3) == 3) or bump_if(x):
            counters[2] += 1
            total -= x
    return total


def reference_search(data, n, seed):
    x, total = seed, 0
    for _ in range(n):
        x = x * 48271 % 2147483647
        target = x % 196608
        index = bisect.bisect_left(data, target)
        total += index if index < len(data) and data[index] == target else -1
    return total


def reference_runtime_scan(n, seed):
    # Build a Python bytearray, then independently express invalid access as zero.
    values = bytearray(65 if ((i + seed) & 7) < 3 else 122 for i in range(n))
    return 8 * sum(value * ((i & 7) + 1) for i, value in enumerate(values) if (i & 127) > 1)


def signed64(value):
    value &= MASK64
    return value if value < (1 << 63) else value - (1 << 64)


def reference_dynamic_i64(n, seed):
    words = [((i * 7919 + seed) & 2147483647) + ((i * 17 + seed) & 65535) * 4294967296 for i in range(n)]
    return signed64(8 * sum(word * ((i & 7) + 1) for i, word in enumerate(words)) + len(words))


def reference_bulk(n, seed):
    chunk = bytearray(65 if ((i + seed) & 7) < 3 else 122 for i in range(32))
    text = chunk * n
    return 8 * sum(value * ((i & 7) + 1) for i, value in enumerate(text) if (i & 127) > 1) + len(text)


def checked_references(expected, tiny_calls):
    result={}
    for name,n,outputs in (("integer_branch_tiny",1,[reference_integer(1,123+i*17) for i in range(tiny_calls)]),
                           ("integer_branch_large",None,expected["integer_branch"])):
        result[name]={"calls":len(outputs),"outputs_hash":memory_hash(outputs),
                      "outputs_first":outputs[:8],"outputs_last":outputs[-8:]}
        if n is not None: result[name]["n"]=n
    return result


def validate_checked(record, expected):
    results={r["name"]:r for r in record["checked_results"]}
    if set(results)!=set(expected): raise ValueError("unexpected checked-call workload set")
    for name,item in results.items():
        wanted=expected[name]
        if item["calls"]!=wanted["calls"]: raise ValueError("checked-call volume mismatch")
        for key in ("native_outputs_hash","checked_outputs_hash"):
            if item[key]!=wanted["outputs_hash"]: raise ValueError(f"{name}: {key} mismatch")
        for key in ("outputs_first","outputs_last"):
            if item[key]!=wanted[key]: raise ValueError(f"{name}: output excerpt mismatch")
        for key in ("native_ns_per_call","checked_ns_per_call"):
            if not math.isfinite(item[key]) or item[key]<=0: raise ValueError("invalid checked-call duration")


def references(calls, integer_n, fib_n, float_n, memory_n,
               unsigned_n=250000, dot_n=262157, logical_n=250000, search_n=20000, suite="original", string_n=16384, vec_n=16384, dynamic_byte_n=16384, dynamic_i64_n=16384, bulk_n=512):
    data = initial_memory()
    cold_memory = reference_memory(data, memory_n, 123)
    result = {
        "first_integer": reference_integer(integer_n, 123),
        "first_fib": reference_fib(fib_n),
        "first_float": reference_float(float_n, 0.5),
        "first_memory": cold_memory,
        "first_memory_hash": memory_hash(data),
        "integer_branch": [], "recursive_fib": [],
        "float_recurrence": [], "indexed_memory": [],
    }
    data = initial_memory()
    for i in range(calls):
        result["integer_branch"].append(reference_integer(integer_n, 123 + i * 17))
        result["recursive_fib"].append(reference_fib(fib_n + i % 2))
        result["float_recurrence"].append(reference_float(float_n, 0.5 + i * 0.0001))
        result["indexed_memory"].append(reference_memory(data, memory_n, 123 + i * 17))
    result["memory_hash"] = memory_hash(data)
    result["kernels"] = list(ORIGINAL_KERNELS)
    if suite in ("expanded", "runtime", "copies", "helpers"):
        a, b = dot_inputs()
        sorted_data = [i * 3 + 1 for i in range(65536)]
        counters = [0, 0, 0]
        result.update({
            "first_unsigned": reference_unsigned(unsigned_n, UNSIGNED_SEED),
            "first_dot": reference_dot(a, b, dot_n, 123),
            "first_logical": reference_logical(counters, logical_n, 123),
            "first_search": reference_search(sorted_data, search_n, 123),
            "unsigned_mix": [], "float_dot": [], "short_circuit": [], "binary_search": [],
            "kernels": list(KERNELS),
        })
        result["first_counters"] = counters.copy()
        counters = [0, 0, 0]
        for i in range(calls):
            seed = 123 + i * 17
            result["unsigned_mix"].append(reference_unsigned(unsigned_n, (UNSIGNED_SEED + i * 17) & MASK64))
            result["float_dot"].append(reference_dot(a, b, dot_n, seed))
            result["short_circuit"].append(reference_logical(counters, logical_n, seed))
            result["binary_search"].append(reference_search(sorted_data, search_n, seed))
        result["counters"] = counters
    if suite in ("runtime", "copies", "helpers"):
        result.update({"first_string": reference_runtime_scan(string_n, 123),
                       "first_vec": reference_runtime_scan(vec_n, 123),
                       "string_scan": [reference_runtime_scan(string_n, 123 + i * 17) for i in range(calls)],
                       "vec_scan_append": [reference_runtime_scan(vec_n, 123 + i * 17) for i in range(calls)],
                       "kernels": list(RUNTIME_KERNELS)})
    if suite in ("copies", "helpers"):
        result.update({"first_dynamic_byte":reference_runtime_scan(dynamic_byte_n,123)+dynamic_byte_n,
                       "first_dynamic_i64":reference_dynamic_i64(dynamic_i64_n,123),
                       "first_bulk":reference_bulk(bulk_n,123),
                       "vec_dynamic_byte":[reference_runtime_scan(dynamic_byte_n,123+i*17)+dynamic_byte_n for i in range(calls)],
                       "vec_dynamic_i64":[reference_dynamic_i64(dynamic_i64_n,123+i*17) for i in range(calls)],
                       "string_bulk_append":[reference_bulk(bulk_n,123+i*17) for i in range(calls)],
                       "kernels":list(COPY_KERNELS)})
    if suite == "helpers":
        result.update({"first_helper_string":result["first_string"],
                       "first_helper_vec":result["first_vec"],
                       "string_scan_helper":result["string_scan"].copy(),
                       "vec_scan_append_helper":result["vec_scan_append"].copy(),
                       "kernels":list(HELPER_KERNELS)})
    return result


def assert_float(actual, expected, label):
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label}: got {actual!r}, expected {expected!r}")


def validate(record, expected, cold_only=False):
    engine = record["engine"]
    for name in ("first_integer", "first_fib", "first_memory"):
        if int(record[name]) != expected[name]:
            raise ValueError(f"{engine}/{name}: got {record[name]}, expected {expected[name]}")
    assert_float(record["first_float"], expected["first_float"], f"{engine}/first_float")
    if record["first_memory_hash"] != expected["first_memory_hash"]:
        raise ValueError(f"{engine}: first-call memory contents differ from Python reference")
    expanded = "unsigned_mix" in expected
    if expanded:
        for name in ("first_unsigned", "first_logical", "first_search"):
            if int(record[name]) != expected[name]:
                raise ValueError(f"{engine}/{name}: got {record[name]}, expected {expected[name]}")
        assert_float(record["first_dot"], expected["first_dot"], f"{engine}/first_dot")
        if record["first_counters"] != expected["first_counters"]:
            raise ValueError(f"{engine}: first-call short-circuit side effects differ: {record['first_counters']} versus {expected['first_counters']}")
    if "string_scan" in expected:
        for name in ("first_string", "first_vec"):
            if int(record[name]) != expected[name]:
                raise ValueError(f"{engine}/{name}: incorrect runtime object result")
    if "vec_dynamic_byte" in expected:
        for name in ("first_dynamic_byte","first_dynamic_i64","first_bulk"):
            if int(record[name])!=expected[name]: raise ValueError(f"{engine}/{name}: copy workload mismatch")
    if "string_scan_helper" in expected:
        for name in ("first_helper_string","first_helper_vec"):
            if int(record[name])!=expected[name]: raise ValueError(f"{engine}/{name}: helper workload mismatch")
    if cold_only:
        return
    kernels = {item["name"]: item for item in record["results"]}
    if set(kernels) != set(expected.get("kernels", ORIGINAL_KERNELS)):
        raise ValueError(f"{engine}: unexpected kernel set")
    for name, item in kernels.items():
        outputs = item["outputs"]
        if len(outputs) != len(expected[name]):
            raise ValueError(f"{engine}/{name}: wrong number of outputs")
        for i, (actual, wanted) in enumerate(zip(outputs, expected[name])):
            if name in ("float_recurrence", "float_dot"):
                assert_float(actual, wanted, f"{engine}/{name}/{i}")
            elif int(actual) != wanted:
                raise ValueError(f"{engine}/{name}/{i}: got {actual}, expected {wanted}")
        if not math.isfinite(item["ns_per_call"]) or item["ns_per_call"] <= 0:
            raise ValueError(f"{engine}/{name}: invalid duration")
    if kernels["indexed_memory"]["memory_hash"] != expected["memory_hash"]:
        raise ValueError(f"{engine}: final memory contents differ from Python reference")
    if expanded and kernels["short_circuit"]["counters"] != expected["counters"]:
        raise ValueError(f"{engine}: final short-circuit side effects differ from Python reference")


def validate_training(record, expected):
    """Check the instrumented execution as well as the final optimized code."""
    kernels = {item["name"]: item for item in record["training_results"]}
    if set(kernels) != set(expected["kernels"]):
        raise ValueError("profile training ran an unexpected kernel set")
    for name, item in kernels.items():
        if len(item["outputs"]) != len(expected[name]):
            raise ValueError(f"training/{name}: wrong output count")
        for index, (actual, wanted) in enumerate(zip(item["outputs"], expected[name])):
            if name in ("float_recurrence", "float_dot"):
                assert_float(actual, wanted, f"training/{name}/{index}")
            elif int(actual) != wanted:
                raise ValueError(f"training/{name}/{index}: incorrect instrumented result")
    if kernels["indexed_memory"]["memory_hash"] != expected["memory_hash"]:
        raise ValueError("instrumentation changed memory contents")
    if "short_circuit" in kernels and kernels["short_circuit"]["counters"] != expected["counters"]:
        raise ValueError("instrumentation changed short-circuit side effects")
    observed = sum(site["true_count"] + site["false_count"] for site in record["profile_sites"])
    if observed != record["profile_observations"] or observed <= 0 or record["profiled_branches"] <= 0:
        raise ValueError("profile counts or applied branch count are invalid")


COMPILE_PHASES = ("parse", "checks", "lowering", "llvm_setup", "ir_parse", "profile_setup",
                  "verification", "optimization", "ir_capture", "symbol_resolution", "materialization", "other")
OPTIMIZATION_PARTS = ("pipeline_ns", "profile_selection_ns", "other_ns", "total_ns")
MATERIALIZATION_PARTS = ("submission_ns", "first_lookup_ns", "remaining_function_lookups_ns",
                         "profile_lookup_ns", "other_ns", "total_ns")
OBJECT_LOOKUP_PARTS = ("first_lookup_before_object_ns", "first_lookup_after_object_ns")
MATERIALIZATION_COUNTS = ("object_count", "object_bytes", "function_lookup_count", "profile_lookup_count")


def validate_materialization_timings(item, record):
    details = item.get("materialization_timings", {})
    wanted = set(MATERIALIZATION_PARTS + OBJECT_LOOKUP_PARTS + MATERIALIZATION_COUNTS) | {"object_observer_available"}
    if set(details) != wanted:
        raise ValueError("invalid Y materialization timing fields")
    if any(type(details[key]) is not int or details[key] < 0
           for key in MATERIALIZATION_PARTS + MATERIALIZATION_COUNTS):
        raise ValueError("invalid Y materialization duration/count")
    if type(details["object_observer_available"]) is not bool:
        raise ValueError("invalid Y materialization observer flag")
    if sum(details[key] for key in MATERIALIZATION_PARTS[:-1]) != details["total_ns"]:
        raise ValueError("Y materialization breakdown sum differs from total")
    if details["total_ns"] != item["timings"]["materialization_ns"]:
        raise ValueError("Y materialization breakdown total differs from flat phase")
    children = [details[key] for key in OBJECT_LOOKUP_PARTS]
    if any(value is not None for value in children):
        if any(type(value) is not int or value < 0 for value in children):
            raise ValueError("invalid Y first-lookup object-ready timing")
        if not details["object_observer_available"] or details["object_count"] != 1:
            raise ValueError("Y first-lookup split lacks a unique observed object")
        if sum(children) != details["first_lookup_ns"]:
            raise ValueError("Y first-lookup children differ from parent")
    if not details["object_observer_available"] and (details["object_count"] or details["object_bytes"]):
        raise ValueError("Y reports objects without an observer")
    if bool(details["object_count"]) != bool(details["object_bytes"]):
        raise ValueError("Y observed object count/bytes disagree")
    expected_functions = {"original": 8, "expanded": 18, "runtime": 22, "copies": 28, "helpers": 36}[record["suite"]]
    expected_profiles = 1 if item["kind"] == "instrumented" else 0
    if details["function_lookup_count"] != expected_functions or details["profile_lookup_count"] != expected_profiles:
        raise ValueError("Y eager function/profile lookup count differs")
    if not expected_profiles and details["profile_lookup_ns"]:
        raise ValueError("Y uninstrumented compilation reports a profile lookup")


def validate_compile_timings(record):
    compilations=record.get("compilations",[])
    expected=["original"] if record["mode"]=="baseline" else ["instrumented","profiled"]
    if [item.get("kind") for item in compilations]!=expected:
        raise ValueError("Y compilation timing entries mismatch")
    wanted={phase+"_ns" for phase in COMPILE_PHASES}|{"total_ns"}
    for item in compilations:
        ir_level = record["training_opt_level"] if item["kind"] == "instrumented" else record["opt_level"]
        wanted_settings = {
            "ir_opt_level": ir_level, "ir_target_opt_level": ir_level,
            "codegen_opt_level": record["codegen_opt_level"], "codegen_override": record["codegen_override"],
            "training_override": record["training_override"], "verify_each_pass": record["verify_each_pass"],
            "profile_edge_counters": record["profile_edge_counters"],
            "profile_loop_edge_counters": record["profile_loop_edge_counters"],
            "final_loop_unrolling": record["final_loop_unrolling"],
            "ir_loop_unrolling": item["kind"] == "instrumented" or record["final_loop_unrolling"],
            "final_unroll_outer_loops": record["final_unroll_outer_loops"],
            "outer_unroll_policy_active": item["kind"] != "instrumented" and record["outer_unroll_policy_active"],
            "outer_unroll_annotations": 0 if item["kind"] == "instrumented" else record["outer_unroll_annotations"],
            "verification_boundary_policy": "input-and-pass-boundaries",
        }
        settings = item.get("compilation_settings", {})
        if set(settings) != set(wanted_settings) or any(
                type(settings[key]) is not type(value) or settings[key] != value
                for key, value in wanted_settings.items()):
            raise ValueError("Y per-compilation IR/native/verification settings mismatch")
        timings=item["timings"]
        if set(timings)!=wanted or any(type(value) is not int or value<0 for value in timings.values()):
            raise ValueError("invalid Y compilation phase fields")
        if sum(timings[phase+"_ns"] for phase in COMPILE_PHASES)!=timings["total_ns"]:
            raise ValueError("Y compilation phase sum differs from total")
        optimization = item.get("optimization_timings", {})
        if set(optimization) != set(OPTIMIZATION_PARTS) or any(
                type(value) is not int or value < 0 for value in optimization.values()):
            raise ValueError("invalid Y optimization timing fields")
        if sum(optimization[key] for key in OPTIMIZATION_PARTS[:-1]) != optimization["total_ns"]:
            raise ValueError("Y optimization breakdown sum differs from total")
        if optimization["total_ns"] != timings["optimization_ns"]:
            raise ValueError("Y optimization breakdown total differs from flat phase")
        validate_materialization_timings(item, record)
        checks = 3 if item["kind"] == "profiled" and record.get("profiled_branches", 0) > 0 and ir_level > 0 else 2
        if type(item.get("verification_checks")) is not int or item["verification_checks"] != checks:
            raise ValueError("Y input/pass-boundary verification count differs")
        if item["kind"] != "profiled" and optimization["profile_selection_ns"] != 0:
            raise ValueError("unprofiled Y compilation unexpectedly ran profile selection")
    if not math.isclose(compilations[-1]["timings"]["total_ns"],record["compile_ns"],rel_tol=0,abs_tol=1):
        raise ValueError("Y final phase total differs from compile_ns")
    if len(compilations)==2 and not math.isclose(compilations[0]["timings"]["total_ns"],record["instrumented_compile_ns"],rel_tol=0,abs_tol=1):
        raise ValueError("Y instrumentation phase total differs from compile timer")


def validate_configuration(record,stage,mode):
    if record.get("mode")!=mode or record.get("optimization_stage")!=stage:
        raise ValueError("Y mode/stage mismatch; rebuild the benchmark worker")
    modern=stage in ("runtime-copies","adapters","helper-effects","verify-each","codegen","training-tier","profile-edges", "profile-loop-edges", "final-unroll", "outer-unroll")
    query=modern or stage=="runtime-append" or (stage=="runtime" and mode=="optimized")
    mutations=modern or (stage=="runtime-append" and mode=="optimized")
    copies=stage in ("adapters","helper-effects","verify-each","codegen","training-tier","profile-edges", "profile-loop-edges", "final-unroll", "outer-unroll") or (stage=="runtime-copies" and mode=="optimized")
    adapters=stage in ("runtime-copies","helper-effects","verify-each","codegen","training-tier","profile-edges", "profile-loop-edges", "final-unroll", "outer-unroll") or (stage=="adapters" and mode=="optimized")
    helpers=stage in CONTROLLED_STAGES or (stage=="helper-effects" and mode=="optimized")
    verify_each=stage!="verify-each" or mode=="previous"
    loops=not modern and stage!="runtime-append" and not query
    for key,wanted in (("optimize_runtime",query),("optimize_runtime_mutations",mutations),
                       ("optimize_runtime_copies",copies),("optimize_call_adapters",adapters),
                       ("optimize_helper_effects",helpers),("profile_loop_controls",loops),
                       ("verify_each_pass",verify_each),("recognize_rotates",mode!="baseline"),
                       ("profile_edge_counters",stage=="profile-edges" and mode=="optimized"),
                       ("profile_loop_edge_counters",stage=="profile-loop-edges" and mode=="optimized"),
                       ("final_loop_unrolling",stage!="final-unroll" or mode!="optimized"),
                       ("ir_loop_unrolling",stage!="final-unroll" or mode!="optimized"),
                       ("final_unroll_outer_loops",stage!="outer-unroll" or mode!="optimized"),
                       ("outer_unroll_policy_active",stage=="outer-unroll" and mode=="optimized")):
        if record.get(key)!=wanted: raise ValueError(f"Y {key} configuration mismatch")
    if (type(record.get("profile_edge_counters")) is not bool
            or type(record.get("profile_loop_edge_counters")) is not bool):
        raise ValueError("Y profile edge-counter policy has invalid type")
    if (type(record.get("final_loop_unrolling")) is not bool
            or type(record.get("ir_loop_unrolling")) is not bool
            or type(record.get("final_unroll_outer_loops")) is not bool
            or type(record.get("outer_unroll_policy_active")) is not bool):
        raise ValueError("Y final IR loop-unrolling policy has invalid type")
    annotations = record.get("outer_unroll_annotations")
    if type(annotations) is not int or annotations < 0:
        raise ValueError("Y outer-loop annotation count has invalid type/value")
    if (record["outer_unroll_policy_active"] and annotations == 0
            or not record["outer_unroll_policy_active"] and annotations != 0):
        raise ValueError("Y controlled outer-loop annotation count differs from effective policy")
    if record.get("opt_level") != 3 or record.get("verification_boundary_policy") != "input-and-pass-boundaries":
        raise ValueError("Y optimization level or explicit verification policy mismatch")
    override = 2 if stage == "codegen" and mode == "optimized" else None
    if record.get("codegen_override") != override or record.get("codegen_opt_level") != (3 if override is None else override):
        raise ValueError("Y ORC code-generation level/override mismatch")
    if type(record.get("codegen_opt_level")) is not int or (override is not None and type(record.get("codegen_override")) is not int):
        raise ValueError("Y ORC code-generation level has invalid type")
    training_override = 1 if stage in ("profile-edges", "profile-loop-edges", "final-unroll", "outer-unroll") or (stage == "training-tier" and mode == "optimized") else None
    if (record.get("training_override") != training_override
            or record.get("training_opt_level") != (3 if training_override is None else training_override)
            or type(record.get("training_opt_level")) is not int
            or type(record.get("training_override")) is not type(training_override)):
        raise ValueError("Y instrumented IR training tier/override mismatch")
    validate_compile_timings(record)


def controlled_ir_kinds(stage):
    if stage in ("final-unroll", "outer-unroll"):
        return ("instrumented",)
    if stage in ("training-tier", "profile-edges", "profile-loop-edges"):
        return ("kernels",)
    return ("instrumented", "kernels")


def validate_final_unroll_artifacts(output, suffix, stage="final-unroll"):
    if ((output / f"previous-instrumented.{suffix}").read_bytes()
            != (output / f"optimized-instrumented.{suffix}").read_bytes()):
        raise ValueError(f"{stage} changed saved instrumented {suffix} artifact")
    if ((output / f"previous-kernels.{suffix}").read_bytes()
            == (output / f"optimized-kernels.{suffix}").read_bytes()):
        raise ValueError(f"{stage} did not change saved final {suffix} artifact")


def validate_controlled_pair(rows):
    """Verify that the selected compilation policy changes no retained work."""
    variants = {row["variant"]: row for row in rows if row["engine"] == "y"}
    if set(variants) != {"y_previous", "y_optimized"}:
        raise ValueError("controlled comparison is missing a Y arm")
    previous, next_row = variants["y_previous"], variants["y_optimized"]
    for key in ("profile_fingerprint", "profile_sites", "profile_observations", "profiled_branches",
                "profile_selection_optimization", "training_results"):
        if previous[key] != next_row[key]:
            raise ValueError(f"controlled comparison changed {key}")
    for key in previous:
        if key.startswith("first_") and key != "first_call_ns" and previous[key] != next_row[key]:
            raise ValueError(f"controlled comparison changed {key}")
    if "results" in previous:
        if len(previous["results"]) != len(next_row["results"]):
            raise ValueError("controlled comparison changed timed workload count")
        for left, right in zip(previous["results"], next_row["results"]):
            for key in ("name", "outputs", "memory_hash", "counters"):
                if left.get(key) != right.get(key):
                    raise ValueError(f"controlled comparison changed {left['name']}/{key}")


def optimization_timing_summary(records, cold):
    result = {}
    for group, rows in (("warm", records), ("cold", cold)):
        result[group] = {}
        for variant in dict.fromkeys(row.get("variant", row["engine"]) for row in rows if row["engine"] == "y"):
            matches = [row for row in rows if row.get("variant", row["engine"]) == variant]
            result[group][variant] = {}
            for kind in ("original", "instrumented", "profiled"):
                entries = [entry for row in matches for entry in row["compilations"] if entry["kind"] == kind]
                if entries:
                    result[group][variant][kind] = {
                        "timings_ns": {key: distribution([entry["optimization_timings"][key] for entry in entries])
                                       for key in OPTIMIZATION_PARTS},
                        "verification_checks": distribution([entry["verification_checks"] for entry in entries]),
                    }
    return {"optimization_breakdown_ns": result}


def paired_cost_distribution(previous, next_values):
    if len(previous) != len(next_values):
        raise ValueError("paired compilation costs have different sample counts")
    ratios = [left / right if left is not None and right is not None and right > 0 else None
              for left, right in zip(previous, next_values)]
    defined = [ratio for ratio in ratios if ratio is not None]
    previous_defined = [value for value in previous if value is not None]
    next_defined = [value for value in next_values if value is not None]
    saved = [left - right if left is not None and right is not None else None
             for left, right in zip(previous, next_values)]
    saved_defined = [value for value in saved if value is not None]
    left_median = statistics.median(previous_defined) if previous_defined else None
    right_median = statistics.median(next_defined) if next_defined else None
    return {
        "previous_ns": distribution(previous_defined) if previous_defined else None,
        "next_ns": distribution(next_defined) if next_defined else None,
        "previous_values_ns": previous, "next_values_ns": next_values,
        "previous_over_next_ratio_of_medians": left_median / right_median
            if left_median is not None and right_median is not None and right_median > 0 else None,
        "paired_saved_ns": distribution(saved_defined) if saved_defined else None,
        "paired_saved_values_ns": saved,
        "paired_previous_over_next": distribution(defined) if defined else None,
        "paired_ratios": ratios,
    }


def compilation_cost_pair_summary(records, cold):
    if not any(row.get("optimization_stage") in CONTROLLED_STAGES for row in records):
        return {}
    if not {"y_previous", "y_optimized"}.issubset({row.get("variant") for row in records}):
        return {}
    result = {}
    for group, rows in (("warm", records), ("cold", cold)):
        by_variant = {variant: {row["repeat"]: row for row in rows if row["variant"] == variant}
                      for variant in ("y_previous", "y_optimized")}
        if set(by_variant["y_previous"]) != set(by_variant["y_optimized"]):
            raise ValueError("compilation cost comparison has unmatched repeats")
        repeats = sorted(by_variant["y_previous"])
        result[group] = {"repeats": repeats, "compilations": {}, "preparation": {}}
        for kind in ("instrumented", "profiled"):
            entries = {variant: [next(entry for entry in by_variant[variant][repeat]["compilations"]
                                      if entry["kind"] == kind) for repeat in repeats]
                       for variant in by_variant}
            metrics = ("pipeline_ns", "profile_selection_ns", "optimization_other_ns", "optimization_ns",
                       "verification_ns", "materialization_ns", "total_ns", "submission_ns", "first_lookup_ns",
                       "remaining_function_lookups_ns", "profile_lookup_ns", "materialization_other_ns") + OBJECT_LOOKUP_PARTS
            result[group]["compilations"][kind] = {}
            for metric in metrics:
                def values(variant):
                    if metric in ("pipeline_ns", "profile_selection_ns"):
                        return [entry["optimization_timings"][metric] for entry in entries[variant]]
                    if metric == "optimization_other_ns":
                        return [entry["optimization_timings"]["other_ns"] for entry in entries[variant]]
                    if metric == "materialization_other_ns":
                        return [entry["materialization_timings"]["other_ns"] for entry in entries[variant]]
                    if metric in MATERIALIZATION_PARTS[:-2] + OBJECT_LOOKUP_PARTS:
                        return [entry["materialization_timings"][metric] for entry in entries[variant]]
                    return [entry["timings"][metric] for entry in entries[variant]]
                result[group]["compilations"][kind][metric] = paired_cost_distribution(
                    values("y_previous"), values("y_optimized"))
        for metric in ("instrumented_compile_ns", "profile_collection_ns", "profile_snapshot_ns",
                       "optimized_recompile_ns", "prepare_ns"):
            result[group]["preparation"][metric] = paired_cost_distribution(
                [by_variant["y_previous"][repeat][metric] for repeat in repeats],
                [by_variant["y_optimized"][repeat][metric] for repeat in repeats])
    return {"compilation_cost_pairs_ns": result}


def final_unroll_use_summary(records):
    """Model paired warm preparation plus complete heterogeneous workload bundles."""
    stages = {row.get("optimization_stage") for row in records}
    stage = "outer-unroll" if "outer-unroll" in stages else "final-unroll"
    if stage not in stages:
        return {}
    if {"final-unroll", "outer-unroll"} <= stages:
        raise ValueError("total-use comparison mixes final unroll policies")
    rows = [row for row in records if row.get("variant") in ("y_previous", "y_optimized")]
    by_variant = {variant: {row["repeat"]: row for row in rows if row["variant"] == variant}
                  for variant in ("y_previous", "y_optimized")}
    repeats = sorted(by_variant["y_previous"])
    if (not repeats or set(repeats) != set(by_variant["y_optimized"])
            or len(rows) != 2 * len(repeats)):
        raise ValueError(f"{stage} total-use comparison has unmatched or duplicate repeats")
    preparation = {variant: [by_variant[variant][repeat]["prepare_ns"] for repeat in repeats]
                   for variant in by_variant}
    bundles = {variant: [] for variant in by_variant}
    for repeat in repeats:
        for variant in by_variant:
            results = by_variant[variant][repeat]["results"]
            if ([entry["name"] for entry in results] != list(HELPER_KERNELS)
                    or any(not math.isfinite(entry["ns_per_call"]) or entry["ns_per_call"] <= 0
                           for entry in results)):
                raise ValueError(f"{stage} total-use comparison lacks a full finite helpers bundle")
            bundles[variant].append(sum(entry["ns_per_call"] for entry in results))
    saved = [a - b for a, b in zip(preparation["y_previous"], preparation["y_optimized"])]
    slowdown = [b - a for a, b in zip(bundles["y_previous"], bundles["y_optimized"])]
    signed_crossovers = [a / b if b != 0 else None for a, b in zip(saved, slowdown)]
    applicable = [a > 0 and b > 0 for a, b in zip(saved, slowdown)]
    finite_positive = [value for value, applies in zip(signed_crossovers, applicable) if applies]
    counts = (1, 12, 32, 100, 1000)
    result = {
        "bundle_definition": "One call to each of the fifteen heterogeneous synthetic helpers workloads.",
        "repeats": repeats, "bundle_counts": list(counts),
        "preparation": paired_cost_distribution(preparation["y_previous"], preparation["y_optimized"]),
        "native_bundle": paired_cost_distribution(bundles["y_previous"], bundles["y_optimized"]),
        "total_use": {str(count): paired_cost_distribution(
            [a + count * b for a, b in zip(preparation["y_previous"], bundles["y_previous"])],
            [a + count * b for a, b in zip(preparation["y_optimized"], bundles["y_optimized"])])
            for count in counts},
        "break_even": {
            "signed_preparation_saved_ns": saved, "signed_native_slowdown_ns": slowdown,
            "signed_crossover_bundles": signed_crossovers,
            "initial_saving_and_native_slowdown": applicable,
            "positive_applicable_crossover_distribution": distribution(finite_positive)
                if finite_positive else None,
        },
    }
    key = "outer_unroll_use_pairs_ns" if stage == "outer-unroll" else "final_unroll_use_pairs_ns"
    return {key: result}


def final_unroll_use_report(summary):
    key = "outer_unroll_use_pairs_ns" if "outer_unroll_use_pairs_ns" in summary else "final_unroll_use_pairs_ns"
    if key not in summary:
        return ""
    values = summary[key]
    lines = ["", "Paired modeled total use", "",
             "For each warm process pair, total use is measured preparation plus N times the sum "
             "of per-kernel native ns/call. One bundle executes each of the fifteen heterogeneous "
             "synthetic workloads once. These are extrapolated models, not newly timed N-bundle runs; "
             "first-call, standardized warmups and surrounding host work are outside this model. "
             "A final-compilation saving does not guarantee a total-use win. Positive signed saving "
             "favors next Y; every negative sample remains retained.", "",
             "| Bundles N | Previous median ms | Next median ms | Paired saved median ms | Paired saved range ms | Next gains |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for count in values["bundle_counts"]:
        pair = values["total_use"][str(count)]
        saved = pair["paired_saved_ns"]
        gains = sum(value > 0 for value in pair["paired_saved_values_ns"])
        lines.append(f"| {count} | {pair['previous_ns']['median']/1e6:.3f} | "
                     f"{pair['next_ns']['median']/1e6:.3f} | {saved['median']/1e6:.3f} | "
                     f"{saved['min']/1e6:.3f}–{saved['max']/1e6:.3f} | {gains}/{len(values['repeats'])} |")
    lines += ["", "Signed crossover is (previous preparation minus next preparation) divided by "
              "(next native bundle minus previous native bundle). When both are positive, next Y "
              "wins below that bundle count and loses above it. Negative crossovers, faster next "
              "native bundles and zero native differences retain their signed values or null with "
              "applicability recorded; they are not positive break-even limits.", "",
              "| Pair | Preparation saved ms | Native bundle slowdown ms | Signed crossover bundles | Initial saving with slower native bundle |",
              "| --- | ---: | ---: | ---: | --- |"]
    if key == "outer_unroll_use_pairs_ns":
        lines[-2:-2] = ["When preparation is slower and the next native bundle is faster, a positive "
            "signed crossover instead means next Y loses below it and wins above it. Faster preparation "
            "and a faster native bundle win at every positive N; slower preparation and a slower native "
            "bundle lose at every positive N. Zero native difference leaves only the preparation delta. "
            "The initial-saving column applies only to preparation saving with native slowdown.", ""]
    cross = values["break_even"]
    for i, repeat in enumerate(values["repeats"]):
        value = cross["signed_crossover_bundles"][i]
        text = f"{value:.3f}" if value is not None else "—"
        lines.append(f"| {repeat} | {cross['signed_preparation_saved_ns'][i]/1e6:.6f} | "
                     f"{cross['signed_native_slowdown_ns'][i]/1e6:.6f} | {text} | "
                     f"{cross['initial_saving_and_native_slowdown'][i]} |")
    lines.append("")
    return "\n".join(lines)


def materialization_timing_summary(records, cold):
    result = {}
    for group, rows in (("warm", records), ("cold", cold)):
        result[group] = {}
        for variant in dict.fromkeys(row.get("variant", row["engine"]) for row in rows if row["engine"] == "y"):
            matches = [row for row in rows if row.get("variant", row["engine"]) == variant]
            result[group][variant] = {}
            for kind in ("original", "instrumented", "profiled"):
                entries = [entry["materialization_timings"] for row in matches
                           for entry in row["compilations"] if entry["kind"] == kind]
                if entries:
                    optional = {key: [entry[key] for entry in entries] for key in OBJECT_LOOKUP_PARTS}
                    result[group][variant][kind] = {
                        "timings_ns": {key: distribution([entry[key] for entry in entries])
                                       for key in MATERIALIZATION_PARTS},
                        "first_lookup_children_ns": {key: {
                            "values": values,
                            "defined_count": sum(value is not None for value in values),
                            "distribution": distribution([value for value in values if value is not None])
                                if any(value is not None for value in values) else None,
                        } for key, values in optional.items()},
                        "counts": {key: distribution([entry[key] for entry in entries])
                                   for key in MATERIALIZATION_COUNTS},
                        "object_observer_available": [entry["object_observer_available"] for entry in entries],
                    }
    return {"materialization_breakdown_ns": result}


def compilation_phase_summary(records,cold):
    if not any("compilations" in row for row in records):return {}
    result={}
    for group,rows in (("warm",records),("cold",cold)):
        result[group]={}
        for variant in dict.fromkeys(r.get("variant",r["engine"]) for r in rows if r["engine"]=="y"):
            matches=[r for r in rows if r.get("variant",r["engine"])==variant]
            result[group][variant]={}
            for kind in ("original","instrumented","profiled"):
                timings=[item["timings"] for r in matches for item in r["compilations"] if item["kind"]==kind]
                if timings:
                    result[group][variant][kind]={key:distribution([item[key] for item in timings]) for key in timings[0]}
    return {"compilation_phases_ns":result}


def helper_pair_summary(records):
    if not any(k["name"]=="string_scan_helper" for k in records[0]["results"]):return {}
    result={}
    for direct,helper in (("string_scan","string_scan_helper"),("vec_scan_append","vec_scan_append_helper")):
        result[helper]={"direct":direct,"variants":{}}
        for variant in dict.fromkeys(r.get("variant",r["engine"]) for r in records):
            rows=[r for r in records if r.get("variant",r["engine"])==variant]
            times=[{k["name"]:k["ns_per_call"] for k in r["results"]} for r in rows]
            result[helper]["variants"][variant]={
                "helper_over_direct_ratio_of_medians":statistics.median(t[helper] for t in times)/statistics.median(t[direct] for t in times),
                "paired_helper_over_direct":distribution([t[helper]/t[direct] for t in times])}
    return {"helper_pairs":result}


def helper_pair_report(summary):
    if "helper_pairs" not in summary:return ""
    variants=list(next(iter(summary["helper_pairs"].values()))["variants"])
    lines=["", "Direct/helper comparison", "",
        "| Helper kernel / direct kernel | "+" | ".join(variants)+" |",
        "| --- | "+" | ".join("---:" for _ in variants)+" |"]
    for name,pair in summary["helper_pairs"].items():
        lines.append("| "+name+" / "+pair["direct"]+" | "+" | ".join(f"{pair['variants'][v]['helper_over_direct_ratio_of_medians']:.2f}x" for v in variants)+" |")
    lines += ["", "Values divide helper median time by direct median time; above one means the helper "
        "form is slower by that statistic. Raw within-process paired ratios remain in the summary. "
        "Kernel batches run in a fixed order, with direct kernels before helper kernels, so allocator, "
        "cache and frequency effects can contribute to these descriptive within-worker comparisons.", ""]
    return "\n".join(lines)


def compilation_phase_report(summary):
    if "compilation_phases_ns" not in summary:return ""
    lines=["", "Compilation phase medians", "",
        "Every original, instrumented and profiled Y compilation records disjoint phase intervals. "
        "Other is the residual; each sample's phases sum exactly to its total. Separate medians need not sum "
        "to the median total. Verification includes input and explicit pass-boundary checks; optimization "
        "includes optional LLVM VerifyEach work inside its two pipeline calls. Materialization combines eager "
        "ORC code generation, linking and entrypoint lookup. The nested first-lookup split observes "
        "compiled-object handoff; it is not an exclusive backend or linker timer.", ""]
    for group,variants in summary["compilation_phases_ns"].items():
        columns=[(variant,kind,phases) for variant,kinds in variants.items() for kind,phases in kinds.items()]
        lines += [group.capitalize()+" processes (ms)", "",
            "| Phase | "+" | ".join(variant+" "+kind for variant,kind,_ in columns)+" |",
            "| --- | "+" | ".join("---:" for _ in columns)+" |"]
        for phase in COMPILE_PHASES+("total",):
            lines.append("| "+phase+" | "+" | ".join(f"{phases[phase+'_ns']['median']/1e6:.3f}" for _,_,phases in columns)+" |")
        lines.append("")
    return "\n".join(lines)


def optimization_timing_report(summary):
    if "optimization_breakdown_ns" not in summary:
        return ""
    lines = ["", "Optimization breakdown medians", "",
             "These intervals are nested inside the flat optimization phase. Their sample totals agree "
             "exactly with that phase; do not add these rows to the flat compilation phases again. "
             "Explicit module checks after the pipelines are charged to verification. The default pipeline "
             "and profile-selection timers include intermediate VerifyEach work only when enabled.", ""]
    for group, variants in summary["optimization_breakdown_ns"].items():
        columns = [(variant, kind, entry) for variant, kinds in variants.items() for kind, entry in kinds.items()]
        lines += [group.capitalize() + " processes (ms)", "",
                  "| Interval | " + " | ".join(variant + " " + kind for variant, kind, _ in columns) + " |",
                  "| --- | " + " | ".join("---:" for _ in columns) + " |"]
        for key in OPTIMIZATION_PARTS:
            lines.append("| " + key + " | " + " | ".join(
                f"{entry['timings_ns'][key]['median'] / 1e6:.3f}" for _, _, entry in columns) + " |")
        lines.append("| Explicit verification checks | " + " | ".join(
            str(entry["verification_checks"]["median"]) for _, _, entry in columns) + " |")
        lines.append("")
    return "\n".join(lines)


def materialization_timing_report(summary):
    if "materialization_breakdown_ns" not in summary:
        return ""
    lines = ["", "Materialization breakdown medians", "",
             "Primary intervals partition each sample's flat materialization phase exactly. First-lookup "
             "before/after-object rows are nested children of first_lookup_ns; do not add them to the "
             "primary total again. The passive object callback leaves bytes and ownership unchanged. "
             "Before-object time includes ORC/IR compile-layer work and native emission; after-object "
             "time includes callback overhead, linking, symbol resolution and lookup completion. These "
             "wall intervals are not exclusive code-generation/linking timers. Null children mean that "
             "the optional observer or an unambiguous first-lookup boundary was unavailable. Object-file "
             "bytes are not allocated executable-memory bytes. Separate medians need not add.", ""]
    for group, variants in summary["materialization_breakdown_ns"].items():
        columns = [(variant, kind, entry) for variant, kinds in variants.items() for kind, entry in kinds.items()]
        lines += [group.capitalize() + " processes (ms)", "",
                  "| Interval | " + " | ".join(variant + " " + kind for variant, kind, _ in columns) + " |",
                  "| --- | " + " | ".join("---:" for _ in columns) + " |"]
        for key in MATERIALIZATION_PARTS:
            lines.append("| " + key + " | " + " | ".join(
                f"{entry['timings_ns'][key]['median'] / 1e6:.3f}" for _, _, entry in columns) + " |")
        for key in OBJECT_LOOKUP_PARTS:
            values = [entry["first_lookup_children_ns"][key]["distribution"] for _, _, entry in columns]
            lines.append("| " + key + " (nested) | " + " | ".join(
                f"{value['median'] / 1e6:.3f}" if value is not None else "—" for value in values) + " |")
        for key in MATERIALIZATION_COUNTS:
            lines.append("| " + key + " | " + " | ".join(
                str(entry["counts"][key]["median"]) for _, _, entry in columns) + " |")
        lines.append("| Observer available samples | " + " | ".join(
            f"{sum(entry['object_observer_available'])}/{len(entry['object_observer_available'])}"
            for _, _, entry in columns) + " |")
        lines.append("")
    return "\n".join(lines)


def compilation_cost_pair_report(summary, stage="verify-each"):
    if "compilation_cost_pairs_ns" not in summary:
        return ""
    policy = ("Previous Y inherits LLVM defaults for all final IR loops; next Y annotates eligible "
              "original natural loops strictly containing another natural loop with llvm.loop.unroll.disable. "
              "Innermost and conservatively skipped loops keep their default metadata policy. Global "
              "final_loop_unrolling remains true in both arms. Both keep temporary training IR1 at LLVM "
              "defaults, final IR/native3, VerifyEach, explicit boundary checks and eager materialization. "
              "Both atomic edge flags remain false. Instrumented IR/assembly and exact profiles/output "
              "streams must match; at least one final IR and reconstructed assembly difference is required. "
              "Only final_unroll_outer_loops changes, true/false; the production default remains true. "
              "outer_unroll_policy_active records whether the policy is enabled for that compilation; "
              "outer_unroll_annotations counts original loops annotated, not dynamic transformations "
              "avoided. Instrument compilation has inactive policy and zero annotations. Compilation "
              "savings must be weighed against native changes in paired total-use models. "
              if stage == "outer-unroll" else
              "Previous Y inherits LLVM default loop-unrolling tuning in final O3 IR pipelines; next Y "
              "disables that tuning for final ordinary/profile-use IR. Both keep temporary training IR1 "
              "at LLVM defaults, native3, VerifyEach, explicit boundary checks and eager materialization. "
              "Both atomic edge flags remain false. Instrumented IR and exact profiles/output streams "
              "must match, while final IR and reconstructed assembly are expected to differ. Only "
              "final_loop_unrolling changes; true means inherited LLVM default, not a forced unroll. "
              "The production default remains true. Compilation savings must be weighed against final "
              "execution changes in paired total-use models. "
              if stage == "final-unroll" else
              "Previous Y selects the counter address before every original conditional branch; next Y "
              "uses fixed-address increments only on predictable original natural-loop iteration "
              "controls without PHI successors. Other branches retain selected-address increments. "
              "Both request training IR level1, native level3, final IR/native level3, VerifyEach "
              "and explicit boundary checks. Each observed outcome still performs one aligned "
              "monotonic atomic increment; there is no sampling or batching. Profiles, retained "
              "outputs and saved final IR must match. Only profile_loop_edge_counters changes; "
              "profile_edge_counters remains false in both arms. The production default remains false. "
              if stage == "profile-loop-edges" else
              "Previous Y selects the counter address before each original conditional branch; next Y "
              "increments a fixed counter address on the corresponding edge. Both request training IR "
              "level1, native level3, final IR/native level3, VerifyEach and explicit boundary checks. "
              "Each branch observation still performs one aligned monotonic atomic increment; there is "
              "no sampling or batching. Profiles, retained outputs and saved final IR must match. "
              "Only profile_edge_counters changes, with the production default remaining false. "
              "Preparation includes both compilation sessions and the measured training execution. "
              if stage == "profile-edges" else
              "Previous Y retains training IR level3; next Y requests training IR level1. Both use native "
              "code-generation level3, intermediate VerifyEach and explicit boundary checks during training. "
              "Both final compilations retain IR/native level3 and identical actual profiles and final IR. "
              "Only the instrumented IR pipeline/target tier changes; the production training default "
              "remains inherited. Preparation includes both compilation sessions and measured training "
              "execution, which can regress even when the initial compiler does less work. "
              if stage == "training-tier" else
              "Previous Y uses inherited ORC code-generation level 3 (Aggressive); next Y explicitly "
              "uses level 2 (Default). Both preserve the IR default<O3> pipeline and its separate level-3 "
              "optimization target, intermediate VerifyEach and explicit boundary checks, runtime/helper "
              "lowering, adapters, host ISA, measured profiles and eager materialization. Only the ORC "
              "code-generation level changes; the production default continues to inherit opt_level. "
              "This observed policy comparison includes host/allocator effects and is not an exclusive "
              "backend timer. Object-ready lookup children retain their narrower wall-time scope. "
              if stage == "codegen" else
             "Previous Y enables optional intermediate VerifyEach diagnosis; next Y keeps explicit input "
             "and pass-boundary checks while disabling those intermediate checks. Both retain the default "
             "O3 pipeline, profile selection and eager materialization. The on/off difference measures the "
             "observed cost of this policy change, including host and allocator effects; it is not an "
             "exclusive verifier timer. ")
    lines = ["", "Controlled compilation costs", "", policy +
             "Positive paired saved time means next Y is faster. All negative differences are retained. "
             "Zero denominators or unavailable intervals have undefined ratios, recorded as null with every paired "
             "sample retained. Unavailable object-ready children also retain null signed differences.", ""]
    for group, entry in summary["compilation_cost_pairs_ns"].items():
        lines += [group.capitalize() + " processes", "",
                  "| Kind / interval | Previous median ms | Next median ms | Previous / next medians | Paired saved median ms | Paired saved range ms |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        rows = [(kind + " / " + name, values) for kind, intervals in entry["compilations"].items()
                for name, values in intervals.items()]
        rows += [("preparation / " + name, values) for name, values in entry["preparation"].items()]
        for name, values in rows:
            ratio = values["previous_over_next_ratio_of_medians"]
            saved = values["paired_saved_ns"]
            ratio_text = f"{ratio:.3f}x" if ratio is not None else "—"
            previous_text = f"{values['previous_ns']['median'] / 1e6:.3f}" if values["previous_ns"] is not None else "—"
            next_text = f"{values['next_ns']['median'] / 1e6:.3f}" if values["next_ns"] is not None else "—"
            saved_text = f"{saved['median'] / 1e6:.3f}" if saved is not None else "—"
            range_text = f"{saved['min'] / 1e6:.3f}–{saved['max'] / 1e6:.3f}" if saved is not None else "—"
            lines.append(f"| {name} | {previous_text} | {next_text} | {ratio_text} | {saved_text} | {range_text} |")
        lines.append("")
    return "\n".join(lines)


def execute(command, env, destination):
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    wall_ms = (time.perf_counter() - started) * 1000
    destination.with_suffix(".stdout").write_text(result.stdout, encoding="utf-8")
    destination.with_suffix(".stderr").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"worker failed ({result.returncode}): {' '.join(command)}\n{result.stderr}")
    try:
        record = json.loads(result.stdout.strip())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"worker did not produce JSON; see {destination.with_suffix('.stdout')}") from error
    record["process_wall_ms"] = wall_ms
    record["command"] = command
    write_json(destination, record)
    return record


def distribution(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values),
            "samples": values}


def summarize(records, cold):
    result = {"kernels": {}, "cold": {}}
    for name in [item["name"] for item in records[0]["results"]]:
        samples = {}
        for engine in ("y", "csharp"):
            samples[engine] = [next(k["ns_per_call"] for k in r["results"] if k["name"] == name)
                               for r in records if r["engine"] == engine]
        paired = [cs / y for cs, y in zip(samples["csharp"], samples["y"])]
        y, cs = statistics.median(samples["y"]), statistics.median(samples["csharp"])
        result["kernels"][name] = {"y_ns": distribution(samples["y"]),
                                    "csharp_ns": distribution(samples["csharp"]),
                                    "csharp_over_y": cs / y,
                                    "paired_csharp_over_y": distribution(paired)}
    for engine in ("y", "csharp"):
        runs = [r for r in cold if r["engine"] == engine]
        result["cold"][engine] = {
            "compile_ms": distribution([r["compile_ns"] / 1e6 for r in runs]),
            "first_call_ms": distribution([r["first_call_ns"] / 1e6 for r in runs]),
            "launch_to_json_ms": distribution([r["process_wall_ms"] for r in runs]),
        }
    result.update(compilation_phase_summary(records,cold))
    result.update(optimization_timing_summary(records,cold))
    result.update(materialization_timing_summary(records,cold))
    result.update(helper_pair_summary(records))
    return result


def summarize_optimized(records, cold):
    variants = ("y_previous" if any(r["variant"] == "y_previous" for r in records) else "y_baseline", "y_optimized", "csharp")
    baseline = variants[0]
    result = {"kernels": {}, "cold": {}}
    for name in [item["name"] for item in records[0]["results"]]:
        samples = {variant: [next(k["ns_per_call"] for k in r["results"] if k["name"] == name)
                             for r in records if r["variant"] == variant] for variant in variants}
        medians = {variant: statistics.median(values) for variant, values in samples.items()}
        result["kernels"][name] = {
            "variants_ns": {variant: distribution(values) for variant, values in samples.items()},
            "baseline_over_optimized": medians[baseline] / medians["y_optimized"],
            "csharp_over_optimized": medians["csharp"] / medians["y_optimized"],
            "paired_baseline_over_optimized": distribution([base / opt for base, opt in
                zip(samples[baseline], samples["y_optimized"])]),
            "paired_csharp_over_optimized": distribution([cs / opt for cs, opt in
                zip(samples["csharp"], samples["y_optimized"])]),
        }
    for variant in variants:
        runs = [r for r in cold if r["variant"] == variant]
        keys = ("compile_ns", "prepare_ns", "instrumented_compile_ns", "profile_collection_ns",
                "profile_snapshot_ns", "optimized_recompile_ns", "first_call_ns", "process_wall_ms")
        result["cold"][variant] = {
            key: distribution([r[key] if key == "process_wall_ms" else r[key] / 1e6 for r in runs])
            for key in keys if all(key in r for r in runs)
        }
    if any("checked_results" in r for r in records):
        result["checked_calls"]={}
        for name in ("integer_branch_tiny","integer_branch_large"):
            data={}
            for variant in variants[:2]:
                rows=[next(k for k in r["checked_results"] if k["name"]==name) for r in records if r["variant"]==variant]
                data[variant]={key:distribution([row[key] for row in rows]) for key in ("native_ns_per_call","checked_ns_per_call")}
            result["checked_calls"][name]=data
    result.update(compilation_phase_summary(records,cold))
    result.update(optimization_timing_summary(records,cold))
    result.update(materialization_timing_summary(records,cold))
    result.update(compilation_cost_pair_summary(records,cold))
    result.update(helper_pair_summary(records))
    return result


def optimized_report(summary, metadata):
    args = metadata["arguments"]
    lines = ["# Y CPU JIT optimization comparison with C#", "",
        f"Measured {metadata['started_utc']} on {metadata['cpu_model']}; pinned to {metadata['affinity']}.",
        f"LLVM {metadata['jit_llvm_version']}, {metadata['csharp_runtime']}. "
        f"{args['repeats']} independent interleaved triples; {args['calls']} calls per timed batch "
        f"after {args['warmup']} standardized warmups per kernel.", "",
        "Baseline Y uses the final compiler with rotate recognition disabled and no branch profiles. "
        "Optimized Y enables rotate recognition, compiles instrumented code, executes "
        f"{args['profile_warmups']} training calls per kernel, snapshots actual branch counts, then "
        "recompiles the same source with those measured profiles. Its timed code contains no profiling probes.", "",
        "| Kernel | Baseline Y ms/call | Optimized Y ms/call | C# ms/call | Y improvement | C# / optimized Y | Paired C# / Y range |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in summary["kernels"].items():
        values = item["variants_ns"]
        paired = item["paired_csharp_over_optimized"]
        lines.append(f"| {name} | {values['y_baseline']['median'] / 1e6:.4f} | "
                     f"{values['y_optimized']['median'] / 1e6:.4f} | {values['csharp']['median'] / 1e6:.4f} | "
                     f"{item['baseline_over_optimized']:.2f}x | {item['csharp_over_optimized']:.2f}x | "
                     f"{paired['min']:.2f}–{paired['max']:.2f}x |")
    lines += ["", "Ratios above 1 favor optimized Y. Paired ranges are descriptive min/max values, "
        "not confidence intervals. Small differences should be read as near parity.", "",
        "| Optimized Y preparation stage | Median ms (small cold inputs) |",
        "| --- | ---: |"]
    for label, key in (("Instrumented source compilation", "instrumented_compile_ns"),
                       ("Measured profile collection", "profile_collection_ns"),
                       ("Profile snapshot", "profile_snapshot_ns"),
                       ("Optimized recompilation", "optimized_recompile_ns"),
                       ("Total preparation", "prepare_ns")):
        lines.append(f"| {label} | {summary['cold']['y_optimized'][key]['median']:.3f} |")
    lines += ["", f"Baseline Y source-to-native compilation: {summary['cold']['y_baseline']['compile_ns']['median']:.3f} ms. "
        f"C# prebuilt IL preparation: {summary['cold']['csharp']['compile_ns']['median']:.3f} ms. "
        "These have different starting points; C# source-to-IL building is excluded.", "",
        "Training is explicit work and its cost is excluded from steady-state kernel timers. "
        "Every warm-run record also retains its full-size training cost; the cold table uses small inputs. "
        "Training uses the same input distribution as this benchmark. Profiles are observed branch "
        "frequencies, which guide code layout and selection without proving branches unreachable.", "",
        "All baseline, instrumented-training and optimized outputs match independent Python implementations. "
        "Memory writes are checked with a full-array hash; all three short-circuit counters are checked exactly. "
        "Raw branch counts, fingerprints and the number of applied profile sites are preserved. "
        "Both baseline and optimized LLVM IR are saved.", "",
        "C# uses Release .NET 8 code with tiering and ReadyToRun disabled, preserving its optimized "
        "non-PGO configuration. Kernel entry methods use NoInlining. Native memory indexing is unchecked "
        "in both languages. Timers include host calls and result storage; allocations, hashing and formatting "
        "are outside timed batches. Each optimized process starts fresh without the optional compilation cache.", "",
        "All three variants are interleaved with balanced rotating orders and run sequentially. "
        "The standardized warmup and timed inputs match across variants. Other host activity and CPU frequency "
        "can affect measurements. These synthetic kernels do not establish a general language speed ranking.", "",
        f"Inputs: integer={args['integer_n']}, Fibonacci={args['fib_n']}/{args['fib_n'] + 1}, "
        f"recurrence={args['float_n']}, memory={args['memory_n']}, unsigned={args['unsigned_n']}, "
        f"dot={args['dot_n']}, logical={args['logical_n']}, search={args['search_n']}.", "",
    ]
    return "\n".join(lines)


def runtime_report(summary, metadata):
    args = metadata["arguments"]
    stage=args.get("optimization_stage")
    append_stage=stage=="runtime-append"
    modern=stage in ("runtime-copies","adapters","helper-effects","verify-each","codegen","training-tier","profile-edges", "profile-loop-edges", "final-unroll", "outer-unroll")
    lines = ["# Y CPU JIT " + ("outer natural-loop unrolling" if stage=="outer-unroll" else "final IR loop-unrolling" if stage=="final-unroll" else "atomic loop-control profile edges" if stage=="profile-loop-edges" else "atomic profile edges" if stage=="profile-edges" else "training IR tier" if stage=="training-tier" else "native code-generation" if stage=="codegen" else "compilation verification" if stage=="verify-each" else "scalar-helper effect" if stage=="helper-effects" else "checked-call adapter" if stage=="adapters" else "copy" if stage=="runtime-copies" else "append" if append_stage else "runtime") + " optimization comparison with C#", "",
        f"Measured {metadata['started_utc']} on {metadata['cpu_model']}; pinned to {metadata['affinity']}.",
        f"LLVM {metadata['jit_llvm_version']}, {metadata['csharp_runtime']}. "
        f"{args['repeats']} independent interleaved triples; {args['calls']} calls per timed batch "
        f"after {args['warmup']} standardized warmups per kernel.", "",
        "Both Y variants use the final compiler and common semantic fixes; previous settings do not replay "
        "a historical compiler binary. Previous Y enables rotate recognition and applies all measured branch profiles, with runtime "
        "queries using the existing callbacks. Next Y also exposes the headers and guarded byte reads "
        "of proven local String/Vec objects to LLVM, lowers the byte-to-ASCII conversion directly, and omits branch weights on structural natural-loop "
        "headers/latches. Both variants compile instrumented code, execute "
        f"{args['profile_warmups']} training calls per kernel, then recompile with their actual profiles. "
        "The final timed code contains no profiling probes. This comparison changes both runtime lowering "
        "and loop-profile policy; it does not isolate their individual performance effects.", "",
        "| Kernel | Previous Y ms/call | Next Y ms/call | C# ms/call | Previous / next Y | C# / next Y | Paired C# / Y range |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, item in summary["kernels"].items():
        values, paired = item["variants_ns"], item["paired_csharp_over_optimized"]
        lines.append(f"| {name} | {values['y_previous']['median'] / 1e6:.4f} | "
            f"{values['y_optimized']['median'] / 1e6:.4f} | {values['csharp']['median'] / 1e6:.4f} | "
            f"{item['baseline_over_optimized']:.2f}x | {item['csharp_over_optimized']:.2f}x | "
            f"{paired['min']:.2f}–{paired['max']:.2f}x |")
    lines += ["", "Ratios above 1 favor next Y. Paired ranges describe observed min/max values, "
        "not confidence intervals. Small differences should be read as near parity.", "",
        "| Preparation stage | Previous Y median ms (small cold inputs) | Next Y median ms (small cold inputs) |",
        "| --- | ---: | ---: |"]
    for label, key in (("Instrumented source compilation", "instrumented_compile_ns"),
                       ("Measured profile collection", "profile_collection_ns"),
                       ("Profile snapshot", "profile_snapshot_ns"),
                       ("Optimized recompilation", "optimized_recompile_ns"),
                       ("Total preparation", "prepare_ns")):
        lines.append(f"| {label} | {summary['cold']['y_previous'][key]['median']:.3f} | "
                     f"{summary['cold']['y_optimized'][key]['median']:.3f} |")
    lines += ["", f"C# prebuilt IL preparation: {summary['cold']['csharp']['compile_ns']['median']:.3f} ms. "
        "Its source-to-IL build is excluded. Y starts with source text; these preparation costs have "
        "different starting points. Small cold inputs use n=64, Fibonacci n=10 and two bulk appends of 32 characters.", "",
        "Profile training and recompilation are explicit work excluded from steady-state kernel timers. "
        "Full-size training costs are retained in every warm-run record. Profiles use the same input "
        "distribution as the timed calls; branch frequencies do not prove a branch unreachable.", "",
        "The original eight definitions remain unchanged. The two new workloads allocate an empty "
        "local object, append runtime-selected ASCII A/z values, scan eight times, and return a weighted "
        "checksum. Every 128 characters they deliberately read index -1 and index length; both languages "
        "return zero for these guarded reads. Y uses mutable byte String/Vec objects; C# uses StringBuilder "
        "and List<byte>. ASCII makes the character values equal despite different byte versus UTF-16 "
        "representations. Container layouts, growth rules and allocators differ. This measures equivalent "
        "algorithms and observable results, not identical runtime representations.", "",
        "Allocation, growth, append callbacks, scans, index guards and explicit Y frees occur inside "
        "each new kernel's timer. C# uses ordinary managed reclamation: collections occurring within a "
        "batch are included, and allocated bytes and collection counts are saved per new kernel. "
        "Deferred reclamation after the batch is not charged. Native input arrays, output buffers, "
        "hashing and JSON formatting remain outside timers for the original eight workloads.", "",
        "All previous, next and instrumented-training outputs match independent Python implementations. "
        "Full-array hashes verify native memory writes and all short-circuit counters are checked exactly. "
        "Raw branch counts, profile fingerprints, applied-site counts, both LLVM IRs and LLVM-generated "
        "native assembly are preserved beside this report.", "",
        "C# retains Release .NET 8 with tiering and ReadyToRun disabled, so this remains a fixed non-PGO "
        "comparison. Kernel methods use NoInlining; the two object workloads use one indirect host call "
        "per invocation in each language. The other eight use the same host boundaries as previous reports. "
        "Workers run sequentially with rotating orders. CPU affinity reduces migration; other host activity "
        "and CPU frequency can still affect results. These synthetic workloads do not establish a general "
        "language speed ranking.", "",
        f"Inputs: integer={args['integer_n']}, Fibonacci={args['fib_n']}/{args['fib_n'] + 1}, "
        f"recurrence={args['float_n']}, memory={args['memory_n']}, unsigned={args['unsigned_n']}, "
        f"dot={args['dot_n']}, logical={args['logical_n']}, search={args['search_n']}, "
        f"string={args['string_n']}, vector={args['vec_n']}.", ""]
    if append_stage:
        lines[5] = ("Both Y variants use the final compiler and common semantic fixes, rotate recognition, "
            "proven-local header/byte query lowering, direct byte-to-ASCII conversion and profiles that omit "
            "structural natural-loop headers/latches. Previous Y appends through callbacks. Next Y emits "
            "inline appends when a proven local object has spare capacity; growth and free remain callbacks. "
            f"Both compile instrumented code, execute {args['profile_warmups']} training calls per kernel, "
            "then recompile with their actual profiles. Timed code contains no probes. This controlled "
            "comparison changes the append optimization setting on the same compiler/source and input distribution.")
    if modern:
        lines[5]=( "Both Y variants use rotate recognition, proven-local queries and byte-to-ASCII conversion, "
            "spare-capacity append lowering and measured profiles excluding structural loop controls. "
            + ("Both use compact checked-call adapters; previous Y disables dynamic-width/bulk copy lowering and next Y enables it. "
               if stage=="runtime-copies" else "Both enable all runtime copy lowering; previous Y permits source bodies to inline into checked-call adapters, while next Y retains calls to the shared native entrypoints. ")
            +f"Both compile instrumented code, execute {args['profile_warmups']} training calls per kernel, then recompile with their actual profiles. Timed code contains no probes. Only the selected copy or adapter setting changes on the same compiler/source/input distribution.")
    if args.get("suite") in ("copies","helpers"):
        lines += ["Three copy workloads extend the preserved ten definitions. The dynamic byte and I64 vectors "
            "receive element_size as a runtime argument (1 or 8 respectively), append from an initialized "
            "scalar local, scan eight times and include final length in their checksums. C# uses List<byte> "
            "and List<long> for the same values; this comparison samples matching scalar widths. The I64 "
            "values populate high bits and the independent Python oracle models modulo-2^64 signed arithmetic. "
            "The bulk string creates a 32-character ASCII chunk from its seed, appends it repeatedly, scans "
            "eight times with the same invalid-index guards, includes final length and frees both objects. "
            "C# uses StringBuilder.Append(StringBuilder). Allocation, growth and copies occur inside "
            "these kernel timers; representations and reclamation still differ.", "",
            f"Copy inputs: byte vector={args['dynamic_byte_n']}, I64 vector={args['dynamic_i64_n']}, "
            f"bulk string={args['bulk_n']} appends of 32 characters.", ""]
    if stage in ("helper-effects","verify-each","codegen","training-tier","profile-edges", "profile-loop-edges", "final-unroll", "outer-unroll"):
        lines[5]=( "Both Y variants use rotate recognition, local query/ASCII/append/copy lowering, compact "
            "checked-call adapters and measured profiles excluding structural loop controls. Previous Y "
            "conservatively discards local runtime proofs around ordinary source calls; next Y preserves "
            "those proofs only for statically checked scalar helpers without object/pointer effects. "
            f"Both compile instrumented code, execute {args['profile_warmups']} training calls per kernel "
            "and recompile with actual profiles. Timed code has no probes; only the helper-effect setting changes.")
        lines += ["The helpers suite preserves the prior thirteen kernel definitions and adds direct/helper "
            "pairs for the complete String/Vec lifecycle. string_scan_helper and vec_scan_append_helper "
            "match the existing direct kernels exactly, but move ASCII selection and checksum weight into "
            "ordinary source helpers taking and returning only I64/bool scalars. Both helpers have no "
            "pointer/object arguments or returns. C# uses the same algorithms and ordinary scalar helpers "
            "under its default inlining policy; kernel entries retain NoInlining. The independent "
            "bytearray oracle checks every direct and helper output. These are algorithm/runtime "
            "comparisons, not a general language ranking or a pure code-generation comparison.", "",
            "Y uses explicitly measured PGO and pays profile collection/recompilation outside steady-state "
            "timers. C# tiering and dynamic PGO remain disabled. Y flat byte Strings/Vecs, C# UTF-16 "
            "StringBuilder and byte-based List<byte> have different representations and growth policies. Object allocation, "
            "growth, guarded reads and Y frees occur in each kernel timer; C# collections within batches "
            "are included, while deferred reclamation is uncharged. Headline ratios divide median times; "
            "paired ranges are descriptive observations, not confidence intervals.", ""]
    if stage=="verify-each":
        lines[5] = ("Both Y variants retain LLVM O3, rotate recognition, scalar-helper analysis, local "
            "query/ASCII/append/copy lowering, compact adapters and measured profiles excluding structural "
            "loop controls. Previous Y enables intermediate LLVM VerifyEach checks; next Y disables them. "
            "Both explicitly verify input and the module after each attempted pipeline: two boundary "
            "checks for ordinary/instrumented compilation, three for profiled compilation with profile "
            "selection. Only verify_each_pass changes. "
            f"Both execute {args['profile_warmups']} training calls per kernel and recompile with actual "
            "profiles. Intermediate verifier diagnosis is traded for checked boundaries; the production "
            "default remains VerifyEach enabled. No initial-tier or lazy-materialization policy changes.")
        lines += ["Both arms must retain identical profiles, first-call results, training streams and "
            "timed outputs. The first warm pair also saves instrumented and profiled IR and requires exact "
            "cross-arm equality. The compilation tables separate default-pipeline and profile-selection "
            "wall costs from explicit boundary checks and eager materialization. An observed on/off "
            "difference is not an exclusive measurement of verifier instructions, and identical saved "
            "IR does not prove identical native code placement. Small smoke runs validate the harness "
            "only; performance claims require repeated full-size measurements and independent audit.", ""]
    if stage == "codegen":
        lines[5] = ("Both Y variants retain default<O3> IR optimization and a separate level-3 optimization "
            "target, host CPU/features, PIC/JITDefault code generation, rotation recognition, scalar-helper "
            "and runtime lowering, compact adapters, intermediate VerifyEach and explicit input/pass-boundary "
            "checks. Structural loop-control weights remain excluded. Previous Y inherits native code-generation "
            "level 3 (Aggressive); next Y overrides only the ORC code-generation target to level 2 (Default). "
            f"Both execute {args['profile_warmups']} identical training calls per kernel and eagerly materialize "
            "the full unit. Instrumented and final IR, profiles and all outputs must match across arms. "
            "The production code-generation default remains inherited; no initial-tier or lazy policy changes.")
        lines += ["This compares only the ORC code-generation level within Y. The fixed .NET 8.0.31 "
            "non-PGO baseline retains its existing algorithm, container and reclamation differences. "
            "The passive object-ready observer leaves compiled bytes unchanged. Before/after-object "
            "first-lookup intervals include ORC/callback work and do not isolate backend/linker instructions. "
            "Object-file sizes do not measure executable-memory allocation. Saved assembly is reconstructed "
            "with llc -O3 for previous Y and -O2 for next Y; each actual command is recorded in metadata. "
            "Those reconstructions use code-model=small rather than the live ORC JITDefault selection and "
            "do not expose actual JIT addresses or code placement. All signed cost differences and kernel "
            "regressions remain in raw/summary evidence. Smoke runs validate the harness only; performance "
            "claims require repeated full-size measurements and independent audit.", ""]
    if stage == "training-tier":
        lines[5] = ("Both Y variants retain final default<O3> IR optimization/native level3, host ISA, "
            "PIC/JITDefault, runtime/helper/adapter/rotation optimizations, intermediate VerifyEach and "
            "explicit boundary checks. Structural loop-control weights remain excluded. Previous Y uses "
            "inherited training IR level3; next Y requests level1 only for the instrumented IR pipeline "
            "and its optimization target. Training native optimization remains level3 in both arms. "
            f"Both execute {args['profile_warmups']} identical training calls per kernel, then compile their "
            "own actual profiles at final IR/native level3. Raw branch counts, all training/first/timed "
            "outputs and final IR must match. Instrumented IR is saved and may differ. The production "
            "training default remains inherited; eager materialization and workload inputs are unchanged.")
        lines += ["Per-compilation metadata records the actual IR pipeline/target and native levels, "
            "requested overrides and verification policy. The total preparation comparison charges "
            "instrumented compilation, training execution, snapshot and final recompilation. Less IR "
            "optimization can shift cost into native emission or slower training code; signed losses "
            "are retained in every interval. Both final and instrumented llc reconstructions use -O3 "
            "for native optimization; only instrumented IR inputs may differ. These are reconstructions "
            "with code-model=small rather than dumped live JITDefault ORC code. Same final IR does not "
            "establish the same code placement/cache/allocator state or a throughput benefit. The fixed "
            ".NET8.0.31 non-PGO baseline and its representation/GC/compilation-scope differences remain. "
            "Diagnostic or smoke timing is not a performance claim; conclusions require the frozen "
            "repeated run and independent audit.", ""]
    if stage == "profile-edges":
        lines[5] = ("Both Y variants explicitly request training IR level1 with native level3 and retain "
            "final default<O3> IR/native level3, host ISA, PIC/JITDefault, runtime/helper/adapter/rotation "
            "optimizations, intermediate VerifyEach and explicit boundary checks. Structural loop-control "
            "weights remain excluded. Previous Y selects each counter address before its original "
            "conditional branch; next Y places the increment on the corresponding edge with a fixed "
            "counter address. Each branch observation still performs one aligned monotonic atomic "
            "increment; there is no sampling or batching. "
            f"Both execute {args['profile_warmups']} identical training calls per kernel, then compile their "
            "own actual profiles at final IR/native level3. Exact profiles, all retained outputs and saved "
            "final IR must match. Instrumented IR is saved and may differ. The requested "
            "profile_edge_counters policy is recorded for every compilation, including final compilation "
            "where it has no effect. Its production default remains false; eager materialization is unchanged.")
        lines += ["Counter array layout, branch/site identity and snapshot semantics are unchanged. "
            "Per-compilation metadata records actual IR/native levels, training/native overrides, "
            "verification and the requested counter policy. The full preparation comparison charges "
            "instrumented compilation, measured training execution, snapshot and final recompilation; "
            "all signed losses remain in the raw and paired evidence. Both llc reconstructions use -O3 "
            "for native optimization with code-model=small; these are not dumped live JITDefault objects. "
            "Matching final IR does not establish matching native placement, allocator/cache state or "
            "a final throughput benefit. The fixed .NET8.0.31 non-PGO baseline retains the existing "
            "representation/GC/compilation-scope differences. Diagnostic and smoke timings validate the "
            "harness only; performance claims require full repeated measurements and independent audit.", ""]
    if stage == "profile-loop-edges":
        lines[5] = ("Both Y variants explicitly request training IR level1 with native level3 and retain "
            "final default<O3> IR/native level3, host ISA, PIC/JITDefault, runtime/helper/adapter/rotation "
            "optimizations, intermediate VerifyEach and explicit boundary checks. Final profile weights "
            "still exclude structural loop controls. Previous Y selects each counter address before its "
            "original conditional branch; next Y places fixed-address increments on the selected edge "
            "only for predictable original natural-loop iteration controls without PHI successors. "
            "Other branches keep selected-address increments. Each observation still performs one "
            "aligned monotonic atomic increment; there is no sampling or batching. "
            f"Both execute {args['profile_warmups']} identical training calls per kernel, then compile their "
            "own actual profiles at final IR/native level3. Exact profiles, all retained outputs and saved "
            "final IR must match. Instrumented IR is saved and may differ. Only profile_loop_edge_counters "
            "is false/true; profile_edge_counters remains false in both arms. Both requested flags are "
            "recorded for every compilation, including final compilation where they have no effect. "
            "The new production default remains false; eager materialization is unchanged.")
        lines += ["Natural-loop classification uses the original lowered CFG and branch/site identities "
            "before instrumentation. PHI-successor branches retain the selected-address form. Counter "
            "array layout, observation counts and owned atomic snapshot semantics remain unchanged. "
            "The full preparation comparison charges instrumented compilation, measured training "
            "execution, snapshot and final recompilation; all signed losses remain in raw and paired "
            "evidence. Fixed loop-edge placement can change compilation and training costs separately, "
            "so a compile saving does not establish faster profiling execution. Both llc reconstructions "
            "use -O3 with code-model=small and are not dumped live JITDefault objects. Matching final IR "
            "does not establish matching native placement, allocator/cache state or a final throughput "
            "benefit. The fixed .NET8.0.31 non-PGO baseline retains representation/GC/compilation-scope "
            "differences. Smoke and diagnostic timings validate the harness only; performance claims "
            "require full repeated measurements and independent audit.", ""]
    if stage == "final-unroll":
        lines[5] = ("Both Y variants explicitly request temporary training IR1/native3 with inherited "
            "LLVM default loop-unrolling tuning and retain final O3 IR/native3, host ISA, PIC/JITDefault, "
            "runtime/helper/adapter/rotation optimizations, VerifyEach and mandatory boundary checks. "
            "Both atomic edge flags remain false; final structural loop-control profile weights remain "
            "excluded. Previous Y inherits LLVM default final-loop-unrolling tuning; next Y disables "
            "that tuning in ordinary/profile-use IR pipelines. Only final_loop_unrolling changes. "
            "The actual ir_loop_unrolling metadata remains true during Instrument compilation and "
            "follows the requested flag during final compilation; true denotes inherited LLVM defaults "
            "rather than forcing an unroll. "
            f"Both execute {args['profile_warmups']} identical training calls per kernel and then compile "
            "their own actual profiles at final O3 IR/native3. Instrumented IR and exact retained profiles, "
            "training/first/timed outputs must match. Saved final IR and reconstructed final assembly are "
            "expected to differ. The production default remains true; eager materialization is unchanged.")
        lines += ["A final IR compilation saving can be offset by slower native execution. Paired total-use "
            "models combine measured warm preparation with the sum of all fifteen per-kernel call costs "
            "for each repeat, preserving both gains and losses. One bundle executes each heterogeneous "
            "synthetic workload once; the bundle is not a representative application or a normalized "
            "cross-language workload. Extrapolated bundle counts are models, not newly timed executions. "
            "Profiles, atomic increments, observation counts and owned snapshot semantics are unchanged. "
            "Native level3 and llc -O3 stay fixed. Saved assembly uses code-model=small and is a "
            "reconstruction, not dumped live JITDefault code. The fixed .NET8.0.31 baseline retains the "
            "existing representation/GC/compilation-scope differences. Every native regression and signed "
            "cost difference remains in the evidence; phase savings do not establish an overall win.", ""]
    if stage == "outer-unroll":
        lines[5] = ("Both Y variants explicitly request temporary training IR1/native3 with inherited "
            "LLVM defaults and final O3 IR/native3, host ISA, PIC/JITDefault, runtime/helper/adapter/rotation "
            "optimizations, VerifyEach and mandatory boundary checks. Both atomic edge flags remain false; "
            "final structural loop-control profile weights remain excluded. Global final_loop_unrolling "
            "stays true. Previous Y inherits LLVM defaults for all final loops; next Y adds "
            "llvm.loop.unroll.disable only to eligible original natural loops strictly containing another "
            "natural loop. Innermost and conservatively skipped loops retain their default metadata policy. "
            "Only final_unroll_outer_loops changes, true/false. The requested flag, effective "
            "outer_unroll_policy_active and original-loop outer_unroll_annotations count are recorded for "
            "every compilation and the final worker record. Instrument compilation ignores the request "
            "and always has inactive policy/zero annotations. Annotation counts are not avoided dynamic "
            "transformations or actual pass timings. "
            f"Both execute {args['profile_warmups']} identical training calls per kernel and compile their "
            "own actual profiles at final O3 IR/native3. Instrumented IR/assembly, exact profiles and all "
            "training/first/timed outputs must match; at least one final IR and reconstructed assembly "
            "difference is required across the suite. The production default remains true; eager "
            "materialization is unchanged.")
        lines += ["Natural-loop nesting is classified from the original lowered CFG before LLVM transforms. "
            "Eligible outer-loop latch metadata suppresses their unrolling without forcing an inner-loop "
            "factor or disabling vectorization; the final native result still depends on the full pipeline. "
            "Paired total-use models combine measured warm preparation with the sum of all fifteen "
            "per-kernel call costs at each repeat, retaining every native regression and signed loss. "
            "One bundle executes each heterogeneous synthetic workload once, not a representative "
            "application; extrapolated counts are models, not newly timed executions. Profiles, aligned "
            "monotonic atomic observations and owned snapshot semantics are unchanged. llc -O3 assembly "
            "uses code-model=small and is reconstructed, not dumped live JITDefault code. The fixed "
            ".NET8.0.31 baseline retains representation/GC/compilation-scope differences. A phase saving "
            "does not establish an overall or native-execution win.", ""]
    if "checked_calls" in summary:
        lines += ["| Checked API workload | Previous native ns/call | Previous checked ns/call | Next native ns/call | Next checked ns/call |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for name,item in summary["checked_calls"].items():
            a,b=item["y_previous"],item["y_optimized"]
            lines.append(f"| {name} | {a['native_ns_per_call']['median']:.1f} | {a['checked_ns_per_call']['median']:.1f} | {b['native_ns_per_call']['median']:.1f} | {b['checked_ns_per_call']['median']:.1f} |")
        lines += ["",f"Tiny checked calls use integer_branch n=1 with {args['checked_tiny_calls']} calls and 64 warmups; "
            "large calls use the standard integer input/call/warmup counts. Native and checked outputs are "
            "compared exactly in the worker; independent Python references check both full-stream hashes "
            "and saved first/last output excerpts. Timers include native dispatch or checked .call validation, "
            "frame packing, allocation and return handling. Output buffers and hashing are outside timers. "
            "Native batches run first and checked batches second in each process, so cache/frequency effects can contribute to the API comparison. These supplemental Y-only measurements are separate from C# native entrypoint comparisons.", ""]
    return "\n".join(lines)


def report(summary, metadata):
    args = metadata["arguments"]
    lines = [
        "# Y CPU JIT versus C#" + (" — expanded CPU workloads" if args.get("suite") == "expanded" else ""), "",
        f"Measured {metadata['started_utc']} on {metadata['cpu_model']} ({metadata['platform']}).",
        f"One CPU pinned: {metadata['affinity']}. {args['repeats']} independent process pairs, "
        f"{args['calls']} calls per timed batch, {args['warmup']} warmup calls per kernel.", "",
        f"Y library: {metadata.get('jit_llvm_version', 'see metadata.json')}. "
        f"C# runtime: {metadata.get('csharp_runtime', 'see metadata.json')}.", "",
        "| Kernel | Y median ms/call | C# median ms/call | C# / Y | Paired ratio range |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, item in summary["kernels"].items():
        paired = item["paired_csharp_over_y"]
        lines.append(f"| {name} | {item['y_ns']['median'] / 1e6:.4f} | "
                     f"{item['csharp_ns']['median'] / 1e6:.4f} | "
                     f"{item['csharp_over_y']:.2f}x | {paired['min']:.2f}–{paired['max']:.2f}x |")
    lines += ["", "A ratio above 1 means Y was faster for that kernel. The range is the minimum "
              "and maximum ratio across paired runs; it is descriptive, not a confidence interval.", "",
              "| Cold measurement | Y median ms | C# median ms |",
              "| --- | ---: | ---: |"]
    for label, key in (("JIT preparation", "compile_ms"), ("First calls (small inputs)", "first_call_ms"),
                       ("Process launch to JSON result", "launch_to_json_ms")):
        lines.append(f"| {label} | {summary['cold']['y'][key]['median']:.3f} | "
                     f"{summary['cold']['csharp'][key]['median']:.3f} |")
    lines += [
        "", "Y preparation includes source parsing, type checking, LLVM optimization and materialization "
        "of the selected functions and their helpers. C# preparation uses RuntimeHelpers.PrepareMethod on the corresponding already-built "
        "IL methods; its source-to-IL build is excluded. These preparation times have different starting "
        "points. Launch-to-result includes runtime startup and JSON formatting, and uses small kernel inputs.", "",
        "The Y compiler uses LLVM O3 and the host CPU target. C# uses Release compilation on .NET 8, "
        "with tiering disabled so the measured methods receive optimized JIT code immediately. "
        "ReadyToRun is disabled. Kernel inputs are passed at runtime; all outputs are saved and checked "
        "against independent Python implementations. The memory array is reset before each batch, "
        "and the entire final array is checked with a 64-bit hash. The expanded short-circuit workload "
        "checks all three side-effect counters exactly. Unsigned outputs use explicit modulo-2^64 "
        "Python arithmetic and include values above I64::MAX. Float outputs use relative and absolute "
        "tolerances of 1e-12.", "",
        "The memory comparison uses unchecked pointer indexing in both languages. Allocations, array "
        "initialization, memory hashing and JSON formatting occur outside the original eight timed kernel batches. "
        "The timed batches include the host loop, output stores and function-call boundaries. "
        "C# kernel methods use NoInlining to preserve those boundaries. Recursive algorithms execute "
        "their own recursion and may be optimized differently by each JIT.", "",
        f"Inputs: integer n={args['integer_n']}; Fibonacci n={args['fib_n']} or n+1; "
        f"float n={args['float_n']}; indexed memory n={args['memory_n']} over 65536 I64 elements.", "",
        "Each pair alternates which engine runs first. Kernels run in the same order in each process. "
        "No benchmark processes run concurrently. CPU affinity reduces migration, but other host activity "
        "and frequency changes can still affect results. These synthetic workloads establish "
        "performance on this machine; they do not establish a general language speed ranking.", "",
        "Raw process output, individual durations, full result arrays, correctness references, source SHA-256 "
        "hashes, binary hashes and tool versions are preserved beside this report.", "",
    ]
    if args.get("y_mode") in ("previous", "optimized"):
        lines += ["This Y run uses explicit measured branch profiling before final compilation: "
                  f"{args.get('profile_warmups', 12)} training calls per kernel, followed by recompilation. "
                  "Training outputs and side effects are also independently checked. The JIT preparation "
                  "row above is the final recompilation only; instrumented compilation and training add "
                  "work before it. Their individual costs and total preparation are retained in every raw "
                  "Y record. Use --compare-optimizations for the controlled three-variant report and "
                  "its separate preparation table.", ""]
    if args.get("suite") in ("expanded", "runtime", "copies", "helpers"):
        lines += [f"Expanded inputs: unsigned mix n={args['unsigned_n']}; F64 dot reduction "
                  f"n={args['dot_n']} over two 65536-element arrays; short-circuit n={args['logical_n']}; "
                  f"binary search n={args['search_n']} queries over 65536 sorted I64 elements.", "",
                  "The dot inputs use exact binary fractions, and reduction order is the same in Y and C#. "
                  "The unsigned kernel deliberately wraps arithmetic and uses only legal constant shifts. "
                  "Short-circuit RHS helpers increment counters, so eager evaluation is detected. "
                  "Binary search includes present and missing values; its independent oracle uses Python's bisect.", ""]
    if args.get("suite") in ("runtime", "copies", "helpers"):
        lines += ["The two runtime-object workloads include allocation, growth, scans, guarded invalid "
                  "reads and Y frees inside their timers. C# uses StringBuilder/List<byte> with managed "
                  "reclamation; deferred collection outside a batch is not charged. ASCII aligns Y byte "
                  "values and C# UTF-16 character values for these inputs only. Both return independently "
                  "checked weighted checksums; their container layouts and allocators differ.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dotnet", default=shutil.which("dotnet"), help="dotnet executable from a .NET 8 SDK")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--cold-repeats", type=int, default=5)
    parser.add_argument("--calls", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--integer-n", type=int, default=250000)
    parser.add_argument("--fib-n", type=int, default=25)
    parser.add_argument("--float-n", type=int, default=1000000)
    parser.add_argument("--memory-n", type=int, default=262144)
    parser.add_argument("--unsigned-n", type=int, default=250000)
    parser.add_argument("--dot-n", type=int, default=262157)
    parser.add_argument("--logical-n", type=int, default=250000)
    parser.add_argument("--search-n", type=int, default=20000)
    parser.add_argument("--string-n", type=int, default=16384)
    parser.add_argument("--vec-n", type=int, default=16384)
    parser.add_argument("--dynamic-byte-n", type=int, default=16384)
    parser.add_argument("--dynamic-i64-n", type=int, default=16384)
    parser.add_argument("--bulk-n", type=int, default=512)
    parser.add_argument("--checked-tiny-calls", type=int, default=20000)
    parser.add_argument("--suite", choices=("original", "expanded", "runtime", "copies", "helpers"), default="expanded")
    parser.add_argument("--optimization-stage", choices=("rotate-profile", "runtime", "runtime-append", "runtime-copies", "adapters", "helper-effects", "verify-each", "codegen", "training-tier", "profile-edges", "profile-loop-edges", "final-unroll", "outer-unroll"), default="rotate-profile")
    parser.add_argument("--compare-optimizations", action="store_true", help="interleave baseline Y, optimized Y and C#")
    parser.add_argument("--y-mode", choices=("baseline", "previous", "optimized"), default="optimized")
    parser.add_argument("--profile-warmups", type=int, default=12)
    parser.add_argument("--cpu", type=int, help="pin workers to this allowed logical CPU")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    if args.optimization_stage=="runtime-copies" and args.suite not in ("copies","helpers"): parser.error("runtime-copies requires --suite copies or helpers")
    if args.optimization_stage in ("helper-effects","verify-each","codegen","training-tier","profile-edges", "profile-loop-edges", "final-unroll", "outer-unroll") and args.suite!="helpers": parser.error("this stage requires --suite helpers")
    if args.optimization_stage in ("codegen", "training-tier", "profile-edges", "profile-loop-edges", "final-unroll", "outer-unroll") and not args.compare_optimizations:
        parser.error(f"{args.optimization_stage} requires --compare-optimizations")
    if args.optimization_stage in ("runtime-append","adapters") and args.suite not in ("runtime","copies","helpers"):
        parser.error("this runtime stage requires --suite runtime or copies")
    if args.optimization_stage in ("runtime-append","runtime-copies","adapters","helper-effects","verify-each","codegen","training-tier","profile-edges", "profile-loop-edges", "final-unroll", "outer-unroll") and args.y_mode=="baseline":
        parser.error("this stage requires --y-mode previous or optimized")
    if not args.dotnet:
        parser.error("a .NET 8 SDK is required; specify --dotnet /path/to/dotnet")
    if (min(args.repeats, args.cold_repeats, args.calls, args.integer_n, args.float_n, args.memory_n,
            args.unsigned_n, args.dot_n, args.logical_n, args.search_n, args.string_n, args.vec_n, args.dynamic_byte_n, args.dynamic_i64_n, args.bulk_n, args.checked_tiny_calls, args.profile_warmups) < 1
            or args.warmup < 0 or not 2 <= args.fib_n <= 40):
        parser.error("invalid benchmark dimensions")
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = ROOT / "build_artifacts" / f"cpu_jit_{stamp}"
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    dotnet = str(Path(shutil.which(args.dotnet) or args.dotnet).resolve())
    env = os.environ.copy()
    env.update({
        "DOTNET_TieredCompilation": "0", "COMPlus_TieredCompilation": "0",
        "DOTNET_ReadyToRun": "0", "COMPlus_ReadyToRun": "0",
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1",
        "DOTNET_CLI_HOME": str(ROOT / "build_artifacts" / "cpu_jit_dotnet_home"),
        "NUGET_PACKAGES": str(ROOT / "build_artifacts" / "cpu_jit_nuget"),
        "DOTNET_NOLOGO": "1",
    })
    artifact_dir = ROOT / "build_artifacts" / "cpu_jit_csharp"
    y_binary = ROOT / "target" / "release" / "examples" / "cpu_jit_bench"
    cs_binary = artifact_dir / "CSharpBench.dll"
    if not args.skip_build:
        commands = [
            ["cargo", "build", "--offline", "--release", "--example", "cpu_jit_bench"],
            [dotnet, "build", str(ROOT / "benchmarks/cpu_jit/CSharpBench.csproj"), "-c", "Release",
             "-o", str(artifact_dir), "--configfile", str(ROOT / "benchmarks/cpu_jit/NuGet.Config"),
             f"-p:BaseIntermediateOutputPath={artifact_dir / 'obj'}/", "--nologo"],
        ]
        for index, command in enumerate(commands):
            print(f"Building {'Y worker' if index == 0 else 'C# worker'}...", flush=True)
            build = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
            (args.output / f"build-{index}.log").write_text(build.stdout + build.stderr, encoding="utf-8")
            if build.returncode:
                raise RuntimeError(f"build failed; see {args.output / f'build-{index}.log'}")
    if not y_binary.is_file() or not cs_binary.is_file():
        raise RuntimeError("worker binaries missing; run without --skip-build")
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    if allowed:
        cpu = args.cpu if args.cpu is not None else allowed[0]
        if cpu not in allowed:
            raise ValueError(f"CPU {cpu} is outside allowed affinity {allowed}")
        os.sched_setaffinity(0, {cpu})
    cpu_model = platform.processor()
    if Path("/proc/cpuinfo").is_file():
        cpu_model = next((line.split(":", 1)[1].strip() for line in
                          Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), cpu_model)
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "cpu_model": cpu_model, "platform": platform.platform(), "allowed_affinity": allowed,
        "affinity": sorted(os.sched_getaffinity(0)) if allowed else [],
        "rustc": metadata_command(["rustc", "-Vv"]),
        "dotnet": metadata_command([dotnet, "--info"]),
        "lscpu": metadata_command(["lscpu"]),
        "git_head": metadata_command(["git", "rev-parse", "HEAD"]),
        "git_status": metadata_command(["git", "status", "--short"]),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in SOURCE_PATHS},
        "binary_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (y_binary, cs_binary)},
        "jit_environment": {name: env[name] for name in env if name.startswith(("DOTNET_", "COMPlus_", "Y_LLVM"))},
        "inherited_build_environment": {name: env[name] for name in ("RUSTFLAGS", "CARGO_BUILD_TARGET") if name in env},
    }
    write_json(args.output / "metadata.json", metadata)
    print("Computing independent Python references...", flush=True)
    expected = references(args.calls, args.integer_n, args.fib_n, args.float_n, args.memory_n,
                          args.unsigned_n, args.dot_n, args.logical_n, args.search_n, args.suite, args.string_n, args.vec_n, args.dynamic_byte_n, args.dynamic_i64_n, args.bulk_n)
    cold_expected = references(1, 64, 10, 64, 64, 64, 64, 64, 64, args.suite, 64, 64, 64, 64, 2)
    write_json(args.output / "references.json", expected)
    write_json(args.output / "cold-references.json", cold_expected)
    train_expected = references(args.profile_warmups, args.integer_n, args.fib_n, args.float_n,
        args.memory_n, args.unsigned_n, args.dot_n, args.logical_n, args.search_n, args.suite, args.string_n, args.vec_n, args.dynamic_byte_n, args.dynamic_i64_n, args.bulk_n)
    cold_train_expected = references(args.profile_warmups, 64, 10, 64, 64, 64, 64, 64, 64, args.suite, 64, 64, 64, 64, 2)
    write_json(args.output / "training-references.json", train_expected)
    write_json(args.output / "cold-training-references.json", cold_train_expected)
    checked_expected=checked_references(expected,args.checked_tiny_calls) if args.optimization_stage=="adapters" else None
    if checked_expected is not None: write_json(args.output / "checked-references.json",checked_expected)
    workers = {"y": [str(y_binary)], "csharp": [dotnet, str(cs_binary)],
               "y_previous": [str(y_binary)], "y_baseline": [str(y_binary)], "y_optimized": [str(y_binary)]}
    dimensions = list(map(str, (args.calls, args.warmup, args.integer_n, args.fib_n, args.float_n,
                               args.memory_n, args.unsigned_n, args.dot_n, args.logical_n, args.search_n, args.string_n, args.vec_n, args.dynamic_byte_n, args.dynamic_i64_n, args.bulk_n, 0, args.suite)))
    records, cold = [], []
    def variants(repeat):
        if not args.compare_optimizations:
            return ("y", "csharp") if repeat % 2 == 0 else ("csharp", "y")
        orders = (("y_baseline", "y_optimized", "csharp"), ("y_optimized", "csharp", "y_baseline"),
                  ("csharp", "y_baseline", "y_optimized"), ("y_baseline", "csharp", "y_optimized"),
                  ("csharp", "y_optimized", "y_baseline"), ("y_optimized", "y_baseline", "csharp"))
        order = orders[repeat % len(orders)]
        return tuple("y_previous" if item == "y_baseline" and args.optimization_stage in ("runtime", "runtime-append", "runtime-copies", "adapters", "helper-effects", "verify-each", "codegen", "training-tier", "profile-edges", "profile-loop-edges", "final-unroll", "outer-unroll") else item for item in order)
    def worker_environment(variant):
        worker_env = env.copy()
        worker_env["Y_CPU_JIT_BENCH_MODE"] = "baseline" if variant == "y_baseline" else "previous" if variant == "y_previous" else args.y_mode if variant == "y" else "optimized"
        worker_env["Y_CPU_JIT_BENCH_STAGE"] = args.optimization_stage
        worker_env["Y_CPU_JIT_BENCH_PROFILE_WARMUP"] = str(args.profile_warmups)
        worker_env["Y_CPU_JIT_BENCH_CHECKED_TINY_CALLS"] = str(args.checked_tiny_calls)
        return worker_env
    for repeat in range(args.repeats):
        order = variants(repeat)
        triple = []
        for engine in order:
            print(f"Sample {repeat + 1}/{args.repeats}: {engine}", flush=True)
            worker_env = worker_environment(engine)
            if repeat == 0 and engine != "csharp":
                ir_name = "previous-kernels.ll" if engine == "y_previous" else "baseline-kernels.ll" if engine == "y_baseline" else "optimized-kernels.ll"
                worker_env["Y_CPU_JIT_BENCH_IR"] = str(args.output / ir_name)
                if args.optimization_stage in CONTROLLED_STAGES:
                    worker_env["Y_CPU_JIT_BENCH_INSTRUMENTED_IR"] = str(
                        args.output / ("previous-instrumented.ll" if engine == "y_previous" else "optimized-instrumented.ll"))
            record = execute(workers[engine] + dimensions, worker_env, args.output / f"sample-{repeat:02}-{engine}.json")
            validate(record, expected)
            if engine != "csharp":
                validate_configuration(record,args.optimization_stage,worker_env["Y_CPU_JIT_BENCH_MODE"])
                if checked_expected is not None: validate_checked(record,checked_expected)
                metadata["jit_llvm_version"] = record["llvm"]
                if record.get("mode") in ("previous", "optimized"):
                    validate_training(record, train_expected)
            else:
                metadata["csharp_runtime"] = record["runtime"]
                if args.optimization_stage in CONTROLLED_STAGES and record["runtime"] != ".NET 8.0.31":
                    raise ValueError("controlled compilation stages require the documented .NET 8.0.31 baseline")
            record["repeat"] = repeat
            record["variant"] = engine
            records.append(record)
            triple.append(record)
        if args.optimization_stage in CONTROLLED_STAGES and args.compare_optimizations:
            validate_controlled_pair(triple)
            if repeat == 0:
                for kind in controlled_ir_kinds(args.optimization_stage):
                    left = (args.output / f"previous-{kind}.ll").read_bytes()
                    right = (args.output / f"optimized-{kind}.ll").read_bytes()
                    if left != right:
                        raise ValueError(f"controlled comparison changed saved {kind} IR")
                if args.optimization_stage in ("final-unroll", "outer-unroll"):
                    validate_final_unroll_artifacts(args.output, "ll", args.optimization_stage)
        write_json(args.output / "samples.json", records)
        write_json(args.output / "metadata.json", metadata)
    for repeat in range(args.cold_repeats):
        order = variants(repeat)
        triple = []
        for engine in order:
            print(f"Cold sample {repeat + 1}/{args.cold_repeats}: {engine}", flush=True)
            record = execute(workers[engine] + ["1", "0", "64", "10", "64", "64", "64", "64", "64", "64", "64", "64", "64", "64", "2", "1", args.suite],
                             worker_environment(engine), args.output / f"cold-{repeat:02}-{engine}.json")
            validate(record, cold_expected, cold_only=True)
            if engine != "csharp":
                validate_configuration(record,args.optimization_stage,worker_environment(engine)["Y_CPU_JIT_BENCH_MODE"])
            elif args.optimization_stage in CONTROLLED_STAGES and record["runtime"] != ".NET 8.0.31":
                raise ValueError("controlled cold worker differs from the .NET 8.0.31 baseline")
            if record.get("mode") in ("previous", "optimized"):
                validate_training(record, cold_train_expected)
            record["variant"] = engine
            record["repeat"] = repeat
            cold.append(record)
            triple.append(record)
        if args.optimization_stage in CONTROLLED_STAGES and args.compare_optimizations:
            validate_controlled_pair(triple)
    write_json(args.output / "cold.json", cold)
    summary = summarize_optimized(records, cold) if args.compare_optimizations else summarize(records, cold)
    summary.update(final_unroll_use_summary(records))
    llc = shutil.which("llc")
    if llc:
        metadata["llc"] = metadata_command([llc, "--version"])
        metadata["llc_commands"] = []
        for name in ("baseline-kernels", "previous-kernels", "optimized-kernels", "previous-instrumented", "optimized-instrumented"):
            ir_path = args.output / f"{name}.ll"
            if ir_path.is_file():
                level = "-O2" if args.optimization_stage == "codegen" and name.startswith("optimized-") else "-O3"
                command = [llc, level, "-mcpu=native", "-relocation-model=pic", "-code-model=small",
                           str(ir_path), "-o", str(args.output / f"{name}.s")]
                assembly = subprocess.run(command,
                                          cwd=ROOT, capture_output=True, text=True, check=False)
                metadata["llc_commands"].append({"artifact": f"{name}.s", "command": command,
                                                 "returncode": assembly.returncode})
                (args.output / f"{name}-llc.log").write_text(assembly.stdout + assembly.stderr, encoding="utf-8")
                if assembly.returncode:
                    raise RuntimeError(f"native assembly generation failed for {name}")
        if args.optimization_stage in ("final-unroll", "outer-unroll"):
            validate_final_unroll_artifacts(args.output, "s", args.optimization_stage)
        write_json(args.output / "metadata.json", metadata)
    write_json(args.output / "summary.json", summary)
    render = runtime_report if args.compare_optimizations and args.optimization_stage in ("runtime", "runtime-append", "runtime-copies", "adapters", "helper-effects", "verify-each", "codegen", "training-tier", "profile-edges", "profile-loop-edges", "final-unroll", "outer-unroll") else optimized_report if args.compare_optimizations else report
    (args.output / "report.md").write_text(render(summary, metadata)+helper_pair_report(summary)+compilation_phase_report(summary)
        +optimization_timing_report(summary)+materialization_timing_report(summary)
        +compilation_cost_pair_report(summary,args.optimization_stage)
        +final_unroll_use_report(summary), encoding="utf-8")
    print(f"Validated all outputs. Report: {args.output / 'report.md'}", flush=True)
    for name, item in summary["kernels"].items():
        if args.compare_optimizations:
            baseline = "y_previous" if args.optimization_stage in ("runtime", "runtime-append", "runtime-copies", "adapters", "helper-effects", "verify-each", "codegen", "training-tier", "profile-edges", "profile-loop-edges", "final-unroll", "outer-unroll") else "y_baseline"
            print(f"{name}: previous/baseline Y {item['variants_ns'][baseline]['median'] / 1e6:.4f} ms, "
                  f"optimized Y {item['variants_ns']['y_optimized']['median'] / 1e6:.4f} ms, "
                  f"C# {item['variants_ns']['csharp']['median'] / 1e6:.4f} ms, "
                  f"C#/optimized {item['csharp_over_optimized']:.2f}x")
        else:
            print(f"{name}: Y {item['y_ns']['median'] / 1e6:.4f} ms, "
                  f"C# {item['csharp_ns']['median'] / 1e6:.4f} ms, C#/Y {item['csharp_over_y']:.2f}x")


if __name__ == "__main__":
    main()
