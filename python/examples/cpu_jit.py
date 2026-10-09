"""Run with PYTHONPATH=python python3 python/examples/cpu_jit.py."""

import ctypes
from y_lang import CPUJit


SOURCE = """
fn square(x: I64) -> I64 { return x * x; }
fn weighted(enabled: bool, weight: F64) -> F64 {
    if enabled { return weight * 3.0 + 0.5; }
    return 0.0;
}
@unsafe
fn bump(values: &mut [I32; 4]) {
    values[0] = values[0] + 10;
    values[3] = values[3] + 20;
}
"""

with CPUJit(SOURCE) as jit:
    square = jit.function("square")
    print(square.signature)
    print("square(123456):", square(123456))
    print("weighted(True, 1.25):", jit("weighted", True, 1.25))
    values = (ctypes.c_int32 * 4)(1, 2, 3, 4)
    jit("bump", values)
    print("bumped array:", list(values))

# Training happens only through these explicit calls. Recompilation produces
# a separately owned native session, so the original callables stay valid.
with CPUJit(SOURCE, instrument=True) as training:
    weighted = training.function("weighted")
    for _ in range(1000):
        weighted(False, 1.25)
    print("measured branches:", training.branch_profile()["sites"])
    with training.recompile_profiled() as optimized:
        print("profiled weighted(True, 1.25):", optimized("weighted", True, 1.25))
    print("original weighted(False, 1.25):", weighted(False, 1.25))
