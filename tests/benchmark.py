# tests/benchmark.py
import os
import time
import torch
import numpy as np

# Ensure GPU is available
assert torch.cuda.is_available(), "CUDA GPU is required for this benchmark."
device = torch.device("cuda")

# 1. PyTorch Eager Setup
def pytorch_eager_accumulate(weights, size):
    acc = torch.tensor(0.0, device=device)
    for _ in range(size):
        acc = acc + weights
    return acc

# 2. PyTorch Compiled Setup
@torch.compile
def pytorch_compiled_accumulate(weights, size):
    acc = torch.tensor(0.0, device=device)
    for _ in range(size):
        acc = acc + weights
    return acc

# Warmup PyTorch Compiled
dummy_weights = torch.tensor(1.23, device=device)
for _ in range(10):
    _ = pytorch_compiled_accumulate(dummy_weights, 1024)

# 3. Y PTX Execution Setup
print("[*] Invoking Y compiler to build train_spec.ptx...")
exit_code = os.system("./target/release/Y tests/train_spec.ysu --emit-ptx")
if exit_code != 0:
    print("[Error] Y compiler failed to compile train_spec.ysu")
    use_y_ptx = False
else:
    try:
        import cupy as cp
        # Load PTX file directly from disk
        module = cp.RawModule(path="tests/train_spec.ptx")
        train_step_gpu = module.get_function("train_step_gpu")
        
        # Setup data
        weights_gpu = cp.array([1.23], dtype=cp.float32)
        size_gpu = cp.int32(1024)
        
        # Warmup
        train_step_gpu((1,1,1), (1,1,1), (weights_gpu, size_gpu))
        use_y_ptx = True
        print("[*] Loaded Y compiled PTX successfully.")
    except Exception as e:
        print(f"[Error] Failed to load/run Y PTX: {e}")
        # Print full traceback/logs by letting it raise
        import traceback
        traceback.print_exc()
        use_y_ptx = False

# Benchmarking
iterations = 1000
weights = torch.tensor(1.23, device=device)
size = 1024

print("Starting benchmarks...")

# Eager Mode
start_eager = torch.cuda.Event(enable_timing=True)
end_eager = torch.cuda.Event(enable_timing=True)
start_eager.record()
for _ in range(iterations):
    _ = pytorch_eager_accumulate(weights, size)
end_eager.record()
torch.cuda.synchronize()
eager_time = start_eager.elapsed_time(end_eager) / iterations

# Compiled Mode
start_compiled = torch.cuda.Event(enable_timing=True)
end_compiled = torch.cuda.Event(enable_timing=True)
start_compiled.record()
for _ in range(iterations):
    _ = pytorch_compiled_accumulate(weights, size)
end_compiled.record()
torch.cuda.synchronize()
compiled_time = start_compiled.elapsed_time(end_compiled) / iterations

# Y PTX Native Mode
if use_y_ptx:
    start_ptx = cp.cuda.Event()
    end_ptx = cp.cuda.Event()
    start_ptx.record()
    for _ in range(iterations):
        train_step_gpu((1,1,1), (1,1,1), (weights_gpu, size_gpu))
    end_ptx.record()
    end_ptx.synchronize()
    ptx_time = cp.cuda.get_elapsed_time(start_ptx, end_ptx) / iterations
else:
    ptx_time = None

print("\n=== BENCHMARK RESULTS (Average execution time per kernel call) ===")
print(f"PyTorch Eager Mode:        {eager_time * 1e3:.4f} microseconds")
print(f"PyTorch Compiled (Triton): {compiled_time * 1e3:.4f} microseconds")
if ptx_time is not None:
    print(f"Y Native PTX Kernel:       {ptx_time * 1e3:.4f} microseconds")
    speedup_eager = eager_time / ptx_time
    speedup_compiled = compiled_time / ptx_time
    print(f"-> Y Speedup vs Eager:     {speedup_eager:.2f}x")
    print(f"-> Y Speedup vs Compiled:  {speedup_compiled:.2f}x")
else:
    print("Y Native PTX Kernel:       N/A")
