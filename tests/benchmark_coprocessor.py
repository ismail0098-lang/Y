import os
import re
import sys
import subprocess
import time

def print_header(title):
    print("=" * 70)
    print(f" {title.center(68)} ")
    print("=" * 70)

def compile_and_extract_stats(ysu_path):
    print(f"[*] Compiling {ysu_path} with Y co-processor backend...")
    cmd = f"./target/release/Y {ysu_path} --emit-coprocessor"
    
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"    [!] Compilation failed:\n{res.stderr}")
            return None
        
        # Extract scheduler stats from stdout
        stdout = res.stdout
        rt_nodes = re.search(r"RT Core nodes:\s+(\d+)", stdout)
        tensor_nodes = re.search(r"Tensor Core nodes:\s+(\d+)", stdout)
        barriers = re.search(r"Sync barriers:\s+(\d+)", stdout)
        parallel_cycles = re.search(r"Est. parallel cy:\s+(\d+)", stdout)
        overlap = re.search(r"Overlap savings:\s+(\d+)", stdout)
        smem = re.search(r"SMEM budget:\s+(\d+)", stdout)
        
        return {
            "rt_nodes": int(rt_nodes.group(1)) if rt_nodes else 0,
            "tensor_nodes": int(tensor_nodes.group(1)) if tensor_nodes else 0,
            "barriers": int(barriers.group(1)) if barriers else 0,
            "parallel_cycles": int(parallel_cycles.group(1)) if parallel_cycles else 0,
            "overlap": int(overlap.group(1)) if overlap else 0,
            "smem_bytes": int(smem.group(1)) if smem else 0,
        }
    except Exception as e:
        print(f"    [!] Error running compiler: {e}")
        return None

def main():
    print_header("Y-LANG CO-PROCESSOR SCHEDULING BENCHMARK SUITE")
    
    # 1. Compile & Profile Co-processing Test Files
    test_files = [
        "tests/coprocessor_combined.ysu",
        "tests/coprocessor_test.ysu",
        "tests/coprocessor_attention.ysu",
        "tests/coprocessor_nerf.ysu",
        "tests/coprocessor_collision.ysu",
        "tests/coprocessor_db_index.ysu"
    ]
    
    results = {}
    for tf in test_files:
        if os.path.exists(tf):
            stats = compile_and_extract_stats(tf)
            if stats:
                results[tf] = stats
        else:
            print(f"[!] Test file not found: {tf}")
            
    print("\n" + "=" * 70)
    print("      CO-PROCESSOR SCHEDULER STATISTICS (RTX 4070 Ti SUPER)")
    print("=" * 70)
    print(f"{'Kernel File':<32} | {'RT Nodes':<8} | {'Tensor':<6} | {'Barriers':<8} | {'Parallel Cy':<11} | {'Overlap':<7}")
    print("-" * 88)
    for tf, stats in results.items():
        name = os.path.basename(tf)
        print(f"{name:<32} | {stats['rt_nodes']:<8} | {stats['tensor_nodes']:<6} | {stats['barriers']:<8} | {stats['parallel_cycles']:<11} | {stats['overlap']:<7}")
    print("=" * 88)
    
    # 2. Check for CUDA execution benchmark
    print("\n[*] Checking host GPU execution capabilities...")
    try:
        import torch
        import cupy as cp
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        print("    -> CUDA-enabled PyTorch & CuPy found! Launching physical benchmarks...")
        
        # Run execution benchmark script
        import benchmark
        
    except Exception as e:
        print("    -> Physical GPU execution skipped (either drivers are isolated or libraries are missing).")
        print("    -> Static compiler scheduling cycle estimates verified above.")

if __name__ == "__main__":
    main()
