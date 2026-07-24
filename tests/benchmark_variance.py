import subprocess
import time
import numpy as np
import os

def run_benchmark(cmd, cwd=None):
    cwd_arg = f", cwd='{cwd}'" if cwd else ""
    py_cmd = f"python3 -c \"import subprocess, resource, time; start = time.perf_counter(); p = subprocess.Popen('{cmd}', shell=True{cwd_arg}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); p.wait(); print(f'{{time.perf_counter() - start:.6f}},{{resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024:.3f}}')\""
    out = subprocess.check_output(py_cmd, shell=True).decode().strip()
    t, mem = out.split(',')
    return float(t), float(mem)

def run_suite(name, cmd, cwd=None, runs=3):
    times = []
    mems = []
    print(f"[*] Benchmarking: {name} ({runs} runs)")
    for i in range(runs):
        t, mem = run_benchmark(cmd, cwd)
        times.append(t)
        mems.append(mem)
        print(f"    Run {i+1}: {t:.3f}s | {mem:.2f} MB")
    
    t_mean, t_std = np.mean(times), np.std(times)
    m_mean, m_std = np.mean(mems), np.std(mems)
    return {
        "t_mean": t_mean, "t_std": t_std,
        "m_mean": m_mean, "m_std": m_std
    }

if __name__ == "__main__":
    runs = 3
    print("==========================================================")
    # 100k scale
    print("--- 100,000 Constraints (dot_product) ---")
    y_100k = run_suite("Y-lang (100k)", "./target/release/Y dot_product.ysu --target=r1cs", runs=runs)
    circom_100k = run_suite("Circom (100k)", "circom dot_product.circom --r1cs --c --sym --O2", runs=runs)
    
    # 1M scale
    print("\n--- 1,000,000 Constraints (heavy_circuit) ---")
    y_1m = run_suite("Y-lang (1M)", "./target/release/Y heavy_circuit.ysu --target=r1cs", runs=runs)
    circom_1m = run_suite("Circom (1M)", "circom heavy_circuit.circom --r1cs --c --sym --O2", runs=runs)
    
    print("\n==========================================================")
    print("                    BENCHMARK RESULTS                      ")
    print("==========================================================")
    print(f"Scale: 100,000 Constraints (dot_product)")
    print(f"  Y-lang: {y_100k['t_mean']:.3f}s ± {y_100k['t_std']:.3f}s | Peak RSS: {y_100k['m_mean']:.2f} MB ± {y_100k['m_std']:.2f} MB")
    print(f"  Circom: {circom_100k['t_mean']:.3f}s ± {circom_100k['t_std']:.3f}s | Peak RSS: {circom_100k['m_mean']:.2f} MB ± {circom_100k['m_std']:.2f} MB")
    speedup_100k = circom_100k['t_mean'] / y_100k['t_mean']
    mem_saving_100k = circom_100k['m_mean'] / y_100k['m_mean']
    print(f"  -> Y-lang speedup: {speedup_100k:.2f}x | Memory reduction: {mem_saving_100k:.2f}x")
    
    print(f"\nScale: 1,000,000 Constraints (heavy_circuit)")
    print(f"  Y-lang: {y_1m['t_mean']:.3f}s ± {y_1m['t_std']:.3f}s | Peak RSS: {y_1m['m_mean']:.2f} MB ± {y_1m['m_std']:.2f} MB")
    print(f"  Circom: {circom_1m['t_mean']:.3f}s ± {circom_1m['t_std']:.3f}s | Peak RSS: {circom_1m['m_mean']:.2f} MB ± {circom_1m['m_std']:.2f} MB")
    speedup_1m = circom_1m['t_mean'] / y_1m['t_mean']
    mem_saving_1m = circom_1m['m_mean'] / y_1m['m_mean']
    print(f"  -> Y-lang speedup: {speedup_1m:.2f}x | Memory reduction: {mem_saving_1m:.2f}x")
    
    print("\n==========================================================")
    print("                     SCALING CURVE                         ")
    print("==========================================================")
    print(f"Compilation Time Scalability (100k -> 1M):")
    y_time_growth = y_1m['t_mean'] / y_100k['t_mean']
    circom_time_growth = circom_1m['t_mean'] / circom_100k['t_mean']
    print(f"  Y-lang time growth: {y_time_growth:.2f}x (from {y_100k['t_mean']:.3f}s to {y_1m['t_mean']:.3f}s)")
    print(f"  Circom time growth: {circom_time_growth:.2f}x (from {circom_100k['t_mean']:.3f}s to {circom_1m['t_mean']:.3f}s)")
    print(f"  -> Y-lang advantage scaling factor (speedup 1M / 100k): {speedup_1m / speedup_100k:.2f}x")
    
    print(f"\nMemory Footprint Scalability (100k -> 1M):")
    y_mem_growth = y_1m['m_mean'] / y_100k['m_mean']
    circom_mem_growth = circom_1m['m_mean'] / circom_100k['m_mean']
    print(f"  Y-lang memory growth: {y_mem_growth:.2f}x (from {y_100k['m_mean']:.2f} MB to {y_1m['m_mean']:.2f} MB)")
    print(f"  Circom memory growth: {circom_mem_growth:.2f}x (from {circom_100k['m_mean']:.2f} MB to {circom_1m['m_mean']:.2f} MB)")
    print(f"  -> Y-lang memory saving advantage scaling factor: {mem_saving_1m / mem_saving_100k:.2f}x")
