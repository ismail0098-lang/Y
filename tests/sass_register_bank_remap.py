#!/usr/bin/env python3
"""SASS register-bank-conflict remap prototype: ptxas differential probe -> dense-FMA
kernel -> binary SASS register mutator -> CUDA Driver API correctness + timing."""
import argparse
import ctypes
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

ARCH = "sm_89"
PTXAS = shutil.which("ptxas") or "/opt/cuda/bin/ptxas"
NVDISASM = shutil.which("nvdisasm") or "/opt/cuda/bin/nvdisasm"

LINE_RE = re.compile(
    r"/\*(?P<off>[0-9a-fA-F]+)\*/\s*"
    r"(?:@!?P\d+\s+)?"
    r"(?P<mnem>[A-Z][A-Z0-9_.]*)\s+"
    r"(?P<ops>[^;]*);\s*"
    r"/\*\s*(?P<word0>0x[0-9a-fA-F]+)\s*\*/"
)
WORD1_RE = re.compile(r"^\s*/\*\s*(0x[0-9a-fA-F]+)\s*\*/\s*$")
REG_RE = re.compile(r"\bR(\d+)\b")

def log(msg):
    print(msg, flush=True)

def ptxas_compile(ptx_src, workdir, name):
    ptx_path = os.path.join(workdir, name + ".ptx")
    cubin_path = os.path.join(workdir, name + ".cubin")
    with open(ptx_path, "w") as f:
        f.write(ptx_src)
    r = subprocess.run([PTXAS, f"-arch={ARCH}", "-O1", "-o", cubin_path, ptx_path],
                        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ptxas failed compiling {name}:\n{r.stderr}")
    with open(cubin_path, "rb") as f:
        return f.read()

def disassemble(cubin_bytes, workdir, name):
    cubin_path = os.path.join(workdir, name + ".cubin")
    with open(cubin_path, "wb") as f:
        f.write(cubin_bytes)
    out = subprocess.run([NVDISASM, "-hex", cubin_path], check=True,
                          capture_output=True, text=True).stdout
    instrs = []
    lines = out.splitlines()
    for i, line in enumerate(lines):
        m = LINE_RE.search(line)
        if not m:
            continue
        word1 = None
        if i + 1 < len(lines):
            m2 = WORD1_RE.match(lines[i + 1])
            if m2:
                word1 = int(m2.group(1), 16)
        instrs.append({
            "offset": int(m.group("off"), 16),
            "mnemonic": m.group("mnem"),
            "operands": m.group("ops").strip(),
            "word0": int(m.group("word0"), 16),
            "word1": word1,
        })
    return instrs, out

def extract_regs(ops_text):
    return [int(x) for x in REG_RE.findall(ops_text)]

def find_section(cubin, name):
    # minimal hand-rolled ELF64 section header walk (no pyelftools dependency)
    e_shoff, = struct.unpack_from("<Q", cubin, 0x28)
    e_shentsize, = struct.unpack_from("<H", cubin, 0x3a)
    e_shnum, = struct.unpack_from("<H", cubin, 0x3c)
    e_shstrndx, = struct.unpack_from("<H", cubin, 0x3e)

    def shdr(i):
        base = e_shoff + i * e_shentsize
        sh_name, _sh_type = struct.unpack_from("<II", cubin, base)
        _flags, _addr, sh_offset, sh_size = struct.unpack_from("<QQQQ", cubin, base + 8)
        return sh_name, sh_offset, sh_size

    strtab_name, strtab_off, _ = shdr(e_shstrndx)
    for i in range(e_shnum):
        sh_name, sh_offset, sh_size = shdr(i)
        end = cubin.index(b"\0", strtab_off + sh_name)
        if cubin[strtab_off + sh_name:end].decode() == name:
            return sh_offset, sh_size
    raise RuntimeError(f"ELF section {name} not found in cubin")

# Phase 1: ptxas differential probe / oracle bit-solver

def oracle_kernel_ptx(n):
    # n independent d_i=a_i*b_i+c_i FFMAs; every operand is also stored back out so the
    # allocator can't reclaim one as Rd, keeping Rd/Ra/Rb genuinely distinct per instruction.
    lines = [
        ".version 7.8", f".target {ARCH}", ".address_size 64", "",
        ".visible .entry oracle_probe(", "  .param .u64 in_ptr,", "  .param .u64 out_ptr", ")", "{",
        f"  .reg .f32 %f<{4 * n}>;", "  .reg .b64 %rd<8>;",
        "  ld.param.u64 %rd1, [in_ptr];", "  ld.param.u64 %rd2, [out_ptr];",
        "  cvta.to.global.u64 %rd3, %rd1;", "  cvta.to.global.u64 %rd4, %rd2;",
    ]
    for i in range(n):
        lines.append(f"  ld.global.f32 %f{i}, [%rd3+{i * 4}];")
        lines.append(f"  ld.global.f32 %f{n + i}, [%rd3+{(n + i) * 4}];")
        lines.append(f"  ld.global.f32 %f{2 * n + i}, [%rd3+{(2 * n + i) * 4}];")
    for i in range(n):
        lines.append(f"  fma.rn.f32 %f{3 * n + i}, %f{i}, %f{n + i}, %f{2 * n + i};")
    for i in range(n):
        lines.append(f"  st.global.f32 [%rd4+{i * 4}], %f{3 * n + i};")
        lines.append(f"  st.global.f32 [%rd4+{(n + i) * 4}], %f{i};")
        lines.append(f"  st.global.f32 [%rd4+{(2 * n + i) * 4}], %f{n + i};")
        lines.append(f"  st.global.f32 [%rd4+{(3 * n + i) * 4}], %f{2 * n + i};")
    lines += ["  ret;", "}"]
    return "\n".join(lines)

def solve_field(samples):
    # samples: list[(regnum, word0)]. Brute-force the (shift, width) bitfield that decodes
    # to the exact register number for every sample; tiebreak on least out-of-field variation.
    uniq = list({(r, w) for r, w in samples})
    if len({r for r, _ in uniq}) < 6:
        raise RuntimeError("not enough distinct register samples to solve field reliably")
    candidates = []
    for width in range(5, 11):
        mask_bits = (1 << width) - 1
        for shift in range(0, 58):
            if all((w >> shift) & mask_bits == r for r, w in uniq):
                candidates.append((shift, width))
    if not candidates:
        raise RuntimeError("no consistent bitfield found for register operand")

    def spurious(sw):
        shift, width = sw
        mask = ((1 << width) - 1) << shift
        return len({w & ~mask for _, w in uniq})

    shift, width = min(candidates, key=lambda sw: (spurious(sw), sw[1]))
    return shift, width

def discover_register_fields(workdir):
    samples = {}
    for n in range(4, 12):
        cubin = ptxas_compile(oracle_kernel_ptx(n), workdir, f"oracle{n}")
        instrs, _ = disassemble(cubin, workdir, f"oracle{n}_dis")
        for ins in instrs:
            regs = extract_regs(ins["operands"])
            if ins["mnemonic"] == "FFMA" and len(regs) == 4 and len(set(regs)) == 4:
                for idx in (0, 1, 2):  # Rd, Ra, Rb - whichever slot the accumulator lands in
                    samples.setdefault(("FFMA", idx), []).append((regs[idx], ins["word0"]))
            elif ins["mnemonic"] == "LDG.E" and regs:
                samples.setdefault(("LDG.E", 0), []).append((regs[0], ins["word0"]))
            elif ins["mnemonic"] == "STG.E" and len(regs) == 2:
                samples.setdefault(("STG.E", 1), []).append((regs[1], ins["word0"]))

    fields = {}
    for key in [("FFMA", 0), ("FFMA", 1), ("FFMA", 2), ("LDG.E", 0), ("STG.E", 1)]:
        shift, width = solve_field(samples[key])
        fields[key] = (shift, width)
        mask = ((1 << width) - 1) << shift
        log(f"[+] Register Bitmask Discovered: {key[0]} src-operand#{key[1]} -> "
            f"0x{mask:016x}  (shift={shift}, width={width}b)")
    return fields

# Phase 2: target dense-FMA kernel

def target_kernel_ptx(num_acc, name):
    x_reg, y_reg = num_acc, num_acc + 1
    slice_bytes = num_acc * 4
    lines = [
        ".version 7.8", f".target {ARCH}", ".address_size 64", "",
        f".visible .entry {name}(",
        "  .param .u64 seed_ptr,", "  .param .u64 out_ptr,",
        "  .param .u32 n_iters,", "  .param .f32 x_val,", "  .param .f32 y_val", ")", "{",
        f"  .reg .f32 %f<{num_acc + 2}>;", "  .reg .b32 %r<8>;", "  .reg .b64 %rd<10>;",
        "  .reg .pred %p<2>;", "",
        "  ld.param.u64 %rd1, [seed_ptr];", "  ld.param.u64 %rd2, [out_ptr];",
        "  ld.param.u32 %r1, [n_iters];",
        f"  ld.param.f32 %f{x_reg}, [x_val];", f"  ld.param.f32 %f{y_reg}, [y_val];",
        "  mov.u32 %r2, %ctaid.x;", "  mov.u32 %r3, %ntid.x;", "  mov.u32 %r4, %tid.x;",
        "  mad.lo.u32 %r5, %r2, %r3, %r4;",
        f"  mul.wide.u32 %rd3, %r5, {slice_bytes};",
        "  cvta.to.global.u64 %rd4, %rd1;", "  add.s64 %rd5, %rd4, %rd3;",
        "  cvta.to.global.u64 %rd6, %rd2;", "  add.s64 %rd7, %rd6, %rd3;",
    ]
    for i in range(num_acc):
        lines.append(f"  ld.global.f32 %f{i}, [%rd5+{i * 4}];")
    lines.append("  mov.u32 %r6, 0;")
    lines.append("LOOP_TOP:")
    for i in range(num_acc):
        lines.append(f"  fma.rn.f32 %f{i}, %f{x_reg}, %f{i}, %f{y_reg};")
    lines += ["  add.u32 %r6, %r6, 1;", "  setp.lt.u32 %p1, %r6, %r1;", "  @%p1 bra LOOP_TOP;"]
    for i in range(num_acc):
        lines.append(f"  st.global.f32 [%rd7+{i * 4}], %f{i};")
    lines += ["  ret;", "}"]
    return "\n".join(lines)

# Phase 3: SASS register mutator

def set_field(word0, shift, width, old_val, new_val):
    mask = ((1 << width) - 1) << shift
    cur = (word0 >> shift) & ((1 << width) - 1)
    if cur != old_val:
        raise RuntimeError(f"field mismatch: expected R{old_val} at shift={shift}, found R{cur}")
    if new_val >= (1 << width):
        raise RuntimeError(f"new register R{new_val} does not fit the discovered {width}-bit field")
    return (word0 & ~mask) | ((new_val << shift) & mask)

def build_patched_cubin(cubin, instrs, raw_text, fields, stride, kernel_name):
    ffmas = [ins for ins in instrs if ins["mnemonic"] == "FFMA"]
    acc_regs = sorted({extract_regs(ins["operands"])[0] for ins in ffmas})
    n = len(acc_regs)
    log(f"[+] Found {n} accumulator registers: {['R%d' % r for r in acc_regs]}")

    m = re.search(r"SHI_REGISTERS=(\d+)", raw_text)
    regcount = int(m.group(1)) if m else 256
    all_used = set()
    for ins in instrs:
        all_used.update(extract_regs(ins["operands"]))
    other_used = all_used - set(acc_regs)

    def plan(stride_try):
        base = acc_regs[0]
        candidate = [base + stride_try * i for i in range(n)]
        bad = len(set(candidate)) != n or any(r >= regcount or r in other_used for r in candidate)
        return None if bad else candidate

    chosen_stride = stride
    new_regs = plan(chosen_stride)
    while new_regs is None and chosen_stride > 1:
        chosen_stride -= 1
        new_regs = plan(chosen_stride)
    if new_regs is None:
        new_regs = list(acc_regs)
        log("[!] No safe register spread fits the register budget; remap degrades to a no-op")
    else:
        if new_regs == acc_regs:
            log(f"[i] Requested stride={stride} found no spread beyond ptxas's own default "
                f"spacing (R{acc_regs[0]}..R{acc_regs[-1]}) within the register budget - "
                f"compacting to adjacent registers instead so the patch has a real before/after")
            chosen_stride = 1
            new_regs = plan(1) or list(acc_regs)
        log(f"[+] Remap plan (stride={chosen_stride}, budget={regcount} regs): " +
            ", ".join(f"R{o}->R{nw}" for o, nw in zip(acc_regs, new_regs)))
    remap = dict(zip(acc_regs, new_regs))

    patches = []
    for old_r, new_r in remap.items():
        if old_r == new_r:
            continue
        ldg = [i for i in instrs if i["mnemonic"] == "LDG.E" and extract_regs(i["operands"])[:1] == [old_r]]
        ffma = [i for i in instrs if i["mnemonic"] == "FFMA" and old_r in extract_regs(i["operands"])[:3]]
        stg = [i for i in instrs if i["mnemonic"] == "STG.E" and extract_regs(i["operands"])[1:2] == [old_r]]
        if len(ldg) != 1 or len(ffma) != 1 or len(stg) != 1:
            raise RuntimeError(
                f"expected exactly one LDG.E/FFMA/STG.E referencing R{old_r}, "
                f"found {len(ldg)}/{len(ffma)}/{len(stg)} - kernel shape assumption broke")

        shift, width = fields[("LDG.E", 0)]
        patches.append((ldg[0]["offset"], set_field(ldg[0]["word0"], shift, width, old_r, new_r)))

        # the accumulator can land at any of FFMA's Rd/Ra/Rb slots depending on how ptxas
        # canonicalizes the commutative multiply - patch whichever slot(s) actually hold it
        w0 = ffma[0]["word0"]
        ffma_regs = extract_regs(ffma[0]["operands"])[:3]
        for idx, r in enumerate(ffma_regs):
            if r == old_r:
                shift, width = fields[("FFMA", idx)]
                w0 = set_field(w0, shift, width, old_r, new_r)
        patches.append((ffma[0]["offset"], w0))

        shift, width = fields[("STG.E", 1)]
        patches.append((stg[0]["offset"], set_field(stg[0]["word0"], shift, width, old_r, new_r)))

    sec_off, _sec_size = find_section(cubin, f".text.{kernel_name}")
    buf = bytearray(cubin)
    for rel_off, new_word0 in patches:
        struct.pack_into("<Q", buf, sec_off + rel_off, new_word0)
    log(f"[+] Patched {len(patches)} instruction words directly in the .text.{kernel_name} section")
    return bytes(buf), remap

def verify_patch(patched_cubin, workdir, remap):
    instrs, _ = disassemble(patched_cubin, workdir, "patched_verify")
    ffmas = [i for i in instrs if i["mnemonic"] == "FFMA"]
    got = sorted({extract_regs(i["operands"])[0] for i in ffmas})
    expected = sorted(set(remap.values()))
    if got != expected:
        raise RuntimeError(f"post-patch re-disassembly mismatch: FFMA dest regs {got} != expected {expected}")
    log(f"[+] Post-patch re-disassembly confirms accumulators now at {['R%d' % r for r in got]}")

# Phase 4: CUDA Driver API harness + hardware verification

class CudaError(RuntimeError):
    pass

class Cuda:
    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self._set_signatures()
        self._check(self.lib.cuInit(0))
        dev = ctypes.c_int()
        self._check(self.lib.cuDeviceGet(ctypes.byref(dev), 0))
        self.context = ctypes.c_void_p()
        self._check(self.lib.cuCtxCreate_v2(ctypes.byref(self.context), 0, dev))

    def _set_signatures(self):
        vp, ipp, cp, u, i = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
                             ctypes.c_char_p, ctypes.c_uint, ctypes.c_int)
        sigs = {
            "cuGetErrorString": [i, ctypes.POINTER(cp)], "cuInit": [u],
            "cuDeviceGet": [ctypes.POINTER(i), i], "cuCtxCreate_v2": [ipp, u, i],
            "cuModuleLoadData": [ipp, vp], "cuModuleUnload": [vp],
            "cuModuleGetFunction": [ipp, vp, cp],
            "cuMemAlloc_v2": [ipp, ctypes.c_size_t], "cuMemFree_v2": [vp],
            "cuMemcpyHtoD_v2": [vp, vp, ctypes.c_size_t], "cuMemcpyDtoH_v2": [vp, vp, ctypes.c_size_t],
            "cuLaunchKernel": [vp, u, u, u, u, u, u, u, vp, vp, vp],
            "cuEventCreate": [ipp, u], "cuEventDestroy_v2": [vp],
            "cuEventRecord": [vp, vp], "cuEventSynchronize": [vp],
            "cuEventElapsedTime": [ctypes.POINTER(ctypes.c_float), vp, vp], "cuCtxSynchronize": [],
        }
        for name, argtypes in sigs.items():
            getattr(self.lib, name).argtypes = argtypes

    def _check(self, err):
        if err != 0:
            msg = ctypes.c_char_p()
            self.lib.cuGetErrorString(err, ctypes.byref(msg))
            raise CudaError(f"CUDA driver error {err}: {msg.value.decode() if msg.value else '?'}")

    def load_module(self, cubin_bytes):
        mod = ctypes.c_void_p()
        buf = ctypes.create_string_buffer(cubin_bytes, len(cubin_bytes))
        self._check(self.lib.cuModuleLoadData(ctypes.byref(mod), buf))
        return mod

    def unload_module(self, mod): self._check(self.lib.cuModuleUnload(mod))

    def get_function(self, module, name):
        fn = ctypes.c_void_p()
        self._check(self.lib.cuModuleGetFunction(ctypes.byref(fn), module, name.encode()))
        return fn

    def mem_alloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self._check(self.lib.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(nbytes)))
        return ptr

    def mem_free(self, ptr): self._check(self.lib.cuMemFree_v2(ptr))

    def memcpy_htod(self, dptr, host_arr):
        self._check(self.lib.cuMemcpyHtoD_v2(dptr, host_arr.ctypes.data_as(ctypes.c_void_p),
                                              ctypes.c_size_t(host_arr.nbytes)))

    def memcpy_dtoh(self, host_arr, dptr):
        self._check(self.lib.cuMemcpyDtoH_v2(host_arr.ctypes.data_as(ctypes.c_void_p), dptr,
                                              ctypes.c_size_t(host_arr.nbytes)))

    def launch(self, fn, grid, block, params):
        ptrs = (ctypes.c_void_p * len(params))(
            *[ctypes.cast(ctypes.pointer(p), ctypes.c_void_p) for p in params])
        self._check(self.lib.cuLaunchKernel(
            fn, grid[0], grid[1], grid[2], block[0], block[1], block[2],
            0, None, ptrs, None))

    def synchronize(self): self._check(self.lib.cuCtxSynchronize())

    def event_create(self):
        ev = ctypes.c_void_p()
        self._check(self.lib.cuEventCreate(ctypes.byref(ev), 0))
        return ev

    def event_destroy(self, ev): self._check(self.lib.cuEventDestroy_v2(ev))
    def event_record(self, ev): self._check(self.lib.cuEventRecord(ev, None))
    def event_sync(self, ev): self._check(self.lib.cuEventSynchronize(ev))

    def event_elapsed_ms(self, start, end):
        ms = ctypes.c_float()
        self._check(self.lib.cuEventElapsedTime(ctypes.byref(ms), start, end)); return ms.value

def time_kernel(cuda, fn, params, grid, block, iterations):
    cuda.launch(fn, grid, block, params)  # warmup
    cuda.synchronize()
    start, end = cuda.event_create(), cuda.event_create()
    cuda.event_record(start)
    for _ in range(iterations):
        cuda.launch(fn, grid, block, params)
    cuda.event_record(end)
    cuda.event_sync(end)
    total_ms = cuda.event_elapsed_ms(start, end)
    cuda.event_destroy(start)
    cuda.event_destroy(end)
    return total_ms / iterations

def run_hardware_verification(base_cubin, patched_cubin, kernel_name, args):
    cuda = Cuda()
    num_threads = args.blocks * args.threads
    rng = np.random.default_rng(0)
    seed = rng.uniform(0.5, 1.5, size=num_threads * args.num_acc).astype(np.float32)
    x_val, y_val = np.float32(1.0000437), np.float32(0.00021)
    out_size = num_threads * args.num_acc
    grid, block = (args.blocks, 1, 1), (args.threads, 1, 1)

    def setup(cubin):
        mod = cuda.load_module(cubin)
        fn = cuda.get_function(mod, kernel_name)
        d_seed = cuda.mem_alloc(seed.nbytes)
        d_out = cuda.mem_alloc(out_size * 4)
        cuda.memcpy_htod(d_seed, seed)
        params = [d_seed, d_out, ctypes.c_uint32(args.n_iters), ctypes.c_float(x_val), ctypes.c_float(y_val)]
        return mod, fn, d_seed, d_out, params

    def read_output(fn, params, d_out):
        cuda.launch(fn, grid, block, params)
        cuda.synchronize()
        out = np.empty(out_size, dtype=np.float32)
        cuda.memcpy_dtoh(out, d_out)
        return out

    mod_b, fn_b, d_seed_b, d_out_b, params_b = setup(base_cubin)
    mod_p, fn_p, d_seed_p, d_out_p, params_p = setup(patched_cubin)

    out_base = read_output(fn_b, params_b, d_out_b)
    out_patched = read_output(fn_p, params_p, d_out_p)

    if not (np.all(np.isfinite(out_base)) and np.all(np.isfinite(out_patched))):
        raise RuntimeError("correctness check FAILED: non-finite values in kernel output")
    max_diff = float(np.max(np.abs(out_patched.astype(np.float64) - out_base.astype(np.float64))))
    log(f"[+] Hardware Correctness Asserted: Max Diff = {max_diff:.6f}")
    if max_diff >= 1e-4:
        raise RuntimeError(f"correctness check FAILED: max_abs_diff={max_diff} >= 1e-4")

    # GPU clocks ramp up over the first several launches of a fresh context; timing
    # base then patched back-to-back without this would systematically favor whichever
    # runs second. Warm both up together first, then measure in alternating rounds and
    # take the median so any leftover drift affects both variants equally.
    for _ in range(30):
        cuda.launch(fn_b, grid, block, params_b)
        cuda.launch(fn_p, grid, block, params_p)
    cuda.synchronize()

    rounds = 5
    base_times, patched_times = [], []
    for _ in range(rounds):
        base_times.append(time_kernel(cuda, fn_b, params_b, grid, block, args.timing_iters))
        patched_times.append(time_kernel(cuda, fn_p, params_p, grid, block, args.timing_iters))
    ms_base = float(np.median(base_times))
    ms_patched = float(np.median(patched_times))

    for ptr in (d_seed_b, d_out_b, d_seed_p, d_out_p):
        cuda.mem_free(ptr)
    for mod in (mod_b, mod_p):
        cuda.unload_module(mod)

    delta_pct = (ms_patched - ms_base) / ms_base * 100.0
    log(f"    unpatched per-round ms (median={ms_base:.3f}): " + ", ".join(f"{t:.3f}" for t in base_times))
    log(f"    remapped per-round ms  (median={ms_patched:.3f}): " + ", ".join(f"{t:.3f}" for t in patched_times))
    log(f"[+] Unpatched SASS Time: {ms_base:.3f} ms")
    log(f"[+] Remapped SASS Time: {ms_patched:.3f} ms (Delta: {delta_pct:+.2f}%)")
    return max_diff, ms_base, ms_patched, delta_pct

# main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-acc", type=int, default=4, help="independent accumulators in the FMA kernel")
    p.add_argument("--n-iters", type=int, default=50000, help="FMA loop trip count inside the kernel")
    p.add_argument("--timing-iters", type=int, default=100, help="kernel relaunches for the timing average")
    p.add_argument("--blocks", type=int, default=4); p.add_argument("--threads", type=int, default=32)
    p.add_argument("--stride", type=int, default=2, help="target register spread (R_k -> R_k+stride*i)")
    args = p.parse_args()

    kernel_name = "dense_fma_kernel"
    with tempfile.TemporaryDirectory(prefix="sass_remap_") as workdir:
        log("=== Phase 1: ptxas differential probe (oracle bit-solver) ===")
        fields = discover_register_fields(workdir)

        log("\n=== Phase 2: compiling target dense-FMA kernel ===")
        ptx_src = target_kernel_ptx(args.num_acc, kernel_name)
        base_cubin = ptxas_compile(ptx_src, workdir, "target")
        instrs, raw_text = disassemble(base_cubin, workdir, "target_dis")
        log(f"[+] Compiled {kernel_name}: {len(base_cubin)} bytes, {len(instrs)} SASS instructions")

        log("\n=== Phase 3: SASS register mutator ===")
        patched_cubin, remap = build_patched_cubin(base_cubin, instrs, raw_text, fields,
                                                    args.stride, kernel_name)
        verify_patch(patched_cubin, workdir, remap)

        log("\n=== Phase 4: hardware verification ===")
        run_hardware_verification(base_cubin, patched_cubin, kernel_name, args)

if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, CudaError) as e:
        log(f"[FATAL] {e}")
        sys.exit(1)
