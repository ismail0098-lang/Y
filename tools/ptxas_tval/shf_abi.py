#!/usr/bin/env python3
"""SHF.L.U32 (no .HI) = the low word of {Rc:Ra} << n, for n < 32 -- refereed.

`sassexec` models exactly that and leaves n >= 32 as a fresh unknown; the
corpus kernel `ptx_integer_ops` needs it for `shl.b32` by `b & 31`.  Checked on
the device against a host shift at all 32 amounts, after asserting the probe's
SASS carries the form the model is about -- a probe emitting another
instruction would referee nothing.  Needs nvcc, cuobjdump, a CUDA device."""
import os, re, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
def main():
    with tempfile.TemporaryDirectory(prefix='shf_') as t:
        exe = os.path.join(t, 'shf')
        r = subprocess.run(['nvcc', '-O2', '-arch=sm_89', '-o', exe, os.path.join(HERE, 'shf_abi.cu')],
                           capture_output=True, text=True)
        if r.returncode: print('FAIL: nvcc', r.stderr); return 1
        sass = subprocess.run(['cuobjdump', '-sass', exe], capture_output=True, text=True, check=True).stdout
        if not re.search(r'\bSHF\.L\.U32\s+R\d+, R\d+, R\d+, RZ', sass):
            print('FAIL: the probe SASS has no `SHF.L.U32 Rd, Ra, Rn, RZ`; it measures another instruction'); return 1
        corpus = open(os.path.join(HERE, 'corpus', 'ptx_integer_ops.sass')).read()
        if 'SHF.L.U32 ' not in corpus:
            print('FAIL: the corpus kernel no longer uses SHF.L.U32; this referee referees nothing'); return 1
        out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout
        m = re.search(r'seen (\d+) amounts (\d+) bad (\d+)', out)
        if not m: print('FAIL: cannot read', out); return 1
        seen, amounts, bad = map(int, m.groups())
        if amounts != 32: print(f'FAIL: only {amounts} of 32 amounts exercised'); return 1
        if bad: print(f'FAIL: {bad} of {seen} disagree with a host shift'); return 1
        print(f'ok   SHF.L.U32 = Ra << n for every n < 32 ({seen} cases, 0 disagreements)')
        return 0
if __name__ == '__main__': sys.exit(main())
