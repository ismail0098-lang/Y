"""Referee the const-bank launch-geometry ABI against the DEVICE.

`batch.mk` maps constant-bank offsets to symbol names; `sassexec` resolves
`c[0x0][0xc]` through that map and `ptxexec` resolves `%nctaid.x` to the same
symbol.  If the map is wrong, the two sides name different quantities and an
obligation is meaningless -- and NO currently-passing kernel reads these slots,
which is exactly why the block was missing in the first place and why no
regression result can see a mistake in it.

Establishing the offsets by compiling with `ptxas` and reading its output would
use the translator under test to license a fact used to validate that
translator.  This is not circular, because it is two independent steps and the
launch is what breaks the circle:

  (a) ptxas reads offset X for %<reg>            -- from the disassembly
  (b) %<reg> returns <extent> on the device      -- from a real launch

The six extents are DISTINCT, so if (a) were wrong the launch in (b) would
return another axis's value and this fails.  Together they say: whatever the
driver places at offset X is <extent>.

Run:  python3 cbank_abi.py        (needs ptxas, nvdisasm, gcc, a GPU)
"""
import re, subprocess, sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or '.')
import batch, mulmode

GEOM = {'ntid_x': 11, 'ntid_y': 13, 'ntid_z': 2,
        'nctaid_x': 3, 'nctaid_y': 5, 'nctaid_z': 7}   # matches cbank_abi.c

def main():
    # NON-VACUITY: the whole argument is that a wrong slot returns ANOTHER
    # axis's value.  With two extents equal, a swap between them is invisible
    # and every check below passes while asserting nothing.
    if len(set(GEOM.values())) != len(GEOM):
        print('FAIL: the launch extents are not distinct -- a swap would be invisible')
        return 1
    here = os.path.dirname(os.path.abspath(__file__)) or '.'
    os.chdir(here)
    sym = batch.mk(mulmode.MODES['wide'](), {})
    cb = sym['cbank']
    # the map, read back OUT of batch.py -- never restated here
    want = {name: off for off, name in cb.items()
            if isinstance(name, str) and name in GEOM}
    missing = set(GEOM) - set(want)
    if missing:
        print(f'FAIL: batch.mk names no const-bank slot for {sorted(missing)}')
        return 1
    order = sorted(GEOM)                      # deterministic store order
    body = '\n'.join(
        f'  mov.u32 %r{i+1}, %{n.replace("_",".",1) if False else n.split("_")[0]}.{n.split("_")[1]};'
        for i, n in enumerate(order))
    sts = '\n'.join(f'  st.global.u32 [%rd2+{4*i}], %r{i+1};' for i in range(len(order)))
    ptx = f""".version 7.8
.target sm_89
.address_size 64
.visible .entry probe(.param .u64 O)
{{
  .reg .b32 %r<{len(order)+1}>;
  .reg .b64 %rd<4>;
  ld.param.u64 %rd1, [O];
  cvta.to.global.u64 %rd2, %rd1;
{body}
{sts}
  ret;
}}
"""
    d = tempfile.mkdtemp(prefix=f'cbank_{os.getpid()}_')   # per-process
    p, cub, sas = f'{d}/p.ptx', f'{d}/p.cubin', f'{d}/p.sass'
    open(p, 'w').write(ptx)
    subprocess.run(['ptxas', '-arch=sm_89', '-o', cub, p], check=True)
    sas_txt = subprocess.run(['nvdisasm', '-c', cub], capture_output=True,
                             text=True, check=True).stdout

    # (a) which const-bank slot does ptxas read for each special register?
    #     recovered through the STORE, so the register renaming is irrelevant.
    reg_of = {}
    for m in re.finditer(r'\b(?:MOV|IMAD\.MOV\.U32)\s+(R\d+),\s*(?:RZ,\s*RZ,\s*)?c\[0x0\]\[0x([0-9a-f]+)\]', sas_txt):
        reg_of[m.group(1)] = int(m.group(2), 16)
    st_slot = {}
    for m in re.finditer(r'STG\.E\s+\[R\d+\.64(?:\+0x([0-9a-f]+))?\],\s*(R\d+)', sas_txt):
        off = int(m.group(1), 16) if m.group(1) else 0
        if m.group(2) in reg_of: st_slot[off // 4] = reg_of[m.group(2)]

    bad = 0
    for i, n in enumerate(order):
        got, exp = st_slot.get(i), want[n]
        ok = got == exp
        bad |= not ok
        print(f'  {n:9s} ptxas reads c[0x0][0x{got:02x}]' if got is not None
              else f'  {n:9s} ptxas slot NOT RECOVERED', end='')
        print(f'   map says 0x{exp:02x}   {"ok" if ok else "MISMATCH"}')
    if bad:
        print('FAIL: batch.mk disagrees with what ptxas reads')
        return 1

    # (b) does the DEVICE put the launched extent there?
    exe = f'{d}/drv'
    inc = next((x for x in ('/opt/cuda/include', '/usr/local/cuda/include')
                if os.path.exists(x + '/cuda.h')), None)
    if inc is None:
        print('  (no cuda.h -- step (b) skipped, so the offsets are ptxas-derived only)')
        return 1
    subprocess.run(['gcc', '-o', exe, 'cbank_abi.c', '-I', inc, '-lcuda'], check=True)
    out = subprocess.run([exe, cub], capture_output=True, text=True, check=True).stdout
    vals = [int(x) for x in out.split()]
    if len(vals) != len(order):
        print(f'FAIL: driver returned {len(vals)} values, expected {len(order)}')
        return 1
    for i, n in enumerate(order):
        ok = vals[i] == GEOM[n]
        bad |= not ok
        print(f'  {n:9s} device returned {vals[i]:3d}   launched {GEOM[n]:3d}   {"ok" if ok else "MISMATCH"}')
    print('FAIL: the device disagrees' if bad else
          'the const-bank map in batch.py agrees with ptxas AND with the device')
    return 1 if bad else 0

if __name__ == '__main__':
    sys.exit(main())
