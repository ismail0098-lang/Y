"""Referee the SUB-WORD global store and the u8 conversion against the DEVICE.

The byte-faithful memory model (`memorder.py`) is what makes a sub-word store
SOUND to model at all: a store carries its width, and a load after it reads the
bytes it covers.  Before it, comparing a sub-word store by address alone could
not tell an 8-bit store from a 32-bit one, so refusing was right.  Two facts the
executors then need cannot be read off a mnemonic, and both are measured here.

  STG.E.U8 / .S8 / .U16 / .S16 write EXACTLY their width, little-endian, at the
  byte address, and nothing else.  A store that wrote a whole word, or wrote the
  wrong end of the value, would be invisible on a zeroed buffer; so every output
  buffer is POISONED (0xAB) and each store lands INSIDE a lane's word at a byte
  offset, so the untouched bytes around it are part of the answer.

  cvt.u8.u32 TRUNCATES.  A u32 -> u8 conversion could as well saturate, and the
  two agree on every input below 256 -- which is every input a careless probe
  uses.  The vectors straddle that boundary and the check fails if none does.

WHAT THIS IS, AND IS NOT.  The SASS instructions are what `sassexec` models, and
running them is a measurement of them.  The PTX side's `cvt.u8.u32` is taken
from the same run -- the PTX opcode has no other executable definition here --
so for THAT ONE instruction a ptxas bug is in the trusted base, exactly as it is
for `max.f32` in `fpsem_abi.py`.  The probe asserts which SASS forms it emitted,
because a probe that ptxas lowered some other way measures some other instruction.

Run:  python3 subword_abi.py        (needs ptxas, nvdisasm, gcc, a GPU)
"""
import os, re, subprocess, sys, tempfile
import fpsem_abi as F

POISON = 0xAB
# The lane count is the C driver's, read from it rather than written twice.
LANES = int(re.search(r'#define\s+N\s+(\d+)', open(os.path.join(F.HERE, 'fpsem_abi.c')).read()).group(1))
VALUES = [0x12345678, 0xFFFFFFFF, 0x000001FF, 0x00000100, 0x000000FF, 0x00000080,
          0x0000007F, 0x00000000, 0xDEADBEEF, 0x8000FF01, 0x00FF00FF, 0x0000FFFF,
          0x00010000, 0x000080FF, 0x7FFFFFFF, 0x80000000]


def cvt_ptx():
    """r2 = cvt.u8.u32(r1), stored as a word; r1 echoed so the load/store path is
    shown to preserve bits before the conversion is blamed for anything."""
    return F.HEAD + F._addrs() + """    ld.global.u32 %r1, [%rd20];
    cvt.u8.u32 %r2, %r1;
    st.global.u32 [%rd23], %r2;
    st.global.u32 [%rd24], %r1;
    ret;
}
"""


def store_ptx(tag):
    """Three sub-word stores into three poisoned lane words, at byte offsets."""
    # Offsets are ones the ALIGNMENT census says do not fault: a 16-bit store at
    # an odd byte faults on the device, which is how that census came to exist.
    if tag == 'a':          # u8 @+1, u16 @+2, s8 @+3
        body = """    add.u64 %rd30, %rd23, 1;
    st.global.u8 [%rd30], %r1;
    add.u64 %rd31, %rd24, 2;
    st.global.u16 [%rd31], %r1;
    add.u64 %rd29, %rd25, 3;
    st.global.s8 [%rd29], %r1;
"""
    else:                   # s16 @+0, u8 @+3, s8 @+1
        body = """    st.global.s16 [%rd23], %r1;
    add.u64 %rd31, %rd24, 3;
    st.global.u8 [%rd31], %r1;
    add.u64 %rd29, %rd25, 1;
    st.global.s8 [%rd29], %r1;
"""
    return F.HEAD + F._addrs() + "    ld.global.u32 %r1, [%rd20];\n" + body + "    ret;\n}\n"


WIDTH_OP = {1: 'u8', 2: 'u16', 4: 'u32'}


def align_ptx(kind, width, off):
    """ONE access at a byte offset into a lane's word, guarded to skip the last
    lane so a wide access that crosses into the next lane's word can never cross
    the end of the buffer.  A fault is sticky in a CUDA context, so every case is
    its own kernel and its own process."""
    head = F.HEAD + F._addrs() + """    mov.u32 %r5, 31;
    setp.lt.u32 %p0, %r0, %r5;
"""
    base = '%rd23' if kind == 'st' else '%rd20'
    addr = base if off == 0 else '%rd30'
    at = '' if off == 0 else f'    add.u64 %rd30, {base}, {off};\n'
    if kind == 'st':
        body = f"""    ld.global.u32 %r1, [%rd21];
{at}    @%p0 st.global.{WIDTH_OP[width]} [{addr}], %r1;
"""
    else:
        body = f"""{at}    @%p0 ld.global.{WIDTH_OP[width]} %r2, [{addr}];
    @%p0 st.global.u32 [%rd23], %r2;
"""
    return head + body + "    ret;\n}\n"


def launch(exe, cub):
    stdin = '\n'.join('0 0 0' for _ in range(LANES))
    r = subprocess.run([exe, cub], input=stdin, capture_output=True, text=True)
    if r.returncode == 0 and 'FAIL' not in r.stdout:
        return 'ok'
    m = re.search(r'FAIL [^:]*: (.*)', r.stdout)
    return m.group(1).strip() if m else (r.stdout.strip() or r.stderr.strip())[:60]


def census_alignment(d, exe):
    """Which global accesses FAULT at which byte offsets.  Not a guess about the
    ISA: the executors model no fault at all, so a width/offset pair the device
    refuses is a program whose meaning the model does not describe."""
    rows = {}
    for kind in ('st', 'ld'):
        for width in (1, 2, 4):
            for off in range(4):
                cub, _ = F.build(d, f'al_{kind}{width}_{off}', align_ptx(kind, width, off))
                rows[(kind, width, off)] = launch(exe, cub)
    for kind in ('st', 'ld'):
        for width in (1, 2, 4):
            cells = ['  ok' if rows[(kind, width, o)] == 'ok' else 'FAULT' for o in range(4)]
            print(f'  {kind}.global.{WIDTH_OP[width]:<4} offset 0..3: ' + ' '.join(f'{c:>5}' for c in cells))
    msgs = sorted({v for v in rows.values() if v != 'ok'})
    print('  fault messages:', msgs or 'none')
    # ASSERTED, not printed: an access of width w runs iff its address is a
    # multiple of w.  Measured on sm_89.  A byte access is never misaligned, so
    # the census would be vacuous without the wide rows faulting somewhere.
    bad = [k for k, v in rows.items() if (v == 'ok') != (k[2] % k[1] == 0)]
    faults = sum(v != 'ok' for v in rows.values())
    if faults == 0:
        print('FAIL: nothing faulted -- the census cannot separate an aligned access from '
              'a misaligned one on this device'); return None
    for k in bad:
        print(f'FAIL: {k[0]}.global.{WIDTH_OP[k[1]]} at +{k[2]} was {rows[k]!r}; expected '
              f'{"ok" if k[2] % k[1] == 0 else "a misaligned-address fault"}')
    return None if bad else rows


# (offset in bytes, width in bytes) of the store into each of the three outputs
LAYOUT = {'a': [(1, 1), (2, 2), (3, 1)], 'b': [(0, 2), (3, 1), (1, 1)]}
FORMS = {'a': ['STG.E.U8', 'STG.E.U16', 'STG.E.S8'], 'b': ['STG.E.S16', 'STG.E.U8', 'STG.E.S8']}


def expect_word(v, off, width):
    word = [POISON] * 4
    for j in range(width):
        word[off + j] = (v >> (8 * j)) & 0xFF
    return sum(b << (8 * i) for i, b in enumerate(word))


def check_cvt(d, exe):
    cub, sass = F.build(d, 'cvt', cvt_ptx())
    lines = [l.strip() for l in sass.splitlines() if re.search(r'LOP3|STG|LDG', l)]
    print('  lowered to:', ' | '.join(re.sub(r'^/\*[0-9a-f]+\*/\s*', '', l) for l in lines))
    vals = (VALUES * 2)[:LANES]
    out = F.run(exe, cub, [(v, 0, 0) for v in vals])
    bad = 0
    if not any((v & 0xFF) != min(v, 0xFF) for v in vals):
        print('FAIL: no vector separates truncation from saturation'); return 1
    for v, (o0, o1, _o2) in zip(vals, out):
        if o1 != v:
            print(f'FAIL: the echo changed 0x{v:08x} to 0x{o1:08x} -- the load/store path, '
                  f'not the conversion'); bad += 1
        elif o0 != (v & 0xFF):
            sat = min(v, 0xFF)
            print(f'  cvt.u8.u32(0x{v:08x}) = 0x{o0:08x}; truncation 0x{v & 0xFF:08x}, '
                  f'saturation 0x{sat:08x}'); bad += 1
    sep = sum((v & 0xFF) != min(v, 0xFF) for v in vals)
    print(f'  {len(vals)} vectors, {sep} of them separate truncation from saturation, '
          f'{bad} disagree with truncation')
    return 1 if bad else 0


def check_stores(d, exe, tag):
    cub, sass = F.build(d, f'st{tag}', store_ptx(tag))
    got_forms = re.findall(r'\b(STG\.E\.[SU](?:8|16))\b', sass)
    print(f'  probe {tag}: SASS store forms {got_forms}')
    if sorted(got_forms) != sorted(FORMS[tag]):
        print(f'FAIL: the probe was lowered to {got_forms}, not {FORMS[tag]} -- it would '
              f'measure some other instruction'); return 1
    vals = (VALUES * 2)[:LANES]
    out = F.run(exe, cub, [(v, 0, 0) for v in vals])
    bad = 0
    for v, words in zip(vals, out):
        for (off, w), got in zip(LAYOUT[tag], words):
            want = expect_word(v, off, w)
            if got != want:
                if bad < 6:
                    print(f'  0x{v:08x}: {w} byte(s) at +{off} left 0x{got:08x}, '
                          f'a byte memory holds 0x{want:08x}')
                bad += 1
    print(f'  {len(vals)} lanes x 3 stores, {bad} disagree with a little-endian '
          f'byte memory')
    return 1 if bad else 0


def main():
    os.chdir(F.HERE)
    inc = next((x for x in ('/opt/cuda/include', '/usr/local/cuda/include')
                if os.path.exists(x + '/cuda.h')), None)
    if inc is None:
        print('SKIP: no cuda.h -- neither fact can be refereed without the device')
        return 0
    d = tempfile.mkdtemp(prefix=f'subword_{os.getpid()}_')
    exe = f'{d}/drv'
    subprocess.run(['gcc', '-O1', '-o', exe, 'fpsem_abi.c', '-I', inc, '-lcuda'], check=True)
    print('ALIGNMENT: which global accesses fault at which byte offset?')
    rc = 0 if census_alignment(d, exe) is not None else 1
    if os.environ.get('SUBWORD_CENSUS_ONLY'):
        return rc
    print('\ncvt.u8.u32: truncation or saturation?')
    rc |= check_cvt(d, exe)
    print('\nSTG.E.{U,S}{8,16}: exactly their bytes, little-endian, nothing else?')
    rc |= check_stores(d, exe, 'a')
    rc |= check_stores(d, exe, 'b')
    return rc


if __name__ == '__main__':
    sys.exit(main())
