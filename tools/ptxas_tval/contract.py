"""Where does ptxas ACTUALLY contract?

A kernel only needs the .rn repair if its SASS contains an FFMA that its PTX
did not ask for.  Counting float ops in the PTX (which is what "29 float
kernels" counts) is not that question.
"""
import glob,os,re


def measure():
    """(kernel, ptx counts..., sass counts..., extra FFMA) per corpus kernel.

    A LIBRARY function because `fpgate.py` needs the contraction set and a
    hardcoded copy of it there is the drift this whole directory is about --
    that list had already gone stale once."""
    rows=[]
    for pf in sorted(glob.glob('corpus/*.ptx')):
        k=os.path.basename(pf)[:-4]
        sf=f'corpus/{k}.sass'
        if not os.path.exists(sf): continue
        P=open(pf).read(); S=open(sf).read()
        # PTX float ops
        pmul=len(re.findall(r'^\s*(?:@\S+\s+)?mul(?:\.rn|\.rz|\.rm|\.rp)?\.f32\b',P,re.M))
        padd=len(re.findall(r'^\s*(?:@\S+\s+)?(?:add|sub)(?:\.rn|\.rz|\.rm|\.rp)?\.f32\b',P,re.M))
        pfma=len(re.findall(r'^\s*(?:@\S+\s+)?fma\.\w+\.f32\b',P,re.M))
        prn =len(re.findall(r'^\s*(?:@\S+\s+)?(?:mul|add|sub|fma)\.r[nzmp]\.f32\b',P,re.M))
        # SASS float ops
        sffma=len(re.findall(r'\bFFMA\b',S)); sfmul=len(re.findall(r'\bFMUL\b',S))
        sfadd=len(re.findall(r'\bFADD\b',S)); shmma=len(re.findall(r'\bHMMA\b',S))
        # a contraction happened iff SASS has more FFMA than the PTX asked for
        extra = sffma - pfma
        rows.append((k,pmul,padd,pfma,prn,sfmul,sfadd,sffma,shmma,extra))
    return rows

def contraction_kernels(_cache={}):
    """Kernels where ptxas actually fused a `mul.f32` into an `add.f32`.

    MEASURED, by forbidding the fusion with `.rn` and re-assembling: if the
    SASS moves, ptxas was fusing.  The obvious inference -- `FFMA(sass) -
    fma(ptx) > 0`, which is what `measure()` reports in its last column --
    is wrong in BOTH directions and was believed for a day:

      * it OVER-reports, because it cannot tell contraction from UNROLLING.
        One `fma.rn.f32` inside a loop becomes N FFMA when ptxas unrolls, and
        `paged_decode_attention_*` reads +32 to +56 that way while forbidding
        the fusion changes nothing.  `gemm_fp8_*` reads +11 from FFMA that
        ptxas synthesised for something else entirely.
      * it UNDER-reports, because it counts `FFMA` only.  The four
        `gemm_f16_swiglu_*` kernels fuse at HALF precision, so their SASS moves
        under `.rn` while their FFMA count is 0 either way.

    16 by inference, **9** by measurement.  Same shape as the hardcoded list
    this function was written to replace, one layer down: a derived number is
    not thereby a measured one."""
    if 'v' in _cache: return _cache['v']
    import subprocess, tempfile
    work = tempfile.mkdtemp(prefix='contract_')
    out = []
    for pf in sorted(glob.glob('corpus/*.ptx')):
        k = os.path.basename(pf)[:-4]
        P = open(pf).read()
        if not re.search(r'^\s*(?:@\S+\s+)?(?:mul|add|sub)\.f32\b', P, re.M): continue
        am = re.search(r'^\.target\s+(\S+)', P, re.M)
        if not am: continue
        rn = re.sub(r'^(\s*(?:@\S+\s+)?)(mul|add|sub)\.f32\b', r'\1\2.rn.f32', P, flags=re.M)
        def sass(src, tag):
            pp = os.path.join(work, f'{k}_{tag}.ptx'); open(pp,'w').write(src)
            cu = os.path.join(work, f'{k}_{tag}.cubin')
            if subprocess.run(['ptxas', f'-arch={am.group(1)}', '-o', cu, pp],
                              capture_output=True).returncode: return None
            return subprocess.run(['nvdisasm','-c',cu],capture_output=True,text=True).stdout
        a, b = sass(P,'as'), sass(rn,'rn')
        if a is not None and b is not None and a != b: out.append(k)
    _cache['v'] = out
    return out


if __name__ == '__main__':
    rows = measure()
    print(f'{"kernel":38s}{"ptx:mul":>8}{"add":>5}{"fma":>5}{".rn":>5} | '
          f'{"FMUL":>6}{"FADD":>6}{"FFMA":>6}{"HMMA":>6}{"contracted":>12}')
    tot=0; nk=0
    for r in rows:
        k,pmul,padd,pfma,prn,sfmul,sfadd,sffma,shmma,extra=r
        if pmul+padd+pfma+sffma+sfmul+sfadd+shmma==0: continue
        flag = f'{extra:+d}' if extra>0 else ('-' if extra==0 else f'{extra}')
        if extra>0: tot+=extra; nk+=1
        print(f'{k:38s}{pmul:8d}{padd:5d}{pfma:5d}{prn:5d} | '
              f'{sfmul:6d}{sfadd:6d}{sffma:6d}{shmma:6d}{flag:>12}')
    print(f'\nkernels where ptxas contracted: {nk}   extra FFMA total: {tot}')
