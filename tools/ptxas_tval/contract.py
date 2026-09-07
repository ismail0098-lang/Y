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
    """Kernels where ptxas actually FUSES a multiply into an add.

    MEASURED as `FMA(plain) - FMA(.rn)`: forbidding the fusion cannot remove a
    fused instruction the PTX asked for (the rewrite touches only mul/add/sub),
    so the difference IS the number of fusions.  Directly the quantity, not a
    proxy for it -- and this number has now been wrong three times, each time
    from an indirect reading:

      * a HARDCODED 9 in `fpgate.py`, wrong in both directions;
      * a DERIVED 16, `FFMA(sass) - fma(ptx) > 0`, which cannot tell a
        contraction from an UNROLLING (one `fma.rn.f32` in a loop becomes N
        FFMA) and counted `FFMA` ptxas synthesised for other reasons;
      * a MEASURED-BUT-PERMISSIVE 9, "the SASS moves under `.rn`", which is
        true of any effect the modifier has.  The four `gemm_f16_swiglu_*`
        move by **FSEL 4 -> 8 and IMAD 202 -> 206 with FFMA 0 -> 0** -- a
        scheduling difference and no fusion at all.  The half-precision
        explanation published for them was a hypothesis asserted as fact:
        `HFMA2` appears in neither build.

    Counting `HFMA2` alongside `FFMA` keeps a genuine half-precision fusion in
    scope; nothing in this corpus has one."""
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
        def fused(src, tag):
            pp = os.path.join(work, f'{k}_{tag}.ptx'); open(pp,'w').write(src)
            cu = os.path.join(work, f'{k}_{tag}.cubin')
            if subprocess.run(['ptxas', f'-arch={am.group(1)}', '-o', cu, pp],
                              capture_output=True).returncode: return None
            d = subprocess.run(['nvdisasm','-c',cu],capture_output=True,text=True).stdout
            return len(re.findall(r'\b(?:FFMA|HFMA2)\b', d))
        a, b = fused(P,'as'), fused(rn,'rn')
        if a is not None and b is not None and a > b: out.append(k)
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
