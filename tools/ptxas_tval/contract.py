"""Where does ptxas ACTUALLY contract?

A kernel only needs the .rn repair if its SASS contains an FFMA that its PTX
did not ask for.  Counting float ops in the PTX (which is what "29 float
kernels" counts) is not that question.
"""
import glob,os,re
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
