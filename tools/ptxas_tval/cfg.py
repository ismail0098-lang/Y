import re,sys,glob,os
def analyse(f):
    L=open(f).read()
    lab={m.group(1):int(m.group(2),16) for m in re.finditer(r'\.L_(\w+):\s*\n\s*/\*([0-9a-f]+)\*/',L)}
    ins=[(int(m.group(1),16), m.group(2).strip()) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+(.*?);',L)]
    # the trailing `.L_x_0: BRA .L_x_0` self-loop nvcc emits after EXIT is dead code
    trap=set()
    for a,t in ins:
        m=re.search(r'BRA\s+`\(\.L_(\w+)\)',t)
        if m and lab.get(m.group(1))==a: trap.add(a)
    back=fwd=0
    for a,t in ins:
        m=re.search(r'BRA\s+`\(\.L_(\w+)\)',t)
        if not m or a in trap: continue
        tg=lab.get(m.group(1))
        if tg is None: continue
        (back:=back+1) if tg<=a else (fwd:=fwd+1)
        if tg>a: fwd=fwd  # noop, counted above
    fwd=sum(1 for a,t in ins if (m:=re.search(r'BRA\s+`\(\.L_(\w+)\)',t)) and a not in trap and (tg:=lab.get(m.group(1))) is not None and tg>a)
    memb=sum(1 for _,t in ins if re.search(r'\bBAR\.|MEMBAR|ARRIVES|LDS|STS|LDSM|ATOM|RED\b',t))
    recon=sum(1 for _,t in ins if re.search(r'\bBSSY\b|\bBSYNC\b',t))
    return len(ins), fwd, back, recon, memb
print(f"{'kernel':34s}{'insns':>6}{'fwdBRA':>7}{'loop':>5}{'recon':>6}{'shmem/atomic':>13}")
rows=[]
for f in sorted(glob.glob('corpus/*.sass')):
    k=os.path.basename(f)[:-5]
    n,fwd,back,rec,mem=analyse(f)
    if fwd+back>0 or rec>0: rows.append((n,k,fwd,back,rec,mem))
for n,k,fwd,back,rec,mem in sorted(rows)[:14]:
    print(f"{k:34s}{n:6d}{fwd:7d}{back:5d}{rec:6d}{mem:13d}")
