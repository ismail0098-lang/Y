"""Did ptxas UNROLL the loop?

A simulation relation at the loop header only works if the two loops run the
same number of times.  If ptxas unrolled by k, the PTX body has to be composed
k times, and the peeled prologue and the remainder cascade each need their own
case -- a different and much larger piece of machinery.  So measure the ratio
before building either.

Proxy: count global memory operations inside the PTX loop body against those
inside the SASS back-edge region.  ptxas cannot invent or delete a load.
"""
import re,glob,os

def ptx_loop(k):
    t=open(f'corpus/{k}.ptx').read()
    lines=[l.strip() for l in t.splitlines()]
    lab={}
    for i,l in enumerate(lines):
        m=re.fullmatch(r'\$?([A-Za-z_$][\w$]*):',l)
        if m: lab[m.group(1).lstrip('$')]=i
    back=[]
    for i,l in enumerate(lines):
        m=re.match(r'(?:@!?%p\d+\s+)?bra(?:\.uni)?\s+\$?([\w$]+);',l)
        if m:
            t2=lab.get(m.group(1).lstrip('$'))
            if t2 is not None and t2<i: back.append((t2,i))
    if not back: return None
    h,e=back[0]
    body=lines[h:e+1]
    return len(back), sum(1 for l in body if re.match(r'(?:@!?%p\d+\s+)?(ld|st)\.global',l))

def sass_loop(k):
    t=open(f'corpus/{k}.sass').read()
    lab={m.group(1):int(m.group(2),16) for m in re.finditer(r'\.L_(\w+):\s*\n\s*/\*([0-9a-f]+)\*/',t)}
    ins=[(int(m.group(1),16),m.group(2).strip())
         for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+(.*?);',t)]
    trap={a for a,x in ins if (m:=re.search(r'BRA\s+`\(\.L_(\w+)\)',x)) and lab.get(m.group(1))==a}
    back=[]
    for a,x in ins:
        m=re.search(r'BRA\s+`\(\.L_(\w+)\)',x)
        if not m or a in trap: continue
        tg=lab.get(m.group(1))
        if tg is not None and tg<=a: back.append((tg,a))
    if not back: return None
    h,e=back[0]
    body=[x for a,x in ins if h<=a<=e]
    return len(back), sum(1 for x in body if re.search(r'\b(LDG|STG)\b',x))

print(f'{"kernel":40s}{"ptx bk":>7}{"ldst":>6} | {"sass bk":>8}{"ldst":>6}{"ratio":>7}   verdict')
one_to_one=[]
for f in sorted(glob.glob('corpus/*.ptx')):
    k=os.path.basename(f)[:-4]
    if not os.path.exists(f'corpus/{k}.sass'): continue
    try: p=ptx_loop(k); s=sass_loop(k)
    except Exception: continue
    if not p or not s: continue
    r = s[1]/p[1] if p[1] else float('nan')
    v = 'UNROLLED x%.0f' % r if r>1.5 else ('1:1  <- tractable' if abs(r-1)<0.01 else f'ratio {r:.2f}')
    if abs(r-1)<0.01 and p[0]==1 and s[0]==1: one_to_one.append(k)
    print(f'{k:40s}{p[0]:7d}{p[1]:6d} | {s[0]:8d}{s[1]:6d}{r:7.2f}   {v}')
print(f'\n1:1 single-loop kernels (a header relation is enough): {len(one_to_one)}')
for k in one_to_one: print('   ', k)
