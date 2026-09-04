"""Does a lower -O avoid the unroll?

If it does, a real kernel's loop can be validated at that level today, and the
-O0..-O3 differential (bit-identical output on this kernel, already measured)
is what relates it to the shipped build.  Weaker than validating -O3, and worth
knowing before building unroll matching.
"""
import re,subprocess,os,tempfile,sys
def build(ptx, O, tag):
    d=tempfile.mkdtemp(prefix=f'ol_{tag}_{O}_{os.getpid()}_')   # per-call tag
    c=os.path.join(d,'k.cubin')
    r=subprocess.run(['ptxas','-arch=sm_89',f'-O{O}',ptx,'-o',c],capture_output=True,text=True)
    if r.returncode: return None
    t=subprocess.run(['nvdisasm','-c',c],capture_output=True,text=True).stdout
    lab={m.group(1):int(m.group(2),16) for m in re.finditer(r'\.L_(\w+):\s*\n\s*/\*([0-9a-f]+)\*/',t)}
    ins=[(int(m.group(1),16),m.group(2).strip()) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+(.*?);',t)]
    trap={a for a,x in ins if (m:=re.search(r'BRA\s+`\(\.L_(\w+)\)',x)) and lab.get(m.group(1))==a}
    bk=[(lab[m.group(1)],a) for a,x in ins if (m:=re.search(r'BRA\s+`\(\.L_(\w+)\)',x)) and a not in trap
        and m.group(1) in lab and lab[m.group(1)]<=a]
    if not bk: return len(ins), 0, 0, 0
    h,e=bk[0]; body=[x for a,x in ins if h<=a<=e]
    return len(ins), len(bk), len(body), sum(1 for x in body if re.search(r'\b(LDG|STG)\b',x))

print(f'{"kernel":26s}{"-O":>4}{"insns":>7}{"backedges":>11}{"body":>6}{"ld/st in body":>15}')
for name,ptx,ptx_body_ldst in [('loop/sum.ptx','loop/sum.ptx',1),
                               ('loop/count.ptx','loop/count.ptx',0),
                               ('corpus/exact_pv.ptx','corpus/exact_pv.ptx',2),
                               ('corpus/naive_gemm_f32.ptx','corpus/naive_gemm_f32.ptx',2)]:
    for O in (0,1,2,3):
        r=build(ptx,O,os.path.basename(ptx).replace('.','_'))
        if r is None: print(f'{name:26s}{O:4d}   ptxas refused'); continue
        n,bk,bl,ls=r
        u = f'  unroll x{ls/ptx_body_ldst:.0f}' if ptx_body_ldst and ls else ('  (no mem in body)' if bk else '')
        print(f'{name:26s}{O:4d}{n:7d}{bk:11d}{bl:6d}{ls:15d}{u}')
    print()
