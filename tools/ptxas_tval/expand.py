"""How many SASS instructions does ptxas emit for ONE PTX float op?

The earlier split into CONTRACTION vs MACRO-OP was too coarse: it put
`sin.approx.f32` (one MUFU) in the same class as `rcp.rn.f32` (96 instructions
and an out-of-line CALL).  Measure the expansion per opcode instead of guessing
from the mnemonic.  Baseline is the same kernel with the op replaced by a move,
so the count is the op's own cost and not the surrounding address arithmetic.
"""
import re,os,subprocess,tempfile
ARCH='sm_89'
TMPL="""//
.version 7.8
.target sm_89
.address_size 64
.visible .entry k(.param .u64 A, .param .u64 B, .param .u64 O)
{{
    .reg .b32 %r<8>;  .reg .f32 %f<8>;  .reg .f64 %fd<8>;  .reg .b64 %rd<16>;
    ld.param.u64 %rd0, [A]; ld.param.u64 %rd1, [B]; ld.param.u64 %rd3, [O];
    mov.u32 %r0, %tid.x; cvt.u64.u32 %rd4, %r0; shl.b64 %rd5, %rd4, 2;
    add.u64 %rd6, %rd0, %rd5; add.u64 %rd7, %rd1, %rd5; add.u64 %rd9, %rd3, %rd5;
    ld.global.f32 %f1, [%rd6]; ld.global.f32 %f2, [%rd7];
{body}    st.global.f32 [%rd9], %f4;
    ret;
}}
"""
def count(body, tag):
    d=tempfile.mkdtemp(prefix=f'exp_{tag}_{os.getpid()}_')      # per-call tag
    p=os.path.join(d,'k.ptx'); c=os.path.join(d,'k.cubin')
    open(p,'w').write(TMPL.format(body=body))
    r=subprocess.run(['ptxas','-arch='+ARCH,'-O3',p,'-o',c],capture_output=True,text=True)
    if r.returncode: return None,None
    s=subprocess.run(['nvdisasm','-c',c],capture_output=True,text=True).stdout
    n=len(re.findall(r'/\*[0-9a-f]{4,}\*/\s+\S',s))
    ops=sorted(set(re.findall(r'/\*[0-9a-f]{4,}\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9._]*)',s)))
    return n,ops

base,_ = count("    mov.f32 %f4, %f1;\n", 'base')
CASES=[
 ('mul.f32',        "    mul.f32 %f4, %f1, %f2;\n"),
 ('sin.approx.f32', "    sin.approx.f32 %f4, %f1;\n"),
 ('cos.approx.f32', "    cos.approx.f32 %f4, %f1;\n"),
 ('ex2.approx.f32', "    ex2.approx.f32 %f4, %f1;\n"),
 ('lg2.approx.f32', "    lg2.approx.f32 %f4, %f1;\n"),
 ('rcp.approx.f32', "    rcp.approx.f32 %f4, %f1;\n"),
 ('rsqrt.approx.f32',"   rsqrt.approx.f32 %f4, %f1;\n"),
 ('div.approx.f32', "    div.approx.f32 %f4, %f1, %f2;\n"),
 ('sqrt.approx.f32',"    sqrt.approx.f32 %f4, %f1;\n"),
 ('rcp.rn.f32',     "    rcp.rn.f32 %f4, %f1;\n"),
 ('sqrt.rn.f32',    "    sqrt.rn.f32 %f4, %f1;\n"),
 ('div.rn.f32',     "    div.rn.f32 %f4, %f1, %f2;\n"),
 ('div.rn.f64',     "    cvt.f64.f32 %fd1,%f1; cvt.f64.f32 %fd2,%f2;\n"
                    "    div.rn.f64 %fd4, %fd1, %fd2;\n    cvt.rn.f32.f64 %f4,%fd4;\n"),
 ('cvt.rn.f32.s32', "    cvt.rzi.s32.f32 %r1,%f1;\n    cvt.rn.f32.s32 %f4, %r1;\n"),
 ('cvt.f32.f16',    "    cvt.rn.f16.f32 %r1,%f1;\n    cvt.f32.f16 %f4, %r1;\n"),
]
print(f'baseline (mov only): {base} instructions\n')
print(f'{"ptx op":20s}{"insns":>7}{"delta":>7}   class      SASS opcodes it becomes')
for name,body in CASES:
    n,ops = count(body, re.sub(r'\W','_',name))
    if n is None: print(f'{name:20s}   ptxas refused'); continue
    d=n-base
    cls = 'TRANSLIT' if d<=2 else ('SMALL' if d<=8 else 'EXPANDED')
    interesting=[o for o in ops if not re.match(r'^(NOP|BRA|EXIT|MOV|S2R|ULDC|IMAD|IADD3|LDG|STG|SHF|LEA|CS2R)',o)]
    print(f'{name:20s}{n:7d}{d:+7d}   {cls:9s}  {" ".join(interesting[:9])}')
