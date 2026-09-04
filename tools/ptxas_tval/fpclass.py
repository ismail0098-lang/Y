"""Split the float problem into its two classes.

  CONTRACTION  ptxas fuses mul.f32 + add.f32 into FFMA.  A PERMITTED freedom
               (mul.rn forbids it), so the repair is one token per operation
               and validation afterwards is bit-exact.

  MACRO-OP     a single PTX instruction with an exact IEEE meaning that ptxas
               implements as a MULTI-INSTRUCTION refinement sequence
               (div/rcp/sqrt -> MUFU.RCP + FFMA Newton-Raphson; ex2/sin/cos ->
               MUFU).  There is no PTX spelling that makes ptxas transliterate
               it.  Bit-exact validation here is an IEEE theorem about the
               sequence, not a term match -- and it is exactly what the
               uninterpreted-function abstraction is built to hide.

These need different machinery, so the corpus has to be split by them and not
by "uses floats".
"""
import re,glob,os
MACRO = re.compile(r'^\s*(?:@\S+\s+)?(div|rcp|sqrt|rsqrt|ex2|lg2|sin|cos|tanh)\.[a-z0-9.]*f(32|64)\b', re.M)
CONTRACTIBLE = re.compile(r'^\s*(?:@\S+\s+)?(mul|add|sub)\.f32\b', re.M)
rows=[]
for pf in sorted(glob.glob('corpus/*.ptx')):
    k=os.path.basename(pf)[:-4]
    sf=f'corpus/{k}.sass'
    if not os.path.exists(sf): continue
    P=open(pf).read(); S=open(sf).read()
    macro=[m.group(0).strip().split()[-1] for m in MACRO.finditer(P)]
    contr=len(CONTRACTIBLE.findall(P))
    ffma=len(re.findall(r'\bFFMA\b',S))
    pfma=len(re.findall(r'^\s*(?:@\S+\s+)?fma\.\w+\.f32\b',P,re.M))
    mufu=len(re.findall(r'\bMUFU\b',S))
    if not (macro or contr or ffma or mufu): continue
    rows.append((k,contr,len(macro),ffma-pfma,mufu,sorted(set(macro))))

print(f'{"kernel":38s}{"contractible":>13}{"macro-op":>10}{"extraFFMA":>11}{"MUFU":>6}  class')
nc=nm=nb=0
for k,contr,nmac,ex,mufu,ms in rows:
    if nmac and contr: c='BOTH'; nb+=1
    elif nmac:         c='MACRO-OP'; nm+=1
    elif contr:        c='CONTRACTION'; nc+=1
    else:              c='(float, neither)'
    print(f'{k:38s}{contr:13d}{nmac:10d}{ex:+11d}{mufu:6d}  {c}'+(f'  {",".join(ms)}' if ms else ''))
print(f'\nCONTRACTION only: {nc}    MACRO-OP only: {nm}    BOTH: {nb}')
print('\n=> option (a) ".rn everywhere" is sufficient for the CONTRACTION-only set,')
print('   and does nothing at all for the other two.')
