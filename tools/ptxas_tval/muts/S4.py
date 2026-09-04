s=open('smut/smem_roundtrip.sass').read()
s=s.replace('                   STS.128 [R2.X16], R4 ;','                   NOP ;'); open('smut/smem_roundtrip.sass','w').write(s)
