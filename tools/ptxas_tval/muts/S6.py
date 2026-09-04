s=open('smut/smem_roundtrip.sass').read()
s=s.replace('                   BAR.SYNC.DEFER_BLOCKING 0x0 ;','                   NOP ;'); open('smut/smem_roundtrip.sass','w').write(s)
