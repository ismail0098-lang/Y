# the SASS store moved ACROSS the barrier, REAL barrier model
s=open('smut/smem_roundtrip.sass').read()
sts='                   STS.128 [R2.X16], R4 ;'; bar='                   BAR.SYNC.DEFER_BLOCKING 0x0 ;'
assert sts in s and bar in s
s=s.replace(sts,'@@STS@@').replace(bar,sts).replace('@@STS@@',bar)
open('smut/smem_roundtrip.sass','w').write(s)
