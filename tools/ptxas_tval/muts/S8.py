# THE ISOLATING MUTATION FOR THE BARRIER FUNCTION.  Move the PTX bar.sync AFTER
# the shared load: the arrays entering the barrier are unchanged, so no snapshot
# can see it and only H_k can.
s=open('smut/smem_roundtrip.ptx').read()
bar='    bar.sync 0;\n'; ld='    ld.shared.v4.u32 {%r32, %r33, %r34, %r35}, [%rd7];\n'
assert bar in s and ld in s
open('smut/smem_roundtrip.ptx','w').write(s.replace(bar,'').replace(ld, ld+bar))
