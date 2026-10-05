# CONTROL: reorder the two alignment-obligation lists.  Semantically neutral.
s=open('smem.py').read(); a='ptx.align_obs + sass.align_obs'
assert s.count(a)==1; s=s.replace(a,'sass.align_obs + ptx.align_obs'); open('smem.py','w').write(s)
