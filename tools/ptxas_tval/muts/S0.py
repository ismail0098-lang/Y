# CONTROL: reorder the two alignment-obligation lists.  Semantically neutral.
s=open('smemval.py').read(); s=s.replace('P.align_obs + S.align_obs','S.align_obs + P.align_obs'); open('smemval.py','w').write(s)
