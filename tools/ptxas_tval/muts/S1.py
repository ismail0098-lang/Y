# barrier modelled as a NO-OP, kernel unchanged.  EXPECTED to validate: the
# kernel is correct, so this alone is not a defect.  See S8/S8b.
s=open('smem.py').read(); s=s.replace("        return self.h(k)(arr)","        return arr"); open('smem.py','w').write(s)
