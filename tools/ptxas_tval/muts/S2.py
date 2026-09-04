s=open('sassexec.py').read()
s=s.replace("        if m3.group(2):\n            base = base * bv(int(m3.group(2)))","        if False:\n            base = base * bv(int(m3.group(2)))")
open('sassexec.py','w').write(s)
