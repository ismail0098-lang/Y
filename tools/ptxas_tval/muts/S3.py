s=open('smem.py').read()
s=s.replace("    return LShR(_win(byte_addr), 2)","    r = LShR(byte_addr, 2)\n    return r if r.size()==W else Extract(W-1,0,r)")
open('smem.py','w').write(s)
