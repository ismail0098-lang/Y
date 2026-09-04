s=open('smem.py').read(); s=s.replace("    return {d[0]: 0 for d in decls}","    return {d[0]: 64 for d in decls}"); open('smem.py','w').write(s)
