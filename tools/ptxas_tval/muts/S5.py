s=open('batch.py').read()
s=s.replace("same_if(P.stores[i][2], P.stores[i][0], S.stores[j][0])","same(P.stores[i][0], S.stores[j][0])")
s=s.replace("same_if(P.loads[i][1], P.loads[i][0], S.loads[j][0])","same(P.loads[i][0], S.loads[j][0])")
open('batch.py','w').write(s)
