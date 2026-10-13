import pandas as pd, glob, re
rows=[]
for f in sorted(glob.glob('_cache/*.tsv')):
    d=pd.read_csv(f,sep='\t')
    d['file']=f.split('/')[-1]
    rows.append(d)
D=pd.concat(rows,ignore_index=True)
def cfgkey(n):
    if n.startswith('ENSEMBLE'):
        return None
    parts=[p for p in n.split('|') if not p.startswith('seed=')]
    return '|'.join(parts)
D['ens']=D.name.str.startswith('ENSEMBLE')
D['key']=D.name.map(cfgkey)
D['acct']=D.name.str.contains('fams=acct_ew(\||$)',regex=True) | D.name.str.contains(' on acct_ew')
print('total rows',len(D),'ens',D.ens.sum())
for f,g in D.groupby('file'):
    s=g[~g.ens]; e=g[g.ens]
    print(f, 'single rows',len(s),'distinct cfg',s.key.nunique(),'acct single cfg',s[s.acct].key.nunique(),'ens rows',len(e), 'ens acct',e.acct.sum())
s=D[(~D.ens)&D.acct]
print('distinct acct_ew single-run configs across files',s.key.nunique())
e=D[D.ens]
print('ENSEMBLE names'); 
for f,n in zip(e.file,e.name): print(f,'|',n)
