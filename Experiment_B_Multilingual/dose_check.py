#!/usr/bin/env python
"""Resolve the dose-response discrepancy: which subset, and how stable is it?"""
import json, sys, os, itertools
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse_b as A
from selection import META
RES="results"
A.NAME.update({r["config"]:r["name"] for r in json.load(open(f"{RES}/fleurs_languages.json"))})
fl={r["config"]:r for r in json.load(open(f"{RES}/fleurs_languages.json"))}
df=pd.read_parquet(f"{RES}/per_file_metrics_full.parquet")
r={}; lm,_=A.report_checkpoint(df,"xlsr300m",r)

rows=[(c, fl[c]["xlsr300_hours"], lm.loc[c,"infonce"], META[c]["group"], A.NAME[c])
      for c in lm.index if fl[c]["xlsr300_hours"]]
noh=[A.NAME[c] for c in lm.index if not fl[c]["xlsr300_hours"]]
print(f"languages with published XLS-R hours: {len(rows)}/19   (no hours: {noh})\n")

def sp(sub, label):
    h=np.array([r[1] for r in sub]); y=np.array([r[2] for r in sub])
    rho,p=stats.spearmanr(h,y)
    return dict(label=label,n=len(sub),rho=float(rho),p=float(p))

print("="*88); print("  1. THE TWO NUMBERS, SIDE BY SIDE"); print("="*88)
allr=[r for r in rows]
seen=[r for r in rows if r[3]=="SEEN"]
unseen=[r for r in rows if r[3]=="UNSEEN"]
ref=[r for r in rows if r[3]=="REFERENCE"]
for s,lab in [(allr,"ALL languages in XLS-R (n=18)"),
              (seen,"XLSR-53 SEEN subset only (n=9)"),
              (unseen,"XLSR-53 UNSEEN subset only (n=8)")]:
    v=sp(s,lab); print(f"  {v['label']:<34} n={v['n']:<3} rho={v['rho']:+.3f}  p={v['p']:.4f}")

print("\n"+"="*88); print("  2. IS THE n=9 RESTRICTION MEANINGFUL FOR AN XLS-R ANALYSIS?"); print("="*88)
print("  The dose (hours) and the outcome (InfoNCE) are both XLS-R quantities.")
print("  The n=9 subset is defined by XLSR-53 membership - a different checkpoint.")
print(f"  If that restriction were meaningful, SEEN and UNSEEN subsets should behave alike.")
vs, vu = sp(seen,"s"), sp(unseen,"u")
print(f"    SEEN   subset: rho={vs['rho']:+.3f} (p={vs['p']:.4f})")
print(f"    UNSEEN subset: rho={vu['rho']:+.3f} (p={vu['p']:.4f})")
print(f"    -> the two halves {'AGREE' if vs['rho']*vu['rho']>0 and abs(vs['rho']-vu['rho'])<0.4 else 'DISAGREE'}; "
      f"difference in rho = {abs(vs['rho']-vu['rho']):.3f}")

print("\n"+"="*88); print("  3. LEVERAGE CHECK ON THE n=9 CORRELATION"); print("="*88)
h=np.array([r[1] for r in seen]); y=np.array([r[2] for r in seen]); nm=[r[4] for r in seen]
base=stats.spearmanr(h,y)
print(f"  full n=9: rho={base.statistic:+.3f} p={base.pvalue:.4f}")
print(f"  hours: {sorted(zip(nm,h.astype(int)), key=lambda t:-t[1])}")
print(f"\n  jackknife - drop one language at a time:")
jk=[]
for i in range(len(seen)):
    k=[j for j in range(len(seen)) if j!=i]
    rr=stats.spearmanr(h[k],y[k])
    jk.append((nm[i],rr.statistic,rr.pvalue))
for n_,rr,pp in sorted(jk,key=lambda t:t[1]):
    print(f"     drop {n_:<12} rho={rr:+.3f}  p={pp:.4f}")
rs=[t[1] for t in jk]
print(f"  -> jackknife rho range: [{min(rs):+.3f}, {max(rs):+.3f}]  (base {base.statistic:+.3f})")
print(f"  -> p<0.05 in {sum(1 for t in jk if t[2]<0.05)}/{len(jk)} jackknife replicates")

hi=[i for i,n_ in enumerate(nm) if n_ in ("Polish","Estonian")]
k=[i for i in range(len(seen)) if i not in hi]
rr=stats.spearmanr(h[k],y[k])
print(f"\n  drop BOTH high-leverage points (Polish {int(h[hi[0]] if nm[hi[0]]=='Polish' else h[hi[1]])}h, "
      f"Estonian): n={len(k)} rho={rr.statistic:+.3f} p={rr.pvalue:.4f}")

print("\n  same leverage check on n=18:")
h2=np.array([r[1] for r in allr]); y2=np.array([r[2] for r in allr]); nm2=[r[4] for r in allr]
b2=stats.spearmanr(h2,y2)
jk2=[stats.spearmanr(np.delete(h2,i),np.delete(y2,i)).statistic for i in range(len(allr))]
print(f"     base rho={b2.statistic:+.3f} p={b2.pvalue:.4f}; jackknife range [{min(jk2):+.3f}, {max(jk2):+.3f}]")
big=[i for i,v in enumerate(h2) if v>5000]
k2=[i for i in range(len(allr)) if i not in big]
r2=stats.spearmanr(h2[k2],y2[k2])
print(f"     drop all {len(big)} languages above 5000h ({[nm2[i] for i in big]}): "
      f"n={len(k2)} rho={r2.statistic:+.3f} p={r2.pvalue:.4f}")
json.dump(dict(all18=sp(allr,"all"),seen9=sp(seen,"seen"),unseen8=sp(unseen,"unseen"),
               jackknife_n9={n_: [rr,pp] for n_,rr,pp in jk},
               n9_drop_polish_estonian=dict(n=len(k),rho=float(rr.statistic),p=float(rr.pvalue)),
               n18_jackknife_range=[float(min(jk2)),float(max(jk2))]),
          open(f"{RES}/dose_response_check.json","w"),indent=1)
