#!/usr/bin/env python
"""Pre-sweep audit: clip counts, frame counts, and whether SEEN/UNSEEN are balanced
on clip duration. Duration matters because longer clips give the transformer more
context at masked positions - Experiment A shows rho(duration, InfoNCE) as strong as -0.60 -
so a duration imbalance between groups could manufacture or mask a group effect."""
import os, sys, json, glob
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selection import META, PAIRS, ALL

RES = "results"
NAME = {r["config"]: r["name"] for r in json.load(open(f"{RES}/fleurs_languages.json"))}
rows = []
for d in sorted(glob.glob("data/audio/*/.done")):
    m = json.load(open(d))
    dur = np.array([r["duration"] for r in m["records"]])
    c = m["config"]
    rows.append(dict(cfg=c, name=NAME[c], group=META[c]["group"],
                     prox=META[c].get("proximity", "-"), n=m["n_staged"],
                     avail=m["n_available"], fail=m["n_failed"],
                     hours=dur.sum() / 3600, mean_s=dur.mean(), med_s=np.median(dur),
                     frames=int(dur.sum() * 50)))
rows.sort(key=lambda r: (r["group"], -r["frames"]))
print(f"{'language':<12}{'group':<11}{'prox':<9}{'clips':>6}{'avail':>7}{'hours':>7}"
      f"{'mean s':>8}{'frames':>10}{'fail':>5}")
print("-" * 76)
for r in rows:
    print(f"{r['name']:<12}{r['group']:<11}{r['prox']:<9}{r['n']:>6}{r['avail']:>7}"
          f"{r['hours']:>7.2f}{r['mean_s']:>8.2f}{r['frames']:>10,}{r['fail']:>5}")
fr = np.array([r["frames"] for r in rows])
print("-" * 76)
print(f"frames: min {fr.min():,}  max {fr.max():,}  "
      f"spread {100*(fr.max()/fr.min()-1):.0f}%  (flag threshold 30%)")

S = [r for r in rows if r["group"] == "SEEN"]
U = [r for r in rows if r["group"] == "UNSEEN"]
if S and U:
    s, u = np.array([r["mean_s"] for r in S]), np.array([r["mean_s"] for r in U])
    t, p = stats.ttest_ind(s, u, equal_var=False)
    print(f"\nDURATION BALANCE  SEEN mean {s.mean():.2f}s (n={len(s)})   "
          f"UNSEEN mean {u.mean():.2f}s (n={len(u)})   diff {u.mean()-s.mean():+.2f}s   "
          f"Welch p={p:.3f}")
    got = {r['cfg']: r for r in rows}
    d = [got[uu]['mean_s'] - got[ss]['mean_s'] for ss, uu, _, _ in PAIRS
         if ss in got and uu in got]
    if d:
        tw, pw = stats.wilcoxon(d) if len(d) > 2 else (np.nan, np.nan)
        print(f"within-pair duration difference: mean {np.mean(d):+.2f}s, "
              f"Wilcoxon p={pw:.3f}  (n={len(d)} pairs)")
        print("  -> if this is significant, the duration-matched arm (--max-seconds) "
              "is required, not optional.")
