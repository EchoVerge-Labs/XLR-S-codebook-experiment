#!/usr/bin/env python
"""Pre-GPU checkpoint: frame counts, duration balance, within-pair deltas, crop tradeoff."""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selection import PAIRS, META, ALL, CROP_PERCENTILE

RES = "results"
KERNELS, STRIDES = [10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2]


def n_frames(n_samples):
    """Exact wav2vec2 CNN output length for a given waveform length."""
    L = np.asarray(n_samples, dtype=np.int64)
    for k, s in zip(KERNELS, STRIDES):
        L = (L - k) // s + 1
    return np.maximum(L, 0)


def main():
    man = json.load(open(f"{RES}/fleurs_manifest.json"))
    NAME = {r["config"]: r["name"] for r in json.load(open(f"{RES}/fleurs_languages.json"))}
    dur = {c: np.array([r["duration"] for r in m["records"]]) for c, m in man.items()}
    frm = {c: n_frames((d * 16000).astype(np.int64)) for c, d in dur.items()}

    # ---------- 1. frame counts and durations
    print("=" * 100)
    print("  TABLE 1 - clips, duration and frame counts (500 clips per language)")
    print("=" * 100)
    print(f"{'language':<13}{'group':<10}{'clips':>6}{'mean s':>8}{'median s':>9}"
          f"{'min s':>7}{'max s':>7}{'hours':>7}{'frames':>10}{'vs min':>8}")
    rows = sorted(man, key=lambda c: -dur[c].mean())
    tot = {c: int(frm[c].sum()) for c in man}
    lo = min(tot.values())
    for c in rows:
        d, g = dur[c], META[c]["group"]
        print(f"{NAME[c]:<13}{g:<10}{len(d):>6}{d.mean():>8.2f}{np.median(d):>9.2f}"
              f"{d.min():>7.2f}{d.max():>7.2f}{d.sum()/3600:>7.2f}{tot[c]:>10,}"
              f"{tot[c]/lo:>7.2f}x")
    hi = max(tot.values())
    print("-" * 100)
    print(f"  frame-count spread: {lo:,} to {hi:,}  =  {100*(hi/lo-1):.1f}% divergence")
    for grp in ("SEEN", "UNSEEN"):
        cs = [c for c in man if META[c]["group"] == grp]
        dm = np.concatenate([dur[c] for c in cs])
        print(f"  {grp:<7} n={len(cs)}  mean clip {dm.mean():.2f}s  "
              f"language-mean {np.mean([dur[c].mean() for c in cs]):.2f}s")

    # ---------- 2. within-pair duration deltas (what actually threatens the paired test)
    print("\n" + "=" * 100)
    print("  TABLE 2 - within-pair duration deltas (SEEN minus UNSEEN)")
    print("=" * 100)
    print(f"{'pair':<26}{'family':<16}{'seen s':>8}{'unseen s':>10}{'delta s':>9}{'sign':>7}")
    deltas = []
    for s, u, fam, desc in PAIRS:
        if s not in dur or u not in dur:
            continue
        dl = dur[s].mean() - dur[u].mean()
        deltas.append(dl)
        nm = f"{NAME[s]}/{NAME[u]}"
        print(f"{nm:<26}{fam:<16}{dur[s].mean():>8.2f}{dur[u].mean():>10.2f}"
              f"{dl:>+9.2f}{'SEEN+' if dl > 0 else 'UNSEEN+':>7}")
    d = np.array(deltas)
    print("-" * 100)
    print(f"  mean signed delta {d.mean():+.3f}s   mean |delta| {np.abs(d).mean():.3f}s   "
          f"SEEN longer in {int((d>0).sum())}/{len(d)} pairs")
    from scipy import stats
    w = stats.wilcoxon(d) if len(d) >= 5 else None
    print(f"  symmetric around zero? Wilcoxon on deltas p={w.pvalue:.3f}" if w else "")
    print(f"  -> {'ROUGHLY SYMMETRIC - paired test not obviously duration-biased' if abs(d.mean()) < 0.5*np.abs(d).mean() else 'SYSTEMATICALLY SIGNED - duration may bias the paired test'}")

    # ---------- 3. crop-window tradeoff
    print("\n" + "=" * 100)
    print("  TABLE 3 - fixed-window tradeoff: differential clip loss by window length")
    print("=" * 100)
    pooled = np.concatenate([dur[c] for c in man])
    p25 = np.floor(np.percentile(pooled, CROP_PERCENTILE) / .5) * .5
    print(f"{'N (s)':>7}{'pctile':>8}{'kept %':>9}{'worst lang':>16}{'worst drop %':>14}"
          f"{'spread of drop %':>18}")
    best = None
    for N in [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        keep = {c: float((dur[c] >= N).mean()) for c in man}
        worst = min(keep, key=lambda c: keep[c])
        drops = np.array([100 * (1 - v) for v in keep.values()])
        pct = float((pooled < N).mean() * 100)
        mark = "   <- pre-registered P25" if abs(N - p25) < 1e-6 else ""
        print(f"{N:>7.1f}{pct:>7.0f}%{100*np.mean(list(keep.values())):>8.1f}%"
              f"{NAME[worst]:>16}{100*(1-keep[worst]):>13.1f}%"
              f"{drops.max()-drops.min():>17.1f}{mark}")
    print(f"\n  pre-registered window = P{CROP_PERCENTILE} of pooled = {p25:.1f}s")


if __name__ == "__main__":
    main()
