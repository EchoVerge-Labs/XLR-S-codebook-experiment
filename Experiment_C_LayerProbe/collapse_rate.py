#!/usr/bin/env python
"""Measure the CTC collapse rate directly, as its own data series.

The divergence rule discards collapsed inits and keeps healthy ones. That is only safe
if the amount of cleaning is even across the comparison being made. The restart counts
from the main run suggest it is not: all 10 events fell in Sinhala and Tamil, at layers
15-17/20, and none in English. Restart counts are a lower bound though, since the rule
cannot see a cell where every init collapses.

So the rate is measured directly: draw a fixed number of independent inits per cell and
count how many collapse, judged from training loss only.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probes as P
from config import LANGUAGES
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
LAYERS = [0, 4, 8, 12, 15, 17, 20, 22]
N_DRAWS = 9
OUT = f"{ROOT}/results/collapse_rate.json"

_ap = argparse.ArgumentParser()
_ap.add_argument("--langs", nargs="*", default=None)
_ap.add_argument("--out", default=None)
_A, _ = _ap.parse_known_args()
if _A.out: OUT = f"{ROOT}/results/{_A.out}"


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    t0 = time.time()
    for lang in (_A.langs or LANGUAGES):
        sp = json.load(open(f"{ROOT}/data/{lang}_split.json", encoding="utf-8"))
        ctr = P.LayerCache("xlsr300m", lang, "train")
        cte = P.LayerCache("xlsr300m", lang, "test")
        res.setdefault(lang, {})
        for L in LAYERS:
            if str(L) in res[lang]:
                continue
            runs = [P.train_ctc(ctr, cte, sp["ctc_train"], sp["ctc_test"], sp["vocab"],
                                L, 10_000 + 137 * s) for s in range(N_DRAWS)]
            ratios = [x["loss_ratio"] for x in runs]
            cers = [x["cer"] for x in runs]
            best = min(ratios)
            collapsed = [q > 3.0 * best for q in ratios]   # same rule as the sweep
            res[lang][str(L)] = dict(
                n_draws=N_DRAWS, n_collapsed=int(sum(collapsed)),
                rate=float(np.mean(collapsed)), loss_ratios=ratios, cers=cers,
                healthy_cer_mean=float(np.mean([c for c, b in zip(cers, collapsed) if not b])),
                collapsed_cer_mean=(float(np.mean([c for c, b in zip(cers, collapsed) if b]))
                                    if any(collapsed) else None))
            print(f"  {lang:<8} L{L:<3} collapsed {sum(collapsed)}/{N_DRAWS} "
                  f"({100*np.mean(collapsed):4.0f}%)  healthy CER "
                  f"{res[lang][str(L)]['healthy_cer_mean']:.4f}   [{time.time()-t0:.0f}s]",
                  flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
