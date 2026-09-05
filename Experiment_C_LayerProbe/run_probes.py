#!/usr/bin/env python
"""Run every probe over the cached features and record results.

24 layers x 3 languages x 2 tasks x 3 seeds, plus the weighted-sum probe and the
data-scaling sweep. Nothing here touches audio or the model - all of it reads the
cache written by extract_features.py.
"""
import os, json, sys, time, argparse, subprocess, datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probes as P
from config import (LANGUAGES, N_LAYERS, SEEDS, SCALING_HOURS, CTC_PROBE,
                    SPEAKER_PROBE, WEIGHTED_SUM_PROBE, WEIGHTED_SPEAKER_PROBE,
                    CHECKPOINT_PRIMARY,
                    CTC_MAX_DRAWS, CTC_DIVERGENCE_OUTLIER_FACTOR,
                    CTC_PROBE_EXTRA_DRAWS)

ROOT = os.path.dirname(os.path.abspath(__file__))


def run_config(tag):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip() or "not-a-git-repo"
    except Exception:
        commit = "unavailable"
    return dict(tag=tag, checkpoint=CHECKPOINT_PRIMARY, n_layers=N_LAYERS, seeds=SEEDS,
                ctc_probe=CTC_PROBE, speaker_probe=SPEAKER_PROBE,
                weighted_sum_probe=WEIGHTED_SUM_PROBE, scaling_hours=SCALING_HOURS,
                git_commit=commit, timestamp=datetime.datetime.now().isoformat())




def ctc_cell(ctr, cte, sp, layer, seeds, hp=CTC_PROBE, max_hours=None):
    """Run one (language, layer) cell, drawing fresh inits until enough runs are healthy.

    A minority of initialisations collapse into a degenerate CTC solution. Collapsed runs
    are identified from TRAINING loss only (no test information): within a cell their
    final/first loss ratio is an order of magnitude worse than a healthy sibling's on
    identical data - Tamil L17 at 0.5 h gave 0.010-0.014 healthy against 0.207-0.288
    collapsed, with 4 of 9 inits surviving.

    The rule is relative rather than absolute because no global threshold separates the
    two: English's early layers are genuinely hard and sit at ratios of 0.15-0.36 with
    every seed agreeing, which an absolute cutoff would wrongly discard.

    Drawing until enough healthy runs accumulate also covers the case where EVERY initial
    seed collapses, which a purely within-set outlier test cannot see - there is then no
    healthy sibling to be an outlier against.
    """
    # Always draw a few MORE inits than are kept. Deciding after exactly `want` draws is
    # unsafe: if every one of them collapses they are mutually consistent and look
    # healthy relative to each other, which is precisely how the Tamil 0.5 h cell slipped
    # through. The extra draws probe for a better mode before anything is accepted.
    want = len(seeds)
    minimum = want + CTC_PROBE_EXTRA_DRAWS
    runs, drawn = [], 0
    while drawn < CTC_MAX_DRAWS:
        s = seeds[drawn] if drawn < len(seeds) else seeds[0] + 7919 * drawn
        runs.append(P.train_ctc(ctr, cte, sp["ctc_train"], sp["ctc_test"], sp["vocab"],
                                layer, s, hp=hp, max_hours=max_hours))
        drawn += 1
        if drawn < minimum:
            continue
        best = min(x["loss_ratio"] for x in runs)
        healthy = [x for x in runs
                   if x["loss_ratio"] <= CTC_DIVERGENCE_OUTLIER_FACTOR * best]
        if len(healthy) >= want:
            break
    best = min(x["loss_ratio"] for x in runs)
    healthy = [x for x in runs if x["loss_ratio"] <= CTC_DIVERGENCE_OUTLIER_FACTOR * best]
    healthy.sort(key=lambda x: x["loss_ratio"])
    keep = (healthy if len(healthy) >= want else
            sorted(runs, key=lambda x: x["loss_ratio"]))[:want]
    cer = [x["cer"] for x in keep]
    return dict(cer_mean=float(np.mean(cer)), cer_std=float(np.std(cer)), cer_seeds=cer,
                loss_ratios=[x["loss_ratio"] for x in keep],
                restarts=drawn - want, draws=drawn,
                enough_healthy=bool(len(healthy) >= want),
                n_train=keep[0]["n_train"], n_test=keep[0]["n_test"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="xlsr300m")
    ap.add_argument("--layers", type=int, default=N_LAYERS)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--tasks", nargs="*", default=["ctc", "speaker"])
    ap.add_argument("--stages", nargs="*", default=["layers", "weighted", "scaling"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_path = a.out or f"{ROOT}/results/probe_results_{a.tag}.json"

    data, caches, mmaps = {}, {}, {}
    for lang in LANGUAGES:
        sp = json.load(open(f"{ROOT}/data/{lang}_split.json", encoding="utf-8"))
        data[lang] = sp
        caches[lang] = (P.LayerCache(a.tag, lang, "train"), P.LayerCache(a.tag, lang, "test"))
        # the weighted-sum probe touches all 24 layers per batch, so it stays on memmaps
        mmaps[lang] = (P.LayerCache(a.tag, lang, "train", preload=False),
                       P.LayerCache(a.tag, lang, "test", preload=False))

    results = dict(config=run_config(a.tag), layers={}, weighted={}, scaling={})
    if os.path.exists(out_path):
        results.update(json.load(open(out_path)))
    t0 = time.time()

    # ---------------- per-layer probes
    if "layers" in a.stages:
        for lang in LANGUAGES:
            ctr, cte = caches[lang]
            sp = data[lang]
            results["layers"].setdefault(lang, {})
            for layer in range(a.layers):
                key = str(layer)
                cell = results["layers"][lang].get(key, {})
                if "ctc" in a.tasks and "ctc" not in cell:
                    cell["ctc"] = ctc_cell(ctr, cte, sp, layer, a.seeds)
                if "speaker" in a.tasks and "speaker" not in cell:
                    r = [P.train_speaker(ctr, cte, sp["speaker_train"], sp["speaker_test"],
                                         sp["speakers"], layer, s)["accuracy"]
                         for s in a.seeds]
                    cell["speaker"] = dict(acc_mean=float(np.mean(r)),
                                           acc_std=float(np.std(r)), acc_seeds=r,
                                           chance=1.0 / len(sp["speakers"]))
                results["layers"][lang][key] = cell
                msg = f"[{a.tag}] {lang} L{layer:02d}"
                if "ctc" in cell:
                    msg += (f"  CER {cell['ctc']['cer_mean']:.4f}"
                            f"+-{cell['ctc']['cer_std']:.4f}"
                            f" r{cell['ctc']['restarts']}")
                if "speaker" in cell:
                    msg += f"  SPK {cell['speaker']['acc_mean']:.4f}+-{cell['speaker']['acc_std']:.4f}"
                print(msg + f"   [{time.time()-t0:.0f}s]", flush=True)
                json.dump(results, open(out_path, "w"), indent=1)

    # ---------------- weighted-sum probe (SUPERB style)
    if "weighted" in a.stages:
        for lang in LANGUAGES:
            ctr, cte = mmaps[lang]; sp = data[lang]
            cell = results["weighted"].get(lang, {})
            if "ctc" not in cell and "ctc" in a.tasks:
                runs = [P.train_ctc(ctr, cte, sp["ctc_train"], sp["ctc_test"], sp["vocab"],
                                    None, s, hp=WEIGHTED_SUM_PROBE, weighted=True)
                        for s in a.seeds]
                cell["ctc"] = dict(cer_mean=float(np.mean([r["cer"] for r in runs])),
                                   cer_std=float(np.std([r["cer"] for r in runs])),
                                   weights_mean=np.mean([r["weights"] for r in runs], 0).tolist(),
                                   weights_seeds=[r["weights"] for r in runs])
                print(f"[{a.tag}] {lang} weighted CTC CER {cell['ctc']['cer_mean']:.4f}", flush=True)
            if "speaker" not in cell and "speaker" in a.tasks:
                runs = [P.train_speaker(ctr, cte, sp["speaker_train"], sp["speaker_test"],
                                        sp["speakers"], None, s, hp=WEIGHTED_SPEAKER_PROBE,
                                        weighted=True)
                        for s in a.seeds]
                cell["speaker"] = dict(acc_mean=float(np.mean([r["accuracy"] for r in runs])),
                                       acc_std=float(np.std([r["accuracy"] for r in runs])),
                                       weights_mean=np.mean([r["weights"] for r in runs], 0).tolist(),
                                       weights_seeds=[r["weights"] for r in runs])
                print(f"[{a.tag}] {lang} weighted SPK acc {cell['speaker']['acc_mean']:.4f}", flush=True)
            results["weighted"][lang] = cell
            json.dump(results, open(out_path, "w"), indent=1)

    # ---------------- data-scaling on each language's own best layer
    if "scaling" in a.stages:
        for lang in LANGUAGES:
            ctr, cte = caches[lang]; sp = data[lang]
            per = results["layers"].get(lang, {})
            if not per:
                continue
            best = min((int(k) for k in per if "ctc" in per[k]),
                       key=lambda k: per[str(k)]["ctc"]["cer_mean"])
            cell = results["scaling"].get(lang, {"best_layer": best})
            for h in SCALING_HOURS:
                k = str(h)
                if k in cell:
                    continue
                cell[k] = ctc_cell(ctr, cte, sp, best, a.seeds, max_hours=h)
                print(f"[{a.tag}] {lang} scaling L{best} {h}h  CER "
                      f"{cell[k]['cer_mean']:.4f}+-{cell[k]['cer_std']:.4f}  "
                      f"(n={cell[k]['n_train']}, restarts={cell[k]['restarts']})", flush=True)
                results["scaling"][lang] = cell
                json.dump(results, open(out_path, "w"), indent=1)

    json.dump(results, open(out_path, "w"), indent=1)
    print(f"\ndone in {time.time()-t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
