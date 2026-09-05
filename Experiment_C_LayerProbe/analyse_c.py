#!/usr/bin/env python
"""Experiment C analysis: curve shape, pre-registered verdict, figures.

The verdict criteria come from config.CRITERIA and were fixed before any curve existed.
Nothing here re-derives them.
"""
import os, json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LANGUAGES, N_LAYERS, CRITERIA, SCALING_HOURS

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = f"{ROOT}/results"
COL = {"sinhala": "#D1495B", "tamil": "#EDAE49", "english": "#00798C"}
NAME = {"sinhala": "Sinhala", "tamil": "Tamil", "english": "English"}


def load(tag):
    return json.load(open(f"{RES}/probe_results_{tag}.json"))


def curves(r, lang, task, key):
    per = r["layers"][lang]
    xs = sorted(int(k) for k in per if task in per[k])
    m = np.array([per[str(k)][task][key + "_mean"] for k in xs])
    s = np.array([per[str(k)][task][key + "_std"] for k in xs])
    return np.array(xs), m, s


def shape_stats(xs, cer):
    """Normalised-shape descriptors. Absolute CER is never compared across languages."""
    norm = cer / cer.min()
    best = int(xs[int(np.argmin(cer))])
    peak = float(norm.max() / norm.min())
    w = 1.0 / norm                    # weight layers by how usable they are
    centroid = float((w * xs).sum() / w.sum())
    return dict(best_layer=best, peakedness=peak, centroid=centroid,
                normalised=norm.tolist())


def verdict(stats):
    ref = stats["english"]
    out = {}
    for lang in LANGUAGES:
        if lang == "english":
            continue
        s = stats[lang]
        db = abs(s["best_layer"] - ref["best_layer"])
        dc = abs(s["centroid"] - ref["centroid"])
        dp = abs(s["peakedness"] - ref["peakedness"]) / ref["peakedness"]
        met = dict(best_layer_shift=db >= CRITERIA["best_layer_shift_material"],
                   centroid_shift=dc >= CRITERIA["centroid_shift_material"],
                   peakedness_diff=dp >= (CRITERIA["peakedness_ratio_material"] - 1))
        out[lang] = dict(delta_best_layer=db, delta_centroid=round(dc, 3),
                         rel_peakedness_diff=round(dp, 3), criteria_met=met,
                         verdict="CONCENTRATED" if (met["best_layer_shift"] or
                                                    met["centroid_shift"]) else "UNIFORM")
    return out


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "xlsr300m"
    r = load(tag)
    floor = None
    if os.path.exists(f"{RES}/probe_results_floor.json"):
        floor = json.load(open(f"{RES}/probe_results_floor.json"))
    ds = json.load(open(f"{RES}/data_statistics.json"))

    stats, ctc, spk = {}, {}, {}
    for lang in LANGUAGES:
        xs, m, s = curves(r, lang, "ctc", "cer")
        ctc[lang] = (xs, m, s)
        stats[lang] = shape_stats(xs, m)
        xs2, m2, s2 = curves(r, lang, "speaker", "acc")
        spk[lang] = (xs2, m2, s2)
        stats[lang]["speaker_best_layer"] = int(xs2[int(np.argmax(m2))])
        stats[lang]["speaker_best_acc"] = float(m2.max())
        stats[lang]["char_vocab"] = ds[lang]["char_vocab_size"]

    v = verdict(stats)
    prof_ok = {l: stats[l]["speaker_best_layer"] < stats[l]["best_layer"] for l in LANGUAGES}

    print("=" * 100)
    print("  SHAPE STATISTICS (normalised per-layer CER; absolute CER not comparable across scripts)")
    print("=" * 100)
    print(f"{'language':<10}{'char vocab':>11}{'best CTC layer':>16}{'centroid':>10}"
          f"{'peakedness':>12}{'best CER':>10}{'spk best layer':>16}{'spk acc':>9}")
    for l in LANGUAGES:
        s = stats[l]
        print(f"{NAME[l]:<10}{s['char_vocab']:>11}{s['best_layer']:>16}{s['centroid']:>10.2f}"
              f"{s['peakedness']:>12.3f}{ctc[l][1].min():>10.4f}"
              f"{s['speaker_best_layer']:>16}{s['speaker_best_acc']:>9.3f}")
    print("\n  (best CER shown for completeness only - it is script-dependent and is never "
          "compared across languages)")

    print("\n" + "=" * 100)
    print("  PRE-REGISTERED VERDICT  (criteria fixed in config.py before any curve existed)")
    print("=" * 100)
    for l, d in v.items():
        print(f"  {NAME[l]} vs English: dBest={d['delta_best_layer']} "
              f"(threshold {CRITERIA['best_layer_shift_material']}), "
              f"dCentroid={d['delta_centroid']} (threshold {CRITERIA['centroid_shift_material']}), "
              f"relPeak={d['rel_peakedness_diff']} "
              f"(threshold {CRITERIA['peakedness_ratio_material']-1})")
        print(f"     -> {d['verdict']}")

    print("\n" + "=" * 100)
    print("  INSTRUMENT VALIDATION")
    print("=" * 100)
    print("  expected profile (speaker-ID peaks earlier than CTC):")
    for l in LANGUAGES:
        print(f"    {NAME[l]:<9} speaker L{stats[l]['speaker_best_layer']:<3} vs "
              f"CTC L{stats[l]['best_layer']:<3}  -> {'PASS' if prof_ok[l] else 'FAIL'}")
    if floor:
        print("\n  floor control (randomly initialised model of identical architecture):")
        for l in LANGUAGES:
            if l not in floor.get("layers", {}):
                continue
            fx, fm, _ = curves(floor, l, "ctc", "cer")
            sx, sm, _ = curves(floor, l, "speaker", "acc")
            print(f"    {NAME[l]:<9} CER range {fm.min():.4f}-{fm.max():.4f} "
                  f"(spread {fm.max()-fm.min():.4f}) | spk acc {sm.min():.3f}-{sm.max():.3f} "
                  f"(chance {1/40:.3f})")

    out = dict(shape_stats=stats, verdict=v, expected_profile_pass=prof_ok,
               criteria=CRITERIA)
    json.dump(out, open(f"{RES}/analysis_{tag}.json", "w"), indent=1, default=float)

    # ---------------------------------------------------------------- figures
    # FIG 1: normalised CER vs layer (+ floor control overlay)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for l in LANGUAGES:
        xs, m, s = ctc[l]
        n = m / m.min(); ns = s / m.min()
        ax.plot(xs, n, "-o", color=COL[l], ms=4.5, lw=2,
                label=f"{NAME[l]} (best L{stats[l]['best_layer']}, vocab {stats[l]['char_vocab']})")
        ax.fill_between(xs, n - ns, n + ns, color=COL[l], alpha=.18)
        ax.axvline(stats[l]["best_layer"], color=COL[l], ls=":", lw=1.2, alpha=.7)
    if floor:
        for l in LANGUAGES:
            if l not in floor.get("layers", {}):
                continue
            fx, fm, _ = curves(floor, l, "ctc", "cer")
            ax.plot(fx, fm / fm.min(), "--", color="#888", lw=1.3, alpha=.8,
                    label="floor control (random init)" if l == LANGUAGES[0] else None)
    ax.set_xlabel("transformer layer"); ax.set_ylabel("CER / best-layer CER (per language)")
    ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=9.5)
    ax.set_title("Normalised per-layer CER - curve SHAPE only", fontsize=12.5, weight="bold")
    fig.text(.5, .015, "Absolute CER is NOT comparable across Sinhala, Tamil and English: different "
             "scripts, orthographic depth and character vocabularies (106 / 49 / 34).\n"
             "Each curve is divided by its own language's best-layer CER, so only shape - where the "
             "minimum sits and how peaked it is - is being compared. Bands are +/-1 SD over 3 seeds.",
             ha="center", fontsize=8.4, style="italic")
    fig.tight_layout(rect=[0, .075, 1, 1])
    fig.savefig(f"{RES}/fig1_normalised_cer.png", dpi=170); plt.close(fig)

    # FIG 2: speaker accuracy vs layer
    fig, ax = plt.subplots(figsize=(11, 6))
    for l in LANGUAGES:
        xs, m, s = spk[l]
        ax.plot(xs, m, "-o", color=COL[l], ms=4.5, lw=2,
                label=f"{NAME[l]} (best L{stats[l]['speaker_best_layer']})")
        ax.fill_between(xs, m - s, m + s, color=COL[l], alpha=.18)
    ax.axhline(1 / 40, color="#444", ls=":", lw=1.4)
    ax.text(0.3, 1 / 40 + .012, "chance (40 speakers)", fontsize=8.6, color="#444")
    if floor:
        for l in LANGUAGES:
            if l not in floor.get("layers", {}):
                continue
            fx, fm, _ = curves(floor, l, "speaker", "acc")
            ax.plot(fx, fm, "--", color="#888", lw=1.3, alpha=.8,
                    label="floor control (random init)" if l == LANGUAGES[0] else None)
    ax.set_xlabel("transformer layer"); ax.set_ylabel("speaker-ID accuracy")
    ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=9.5)
    ax.set_title("Speaker-ID accuracy by layer", fontsize=12.5, weight="bold")
    fig.text(.5, .015, "Unlike CER, this IS directly comparable across languages: the task is "
             "script-free, all three have 40 speaker classes, and chance is 1/40 for each.",
             ha="center", fontsize=8.6, style="italic")
    fig.tight_layout(rect=[0, .055, 1, 1])
    fig.savefig(f"{RES}/fig2_speaker_accuracy.png", dpi=170); plt.close(fig)

    # FIG 3: learned weighted-sum weights
    if r.get("weighted"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=False)
        for ax, task, lab in [(axes[0], "ctc", "CTC (character)"),
                              (axes[1], "speaker", "Speaker ID")]:
            for l in LANGUAGES:
                c = r["weighted"].get(l, {}).get(task)
                if not c:
                    continue
                w = np.array(c["weights_mean"])
                cen = (w * np.arange(len(w))).sum() / w.sum()
                ax.plot(range(len(w)), w, "-o", color=COL[l], ms=4, lw=1.8,
                        label=f"{NAME[l]} (centroid {cen:.1f})")
                ax.axvline(cen, color=COL[l], ls=":", lw=1.1, alpha=.7)
            ax.set_xlabel("layer"); ax.set_ylabel("learned softmax weight")
            ax.set_title(lab, fontsize=11, weight="bold"); ax.grid(alpha=.25)
            ax.legend(frameon=False, fontsize=9)
        fig.suptitle("Weighted-sum probe: learned layer-importance profile",
                     fontsize=12.5, weight="bold")
        fig.tight_layout(rect=[0, 0, 1, .93])
        fig.savefig(f"{RES}/fig3_weighted_sum.png", dpi=170); plt.close(fig)

    # FIG 4: data-scaling
    if r.get("scaling"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
        for l in LANGUAGES:
            c = r["scaling"].get(l)
            if not c:
                continue
            hs = [h for h in SCALING_HOURS if str(h) in c]
            ys = np.array([c[str(h)]["cer_mean"] for h in hs])
            es = np.array([c[str(h)]["cer_std"] for h in hs])
            axes[0].errorbar(hs, ys, yerr=es, fmt="-o", color=COL[l], capsize=3,
                             label=f"{NAME[l]} (L{c['best_layer']})")
            axes[1].errorbar(hs, ys / ys[-1], yerr=es / ys[-1], fmt="-o", color=COL[l],
                             capsize=3, label=NAME[l])
        axes[0].set_ylabel("CER (script-dependent - do NOT compare across languages)")
        axes[1].set_ylabel("CER relative to that language's 3 h value")
        for ax in axes:
            ax.set_xlabel("labelled training hours"); ax.grid(alpha=.25)
            ax.legend(frameon=False, fontsize=9.5)
        axes[0].set_title("Absolute (within-language only)", fontsize=11, weight="bold")
        axes[1].set_title("Normalised - shape comparison", fontsize=11, weight="bold")
        fig.suptitle("Data-scaling on each language's own best layer: efficiency vs ceiling",
                     fontsize=12.5, weight="bold")
        fig.tight_layout(rect=[0, 0, 1, .92])
        fig.savefig(f"{RES}/fig4_data_scaling.png", dpi=170); plt.close(fig)

    print(f"\nfigures + analysis -> {RES}/")


if __name__ == "__main__":
    main()
