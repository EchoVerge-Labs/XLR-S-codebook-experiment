#!/usr/bin/env python
import json, sys, os
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse_b as A
from selection import PAIRS, META
from final_stats import tost_paired, MARGIN_PRIMARY, MARGIN_SECONDARY, STAGE_A_CONTROLS

RES = "results"
ARMS = [("_full", "Full clips", "#1B3A54"), ("_crop6.0", "6.0 s crop", "#00798C"),
        ("_crop9.0", "9.0 s crop", "#7A9E9F")]
C = {"SEEN": "#00798C", "UNSEEN": "#D1495B", "REFERENCE": "#4A4A4A"}
PC = {"CLOSE": "#E8A33D", "DISTANT": "#8E2C48"}


def lmeans(tag, ck="xlsr53"):
    df = pd.read_parquet(f"{RES}/per_file_metrics{tag}.parquet")
    r = {}
    return A.report_checkpoint(df, ck, r)[0], df


def main():
    A.NAME.update({r["config"]: r["name"]
                   for r in json.load(open(f"{RES}/fleurs_languages.json"))})
    man = json.load(open(f"{RES}/fleurs_manifest.json"))
    odur = {c: np.mean([r["duration"] for r in m["records"]]) for c, m in man.items()}
    L = {tag: lmeans(tag) for tag, _, _ in ARMS}

    # ---------------- FIG 1 (LEAD): cross-arm forest plot
    fig, ax = plt.subplots(figsize=(12.4, 6.0))
    ax.axvspan(-MARGIN_SECONDARY, MARGIN_SECONDARY, color="#00798C", alpha=.06, zorder=0)
    ax.axvspan(-MARGIN_PRIMARY, MARGIN_PRIMARY, color="#00798C", alpha=.13, zorder=0)
    ax.axvline(0, color="#222", lw=1.6, zorder=2)
    for x in (MARGIN_PRIMARY, -MARGIN_PRIMARY):
        ax.axvline(x, color="#00798C", lw=1.5, ls="--", zorder=2)
    ys = [2, 1, 0]
    for (tag, lab, col), y in zip(ARMS, ys):
        lm = L[tag][0]
        d = np.array([lm.loc[u, "infonce"] - lm.loc[s, "infonce"] for s, u, _, _ in PAIRS])
        t = tost_paired(d, MARGIN_PRIMARY)
        ax.plot([t["ci95_lo"], t["ci95_hi"]], [y, y], color=col, lw=2.4, solid_capstyle="round", zorder=3)
        ax.plot([t["ci90_lo"], t["ci90_hi"]], [y, y], color=col, lw=7, alpha=.45,
                solid_capstyle="butt", zorder=3)
        ax.plot(t["mean"], y, "o", color=col, ms=11, zorder=5)
        dz = t["mean"] / t["sd"]
        ax.text(-0.72, y + .30, f"diff {t['mean']:+.3f}    dz {dz:+.2f}    TOST p = {t['p_tost']:.4f}",
                fontsize=9.6, ha="left", va="center", color=col, weight="600")
    # Stage A calibration anchors, staggered so the two near 0.29/0.31 do not collide
    for i, (lab, v) in enumerate(list(STAGE_A_CONTROLS.items())[:3]):
        yv = -0.62 - 0.30 * (i % 2)
        ax.plot(v, yv, "v", color="#8E2C48", ms=8, zorder=4)
        ax.text(v + .015, yv, f" {lab} ({v:+.2f})", fontsize=8.2, ha="left", va="center",
                color="#8E2C48")
    ax.text(-0.72, -0.62, "Stage A positive controls:", fontsize=8.6, style="italic",
            color="#8E2C48", va="center")
    ax.set_yticks(ys); ax.set_yticklabels([a[1] for a in ARMS], fontsize=11.5)
    ax.set_ylim(-1.30, 2.75); ax.set_xlim(-0.78, 0.95)
    ax.set_xlabel("paired InfoNCE difference,  UNSEEN - SEEN   (positive = unseen fits worse)",
                  fontsize=10.5)
    ax.grid(alpha=.22, axis="x")
    ax.set_title("Seen/unseen effect is smaller than the mildest corruption known to harm the codebook",
                 fontsize=13, weight="bold", pad=26)
    ax.text(.5, 1.035, "9 family-matched pairs, XLSR-53   |   thick bar = 90% CI (TOST), "
            "thin = 95% CI   |   shaded = equivalence bounds (0.289 / 0.590)",
            transform=ax.transAxes, ha="center", fontsize=9.6, color="#444")
    fig.text(.5, .022, "Equivalence margins are calibrated on Stage A positive controls. Those are unpaired "
             "absolute differences across corpora on XLS-R 0.3B, while this is a paired within-corpus\n"
             "difference on XLSR-53 - a calibration anchor, not a formally matched equivalence margin.",
             ha="center", fontsize=8.5, style="italic")
    fig.tight_layout(rect=[0, .075, 1, .97])
    fig.savefig(f"{RES}/fig1_forest_cross_arm.png", dpi=170); plt.close(fig)

    # ---------------- FIG 2: per-language, relative to English
    lm = L["_full"][0]
    fig, ax = plt.subplots(figsize=(11, 6.6))
    base = lm.loc["en_us", "infonce"]
    d = lm.drop(index=["en_us"]).copy()
    d["rel"] = d["infonce"] / base
    d = d.sort_values("rel")
    ax.barh(range(len(d)), d["rel"], color=[C[g] for g in d.group], height=.72)
    ax.axvline(1.0, color="#222", lw=1.8)
    for i, (v, c) in enumerate(zip(d["rel"], d.index)):
        ax.text(v + .004, i, f"{v:.3f}  ({odur[c]:.1f}s)", va="center", fontsize=8.2)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d["name"], fontsize=10)
    ax.set_xlabel(f"InfoNCE relative to English (English mean clip = {odur['en_us']:.2f} s)  "
                  "- lower = better codebook fit")
    ax.set_xlim(.9, max(d["rel"]) * 1.08); ax.grid(alpha=.22, axis="x")
    h = [plt.Rectangle((0, 0), 1, 1, color=C[k]) for k in ("SEEN", "UNSEEN")]
    ax.legend(h, ["SEEN by XLSR-53", "UNSEEN by XLSR-53"], frameon=False, fontsize=10, loc="lower right")
    ax.set_title("Per-language InfoNCE, full-clip arm (XLSR-53)", fontsize=12.5, weight="bold")
    fig.text(.5, .018,
             "CAUTION - not a language-quality ranking. Clip duration lowers InfoNCE "
             "(median rho = -0.36 across these 19 languages) and English is the second-shortest "
             "corpus at 10.00 s, below every language except Polish. Bars above the reference line are\n"
             "therefore expected from duration alone. Mean clip duration is printed beside each bar. "
             "The seen/unseen contrast is unaffected, since English is a common divisor.",
             ha="center", fontsize=8.3, style="italic")
    fig.tight_layout(rect=[0, .075, 1, 1])
    fig.savefig(f"{RES}/fig2_per_language_infonce.png", dpi=170); plt.close(fig)

    # ---------------- FIG 3: pair InfoNCE delta vs ORIGINAL pair duration delta
    fig, ax = plt.subplots(figsize=(9.6, 6))
    dd = np.array([odur[s] - odur[u] for s, u, _, _ in PAIRS])
    for tag, lab, col in ARMS:
        lmx = L[tag][0]
        ni = np.array([lmx.loc[u, "infonce"] - lmx.loc[s, "infonce"] for s, u, _, _ in PAIRS])
        rho, p = stats.spearmanr(dd, ni)
        ax.scatter(dd, ni, s=62, color=col, label=f"{lab}   rho={rho:+.2f} (p={p:.2f})", zorder=3)
        b, a = np.polyfit(dd, ni, 1)
        xx = np.linspace(dd.min(), dd.max(), 30)
        ax.plot(xx, a + b * xx, "--", color=col, lw=1.6, zorder=2)
    ax.axhline(0, color="#888", lw=1); ax.axvline(0, color="#888", lw=1)
    for i, (s, u, _, _) in enumerate(PAIRS):
        lmx = L["_full"][0]
        ax.annotate(f"{lmx.loc[s,'name'][:3]}/{lmx.loc[u,'name'][:3]}",
                    (dd[i], lmx.loc[u, "infonce"] - lmx.loc[s, "infonce"]),
                    fontsize=7.4, xytext=(5, 4), textcoords="offset points", color="#444")
    ax.set_xlabel("pair duration delta, SEEN - UNSEEN (s)")
    ax.set_ylabel("pair InfoNCE delta, UNSEEN - SEEN")
    ax.legend(frameon=False, fontsize=9.5); ax.grid(alpha=.22)
    ax.set_title("Duration tracking in the pair differences collapses under matched context\n"
                 "the full-arm +0.55 is the sign the mechanism predicts; it does not survive cropping",
                 fontsize=11.5, weight="bold")
    fig.tight_layout(); fig.savefig(f"{RES}/fig3_pair_duration_tracking.png", dpi=170); plt.close(fig)

    # ---------------- FIG 4: the 9 pairs, interleave made visible
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.4), sharey=True)
    for ax, (tag, lab, col) in zip(axes, ARMS):
        lmx = L[tag][0]
        order = sorted(PAIRS, key=lambda P: lmx.loc[P[1], "infonce"] - lmx.loc[P[0], "infonce"])
        for i, (s, u, fam, _) in enumerate(order):
            a, b = lmx.loc[s, "infonce"], lmx.loc[u, "infonce"]
            ax.plot([a, b], [i, i], "-", color="#BBB", lw=1.6, zorder=1)
            ax.plot(a, i, "o", color=C["SEEN"], ms=8, zorder=3)
            ax.plot(b, i, "o", color=C["UNSEEN"], ms=8, zorder=3)
            ax.annotate("", xy=(b, i), xytext=(a, i),
                        arrowprops=dict(arrowstyle="->", color="#777", lw=1.1))
        ax.set_yticks(range(9))
        ax.set_yticklabels([f"{lmx.loc[s,'name']} / {lmx.loc[u,'name']}" for s, u, _, _ in order],
                           fontsize=8.8)
        nw = sum(1 for s, u, _, _ in PAIRS if lmx.loc[u, "infonce"] > lmx.loc[s, "infonce"])
        ax.set_title(f"{lab}\n{nw} unseen worse / {9-nw} unseen better", fontsize=10.5, weight="bold")
        ax.set_xlabel("InfoNCE"); ax.grid(alpha=.22, axis="x")
    h = [plt.Line2D([], [], marker="o", ls="", color=C[k], label=v)
         for k, v in [("SEEN", "SEEN language"), ("UNSEEN", "UNSEEN language")]]
    axes[0].legend(handles=h, frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("The 9 family-matched pairs: direction flips pair by pair in every arm",
                 fontsize=12.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(f"{RES}/fig4_pairs_interleave.png", dpi=170); plt.close(fig)

    # ---------------- FIG 5: dose-response, all XLS-R languages, with leverage shown
    fl = {r["config"]: r for r in json.load(open(f"{RES}/fleurs_languages.json"))}
    df30 = pd.read_parquet(f"{RES}/per_file_metrics_full.parquet")
    r30 = {}
    lm30, _ = A.report_checkpoint(df30, "xlsr300m", r30)
    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    langs = [c for c in lm30.index if fl[c]["xlsr300_hours"]]
    xs = np.array([fl[c]["xlsr300_hours"] for c in langs], float)
    ys = np.array([lm30.loc[c, "infonce"] for c in langs])
    cols = [C[lm30.loc[c, "group"]] for c in langs]
    ax.axvspan(5000, xs.max() * 1.9, color="#8E2C48", alpha=.07, zorder=0)
    ax.text(7000, ys.max() - .01, "high-leverage region\n(6 languages > 5000 h)",
            fontsize=8.4, color="#8E2C48", va="top")
    ax.scatter(xs, ys, s=78, color=cols, zorder=3, edgecolor="white", linewidth=.7)
    for c, x, y in zip(langs, xs, ys):
        ax.annotate(lm30.loc[c, "name"], (x, y), fontsize=8.3, xytext=(6, 4),
                    textcoords="offset points")
    lx = np.log10(xs); b, a = np.polyfit(lx, ys, 1)
    xx = np.linspace(lx.min(), lx.max(), 40)
    ax.plot(10 ** xx, a + b * xx, "--", color="#333", lw=1.8, zorder=2,
            label="fit, all 18")
    low = xs <= 5000
    b2, a2 = np.polyfit(np.log10(xs[low]), ys[low], 1)
    xx2 = np.linspace(np.log10(xs[low].min()), np.log10(5000), 30)
    ax.plot(10 ** xx2, a2 + b2 * xx2, ":", color="#8E2C48", lw=2.1, zorder=2,
            label="fit, low-resource only")
    r_all, p_all = stats.spearmanr(xs, ys)
    r_low, p_low = stats.spearmanr(xs[low], ys[low])
    ax.set_xscale("log"); ax.grid(alpha=.22)
    ax.set_xlabel("XLS-R pre-training hours for that language (log scale)")
    ax.set_ylabel("InfoNCE (lower = better fit)")
    h = [plt.Line2D([], [], marker="o", ls="", color=C[k], label=v) for k, v in
         [("SEEN", "XLSR-53 SEEN"), ("UNSEEN", "XLSR-53 UNSEEN"), ("REFERENCE", "English")]]
    ax.legend(handles=h + ax.get_legend_handles_labels()[0], frameon=False, fontsize=9,
              loc="lower right")
    ax.set_title(f"Dose-response is carried entirely by the high-resource tail\n"
                 f"all 18 XLS-R languages: rho={r_all:+.2f} (p={p_all:.3f})   |   "
                 f"excluding the 6 above 5000 h: rho={r_low:+.2f} (p={p_low:.3f})",
                 fontsize=11.5, weight="bold")
    fig.text(.5, .015, "Colour marks XLSR-53 membership, which is irrelevant to this analysis - both "
             "dose and outcome are XLS-R quantities. Xhosa is absent from XLS-R and has no published hours.",
             ha="center", fontsize=8.4, style="italic")
    fig.tight_layout(rect=[0, .045, 1, 1])
    fig.savefig(f"{RES}/fig5_dose_response.png", dpi=170); plt.close(fig)

    # ---------------- FIG 6: proximity + negative control
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for ax, ck, ttl in [(axes[0], "xlsr53", "XLSR-53 - the split is real here"),
                        (axes[1], "xlsr300m", "XLS-R 0.3B - negative control (all but Xhosa seen)")]:
        dfx = pd.read_parquet(f"{RES}/per_file_metrics_full.parquet")
        rr = {}
        lmx, _ = A.report_checkpoint(dfx, ck, rr)
        seenv = lmx.loc[[c for c in lmx.index if lmx.loc[c, "group"] == "SEEN"], "infonce"].values
        pos = 0
        for lab, sel, col in [("SEEN", [c for c in lmx.index if lmx.loc[c, "group"] == "SEEN"], C["SEEN"]),
                              ("UNSEEN\nCLOSE", [c for c in lmx.index if META[c].get("proximity") == "CLOSE"], PC["CLOSE"]),
                              ("UNSEEN\nDISTANT", [c for c in lmx.index if META[c].get("proximity") == "DISTANT"], PC["DISTANT"])]:
            v = lmx.loc[sel, "infonce"].values
            ax.scatter(np.random.default_rng(1).normal(pos, .06, len(v)), v, s=58,
                       color=col, zorder=4, edgecolor="white", linewidth=.8)
            ax.hlines(v.mean(), pos - .3, pos + .3, color=col, lw=3, zorder=5)
            if lab.startswith("UNSEEN"):
                for c, y in zip(sel, v):
                    ax.annotate(lmx.loc[c, "name"], (pos + .12, y), fontsize=7.6, va="center")
            pos += 1
        ax.axhspan(seenv.mean() - seenv.std(ddof=1), seenv.mean() + seenv.std(ddof=1),
                   color=C["SEEN"], alpha=.10)
        ax.set_xticks(range(3)); ax.set_xticklabels(["SEEN\n(n=9)", "UNSEEN\nCLOSE (n=5)",
                                                     "UNSEEN\nDISTANT (n=4)"], fontsize=9.5)
        ax.set_ylabel("InfoNCE"); ax.grid(alpha=.22, axis="y")
        ax.set_title(ttl, fontsize=11, weight="bold")
    fig.suptitle("Membership and proximity subgroups (shaded band = SEEN mean +/- 1 SD)",
                 fontsize=12.5, weight="bold")
    fig.text(.5, .015, "UNSEEN-DISTANT (n=4) is descriptive only - exploratory and underpowered "
             "(MDE d >= 1.85); no significance test is reported for it.",
             ha="center", fontsize=8.6, style="italic")
    fig.tight_layout(rect=[0, .05, 1, .93])
    fig.savefig(f"{RES}/fig6_proximity_control.png", dpi=170); plt.close(fig)
    print("wrote fig1..fig6 to", RES)


if __name__ == "__main__":
    main()
