#!/usr/bin/env python
"""Publication figures for the ICASSP paper: single-column (3.39 in), vector PDF,
Times-like fonts, readable in greyscale (line style + marker, not colour alone)."""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = os.path.expanduser("~/EchoVerge-LABS")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "lines.linewidth": 1.1})
COLW = 3.39
STY = {"sinhala": dict(c="#b2182b", ls="-",  m="o", lab="Sinhala"),
       "tamil":   dict(c="#d6851b", ls="--", m="s", lab="Tamil"),
       "english": dict(c="#2166ac", ls=":",  m="^", lab="English")}

# ---------------------------------------------------------------- Fig. 1: Exp B forest
st = json.load(open(f"{LAB}/Experiment_B_Multilingual/results/final_statistics.json"))
arms = [("full", "Full clips"), ("crop6", "6.0 s crop"), ("crop9", "9.0 s crop")]
fig, ax = plt.subplots(figsize=(COLW, 1.55))
ax.axvspan(-0.590, 0.590, color="0.93", zorder=0)
ax.axvspan(-0.289, 0.289, color="0.82", zorder=0)
ax.axvline(0, color="k", lw=0.7, zorder=1)
for i, (k, lab) in enumerate(arms):
    t = st["arms"][k]["tost_0.289"]; y = 2 - i
    ax.plot([t["ci95_lo"], t["ci95_hi"]], [y, y], color="k", lw=0.8, zorder=2)
    ax.plot([t["ci90_lo"], t["ci90_hi"]], [y, y], color="k", lw=3.2, zorder=2,
            solid_capstyle="butt")
    ax.plot(t["mean"], y, "o", ms=4.5, mfc="white", mec="k", mew=0.9, zorder=3)
    ax.text(0.70, y, f"TOST $p$={t['p_tost']:.2g}", va="center", fontsize=6.8)
ax.set_yticks([2, 1, 0]); ax.set_yticklabels([a[1] for a in arms])
ax.set_xlim(-0.66, 1.02); ax.set_ylim(-0.6, 2.6)
ax.set_xticks([-0.59, -0.289, 0, 0.289, 0.59])
ax.set_xticklabels(["$-$0.59", "$-$0.29", "0", "0.29", "0.59"])
ax.set_xlabel("Paired InfoNCE difference (unseen $-$ seen)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.savefig(f"{OUT}/fig_equivalence.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 2: Exp C layers
R = f"{LAB}/Experiment_C_LayerProbe/results"
pr = json.load(open(f"{R}/probe_results_xlsr300m.json"))
fl = json.load(open(f"{R}/probe_results_floor.json"))
L = np.arange(24)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(COLW, 3.2), sharex=True)
for lang, s in STY.items():
    m = np.array([pr["layers"][lang][str(l)]["ctc"]["cer_mean"] for l in L])
    sd = np.array([pr["layers"][lang][str(l)]["ctc"]["cer_std"] for l in L])
    n, ns = m / m.min(), sd / m.min()
    a1.fill_between(L, n - ns, n + ns, color=s["c"], alpha=0.18, lw=0)
    a1.plot(L, n, ls=s["ls"], marker=s["m"], ms=2.6, color=s["c"], mew=0,
            label=f"{s['lab']} (best L{int(L[m.argmin()])})")
    f = np.array([fl["layers"][lang][str(l)]["ctc"]["cer_mean"] for l in L])
    a1.plot(L, f / f.min(), color="0.55", lw=0.6, ls=(0, (1, 1)))
    acc = np.array([pr["layers"][lang][str(l)]["speaker"]["acc_mean"] for l in L])
    asd = np.array([pr["layers"][lang][str(l)]["speaker"]["acc_std"] for l in L])
    a2.fill_between(L, acc - asd, acc + asd, color=s["c"], alpha=0.18, lw=0)
    a2.plot(L, acc, ls=s["ls"], marker=s["m"], ms=2.6, color=s["c"], mew=0,
            label=f"{s['lab']} (best L{int(L[acc.argmax()])})")
    fa = np.array([fl["layers"][lang][str(l)]["speaker"]["acc_mean"] for l in L])
    a2.plot(L, fa, color="0.55", lw=0.6, ls=(0, (1, 1)))
a1.plot([], [], color="0.55", lw=0.6, ls=(0, (1, 1)), label="random-init floor")
a1.set_ylabel("CER / best-layer CER"); a1.set_ylim(0.9, 4.3)
a1.legend(loc="upper center", ncol=2, frameon=False, handlelength=2.2, columnspacing=1.0)
a1.set_title("(a) Character CTC error, normalised per language", loc="left", pad=2)
a2.axhline(1 / 40, color="k", lw=0.5, ls="-")
a2.text(0.3, 0.045, "chance", fontsize=6.5)
a2.set_ylabel("Speaker-ID accuracy"); a2.set_xlabel("Transformer layer")
a2.set_ylim(0, 1.02); a2.set_xlim(-0.5, 23.5); a2.set_xticks(range(0, 24, 3))
# line styles are identified by the legend in panel (a)
a2.set_title("(b) Speaker identification (40 speakers)", loc="left", pad=2)
for a in (a1, a2):
    for s in ("top", "right"): a.spines[s].set_visible(False)
fig.subplots_adjust(hspace=0.30)
fig.savefig(f"{OUT}/fig_layers.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 3: collapse rate
a = json.load(open(f"{R}/collapse_rate.json")); b = json.load(open(f"{R}/collapse_rate_cv_tamil.json"))
LL = [0, 4, 8, 12, 15, 17, 20, 22]
OFF = {"OpenSLR Sinhala": -0.33, "OpenSLR Tamil": -0.11, "CV Tamil": 0.11, "CV English": 0.33}
arms = [("OpenSLR Sinhala", a["sinhala"], "#b2182b", "-", "o"),
        ("OpenSLR Tamil", a["tamil"], "#d6851b", "--", "s"),
        ("CV Tamil", b["tamil_cv"], "#7f3b08", "-.", "D"),
        ("CV English", a["english"], "#2166ac", ":", "^")]
fig, ax = plt.subplots(figsize=(COLW, 1.55))
for lab, d, c, ls, mk in arms:
    y = [100 * d[str(l)]["rate"] for l in LL]
    tot = sum(d[str(l)]["n_collapsed"] for l in LL)
    ax.plot([l + OFF[lab] for l in LL], y, ls=ls, marker=mk, ms=3, color=c, label=f"{lab} ({tot}/72)")
ax.set_xlabel("Transformer layer"); ax.set_ylabel("Collapsed inits (%)")
ax.set_xticks(LL); ax.set_ylim(-2, 38)
ax.legend(ncol=2, frameon=False, fontsize=6.6, loc="upper left", handlelength=2.2,
          columnspacing=0.8)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.savefig(f"{OUT}/fig_collapse.pdf"); plt.close(fig)
print("wrote:", sorted(os.listdir(OUT)))
