"""Sensitivity ladder: place the language differences against knowingly OOD audio,
so the size of the language effect can be read against the metric's actual range."""
import os, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
RES = os.environ.get("XLSR_RESULTS", "results")
MODEL_SHORT = os.environ.get("MODEL_SHORT", "XLS-R 0.3B (wav2vec2-xls-r-300m)")
BASE = "English_CV"
LANG = ["Sinhala_YT", "Tamil_YT", "Sinhala_read", "Tamil_read"]
CTL = ["Ctl_SpeedPitch", "Ctl_Babble", "Ctl_Reversed", "Ctl_Tones", "Ctl_WhiteNoise"]
NAME = {"Sinhala_YT": "Sinhala (YouTube)", "Tamil_YT": "Tamil (YouTube)",
        "Sinhala_read": "Sinhala (read)", "Tamil_read": "Tamil (read)",
        "Ctl_SpeedPitch": "1.45x speed/pitch", "Ctl_Babble": "6-way babble",
        "Ctl_Reversed": "reversed speech", "Ctl_Tones": "sine tones",
        "Ctl_WhiteNoise": "white noise", BASE: "English (Common Voice)"}

def frame_stats(c):
    z = np.load(f"{RES}/frames_{c}.npz"); fid, rms = z["file_id"], z["rms"]
    p90 = np.array([np.quantile(rms[fid == f], .90) for f in range(fid.max() + 1)], dtype=np.float32)
    m = rms > .10 * p90[fid]; cd = z["codes"][m]
    perp = []
    for g in range(2):
        cnt = np.bincount(cd[:, g], minlength=320).astype(float); p = cnt / cnt.sum(); nz = p[p > 0]
        perp.append(np.exp(-(nz * np.log(nz)).sum()))
    return dict(resid=float(z["resid_cos"][m].mean()), ent=float(z["entropy"][m].mean()),
                perp=float(np.mean(perp)))

S = {c: frame_stats(c) for c in [BASE] + LANG + CTL}
mk = json.load(open(f"{RES}/masked_summary.json"))
for c in S: S[c].update(nce=mk[c]["nce_mean"], gap=mk[c]["pos_gap_mean"])

METRICS = [("resid", "Unmasked cosine residual\n(the originally specified metric)", False),
           ("ent", "Code-assignment entropy", False),
           ("perp", "Codebook perplexity", True),
           ("nce", "Masked contrastive loss (InfoNCE)", False),
           ("gap", "Positive-negative similarity gap", True)]

fig, axes = plt.subplots(1, 5, figsize=(19, 6.4), sharey=True)
rows = LANG + CTL
ypos = np.arange(len(rows))[::-1]
for ax, (key, title, higher_better) in zip(axes, METRICS):
    base = S[BASE][key]
    vals = [S[c][key] / base for c in rows]
    cols = ["#D1495B" if c in LANG else "#6C757D" for c in rows]
    ax.barh(ypos, vals, color=cols, height=.68)
    ax.axvline(1.0, color="#00798C", lw=2.2)
    ax.axvspan(0.9, 1.1, color="#00798C", alpha=.10)
    for y, v in zip(ypos, vals):
        ax.text(v + .03, y, f"{v:.2f}x", va="center", fontsize=8.6)
    ax.set_yticks(ypos); ax.set_yticklabels([NAME[c] for c in rows], fontsize=9.5)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_xlabel("ratio vs English control")
    ax.set_xlim(0, max(vals) * 1.28); ax.grid(alpha=.25, axis="x")
axes[0].text(.5, -.145, "teal band = within ±10% of English (the brief's \"no gap\" zone)",
             transform=axes[0].transAxes, ha="center", fontsize=9, color="#00798C")
fig.suptitle(f"Sensitivity ladder — do the metrics move at all?   {MODEL_SHORT}\n"
             "red = real languages   ·   grey = knowingly out-of-distribution audio (positive controls)",
             fontsize=13, weight="bold")
fig.tight_layout(rect=[0, .04, 1, .90])
fig.savefig(f"{RES}/sensitivity_ladder.png", dpi=160); plt.close(fig)
print(f"-> {RES}/sensitivity_ladder.png")
