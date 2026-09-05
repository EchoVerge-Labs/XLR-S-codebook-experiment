#!/usr/bin/env python
"""Aggregate Stage A per-frame arrays into statistics, figures and CSVs."""
import os, json, itertools
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/EchoVerge-LABS/Experiment_A_Diagnostic")
RES = os.environ.get("XLSR_RESULTS", os.path.join(ROOT, "results"))
MODEL_NAME = os.environ.get("XLSR_MODEL", "facebook/wav2vec2-xls-r-300m")
MODEL_SHORT = {"facebook/wav2vec2-xls-r-300m": "XLS-R 0.3B (wav2vec2-xls-r-300m)",
               "facebook/wav2vec2-large-xlsr-53": "XLSR-53 (wav2vec2-large-xlsr-53)"}.get(MODEL_NAME, MODEL_NAME)
CORPORA = ["Sinhala_YT", "Tamil_YT", "English_CV", "Sinhala_read", "Tamil_read"]
ARMS = {"in_the_wild": ["Sinhala_YT", "Tamil_YT", "English_CV"],
        "read_matched": ["Sinhala_read", "Tamil_read", "English_CV"]}
COLOR = {"Sinhala_YT": "#D1495B", "Tamil_YT": "#EDAE49", "English_CV": "#00798C",
         "Sinhala_read": "#8B2635", "Tamil_read": "#B07D2B"}
LABEL = {"Sinhala_YT": "Sinhala (YouTube)", "Tamil_YT": "Tamil (YouTube)",
         "English_CV": "English (Common Voice)", "Sinhala_read": "Sinhala (OpenSLR, read)",
         "Tamil_read": "Tamil (OpenSLR, read)"}
G, V = 2, 320
SPEECH_Q, SPEECH_FRAC = 0.90, 0.10     # frame is "speech-like" if rms > 10% of file p90


def load():
    D = {}
    for c in CORPORA:
        z = np.load(f"{RES}/frames_{c}.npz")
        d = {k: z[k] for k in z.files}
        # per-file relative-energy mask, robust to gain differences between corpora
        fid, rms = d["file_id"], d["rms"]
        p90 = np.zeros(fid.max() + 1, dtype=np.float32)
        for f in range(fid.max() + 1):
            p90[f] = np.quantile(rms[fid == f], SPEECH_Q)
        d["speech"] = rms > SPEECH_FRAC * p90[fid]
        d["perfile"] = pd.DataFrame(json.load(open(f"{RES}/perfile_{c}.json")))
        D[c] = d
    return D


def usage_stats(codes, n_frames=None, seed=0):
    """Perplexity / utilisation of the codebook, optionally at a matched frame budget."""
    if n_frames is not None and n_frames < len(codes):
        codes = codes[np.random.default_rng(seed).choice(len(codes), n_frames, replace=False)]
    perp, util, out = [], [], {}
    for g in range(G):
        cnt = np.bincount(codes[:, g], minlength=V).astype(np.float64)
        p = cnt / cnt.sum()
        nz = p[p > 0]
        perp.append(float(np.exp(-(nz * np.log(nz)).sum())))
        util.append(float((cnt > 0).mean()))
    # joint code (pair of group indices) usage
    joint = codes[:, 0].astype(np.int32) * V + codes[:, 1]
    jc = np.bincount(joint, minlength=V * V).astype(np.float64)
    jp = jc / jc.sum(); jnz = jp[jp > 0]
    out.update(perplexity_g0=perp[0], perplexity_g1=perp[1],
               perplexity_mean=float(np.mean(perp)),
               utilisation_g0=util[0], utilisation_g1=util[1],
               utilisation_mean=float(np.mean(util)),
               joint_perplexity=float(np.exp(-(jnz * np.log(jnz)).sum())),
               joint_codes_used=int((jc > 0).sum()), n_frames_used=int(len(codes)))
    return out



_CV_SPK = None
def source_id(corpus, src):
    """Recording-source identity for a file: speaker where known, else the source
    video. Code diversity tracks speaker/session diversity, so perplexity is only
    comparable across corpora once the number of distinct sources is matched."""
    global _CV_SPK
    b = os.path.basename(src)
    if corpus == "English_CV":
        if _CV_SPK is None:
            import csv
            _CV_SPK = {r["path"]: r["client_id"]
                       for r in csv.DictReader(open(f"{ROOT}/data/cv_en_dev.tsv"), delimiter="\t")}
        return _CV_SPK.get(b, b)
    if corpus == "Sinhala_read":                 # sin_<speaker>_<utt>.wav
        return b.split("_")[1] if b.startswith("sin_") else b
    if corpus == "Tamil_read":                   # ta{f,m}_<speaker>_<utt>.wav
        pr = b.split("_")
        return f"{pr[0]}_{pr[1]}" if len(pr) > 2 else b
    return os.path.basename(os.path.dirname(src))   # YouTube: the source video


def source_matched_usage(D, corpus, n_sources, frames_per_source, seed=0):
    """Codebook usage over frames drawn from a matched number of distinct sources."""
    d = D[corpus]; pf = d["perfile"]
    rng = np.random.default_rng(seed)
    by_src = {}
    for _, r in pf.iterrows():
        by_src.setdefault(source_id(corpus, r["src"]), []).append(int(r["file_id"]))
    srcs = sorted(by_src)
    if len(srcs) < n_sources:
        return None
    pick = rng.choice(len(srcs), n_sources, replace=False)
    m, fid = d["speech"], d["file_id"]
    codes_all = d["codes"]
    chunks = []
    for i in pick:
        ids = set(by_src[srcs[i]])
        sel = m & np.isin(fid, list(ids))
        c = codes_all[sel]
        if len(c) == 0:
            continue
        take = min(frames_per_source, len(c))
        chunks.append(c[rng.choice(len(c), take, replace=False)])
    if not chunks:
        return None
    return usage_stats(np.concatenate(chunks))


def summarise(D):
    budget = min(int(D[c]["speech"].sum()) for c in CORPORA)
    rows = []
    for c in CORPORA:
        d = D[c]; m = d["speech"]
        r, pf = d["resid_cos"][m], d["perfile"]
        row = dict(corpus=c, label=LABEL[c], n_files=len(pf),
                   n_frames_total=int(len(d["resid_cos"])), n_frames_speech=int(m.sum()),
                   hours=float(pf["duration"].sum() / 3600),
                   resid_cos_mean=float(r.mean()), resid_cos_median=float(np.median(r)),
                   resid_cos_std=float(r.std()), resid_cos_p95=float(np.quantile(r, .95)),
                   resid_l2sq_mean=float(d["resid_l2sq"][m].mean()),
                   resid_l2sq_median=float(np.median(d["resid_l2sq"][m])),
                   resid_l2sq_p95=float(np.quantile(d["resid_l2sq"][m], .95)),
                   entropy_mean=float(d["entropy"][m].mean()),
                   maxprob_mean=float(d["maxprob"][m].mean()),
                   margin_mean=float(d["margin"][m].mean()))
        row.update({f"raw_{k}": v for k, v in usage_stats(d["codes"][m]).items()})
        row.update({f"matched_{k}": v for k, v in usage_stats(d["codes"][m], budget).items()})
        row["n_sources"] = len({source_id(c, r) for r in pf["src"]})
        rows.append(row)
    df = pd.DataFrame(rows)
    # source-matched codebook usage: equalise BOTH distinct sources and frames
    K = int(df["n_sources"].min())
    F = max(1, budget // K)
    sm = []
    for c in CORPORA:
        u = source_matched_usage(D, c, K, F)
        sm.append({f"srcmatched_{k}": v for k, v in (u or {}).items()})
    df = pd.concat([df, pd.DataFrame(sm)], axis=1)
    df.attrs["K"], df.attrs["F"] = K, F
    return df, budget


def file_level_tests(D):
    """Compare corpora using per-file means: frames within a file are autocorrelated,
    so files - not frames - are the independent unit."""
    out = []
    for arm, members in ARMS.items():
        for a, b in itertools.combinations(members, 2):
            for metric in ["mean_resid_cos", "mean_entropy", "mean_margin"]:
                x = D[a]["perfile"][metric].values
                y = D[b]["perfile"][metric].values
                t, p = stats.ttest_ind(x, y, equal_var=False)
                u, pu = stats.mannwhitneyu(x, y)
                pooled = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
                out.append(dict(arm=arm, a=a, b=b, metric=metric,
                                mean_a=x.mean(), mean_b=y.mean(),
                                ratio_a_over_b=x.mean() / y.mean(),
                                cohens_d=(x.mean() - y.mean()) / pooled,
                                welch_p=p, mannwhitney_p=pu))
    return pd.DataFrame(out)


def fig_residual_distribution(D):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    specs = [("in_the_wild", "resid_cos", "Cosine residual  1 - cos(project_hid(c), project_q(q))"),
             ("read_matched", "resid_cos", "Cosine residual  1 - cos(project_hid(c), project_q(q))"),
             ("in_the_wild", "entropy", "Code-assignment posterior entropy (nats)"),
             ("read_matched", "entropy", "Code-assignment posterior entropy (nats)")]
    titles = ["As requested: in-the-wild Si/Ta vs read English\n(language AND acoustic domain differ)",
              "Domain-matched control: read Si/Ta vs read English\n(language differs, domain held constant)",
              "", ""]
    for ax, (arm, metric, xlab), ttl in zip(axes.ravel(), specs, titles):
        for c in ARMS[arm]:
            d = D[c]; m = d["speech"]
            v = d[metric][m]
            v = v.mean(axis=1) if v.ndim > 1 else v
            lo, hi = np.quantile(v, [.001, .995])
            ax.hist(v, bins=140, range=(lo, hi), density=True, histtype="step", lw=2.1,
                    color=COLOR[c], label=f"{LABEL[c]}  (mean {v.mean():.3f})")
            ax.axvline(v.mean(), color=COLOR[c], ls=":", lw=1.2, alpha=.8)
        ax.set_xlabel(xlab); ax.set_ylabel("density")
        ax.legend(fontsize=8.5, frameon=False); ax.grid(alpha=.25)
        if ttl: ax.set_title(ttl, fontsize=10.5, weight="bold")
    fig.suptitle(f"Quantization Residual Distribution by Language — {MODEL_SHORT}, frozen codebook",
                 fontsize=13.5, weight="bold")
    fig.text(.5, .005, "Speech-like frames only. Left column is the comparison as originally specified; "
             "right column holds acoustic domain constant to isolate the language effect.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, .022, 1, .95])
    fig.savefig(f"{RES}/residual_distribution.png", dpi=160); plt.close(fig)


def fig_codebook_usage(D):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for j, c in enumerate(["Sinhala_YT", "Tamil_YT", "English_CV"]):
        d = D[c]; codes = d["codes"][d["speech"]]
        for g in range(G):
            ax = axes[g, j]
            cnt = np.bincount(codes[:, g], minlength=V).astype(float)
            cnt = 100 * cnt / cnt.sum()
            order = np.argsort(cnt)[::-1]
            top, bot = cnt[order[:50]], cnt[order[-50:]]
            ax.bar(range(50), top, color=COLOR[c], width=1.0)
            ax.bar(range(55, 105), bot, color="#999999", width=1.0)
            ax.axvline(52.5, color="k", lw=.8, ls="--")
            ax.set_title(f"{LABEL[c]} — group {g}", fontsize=10)
            ax.set_ylabel("% of frames"); ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_xticks([25, 80]); ax.set_xticklabels(["top-50 codes", "bottom-50 codes"], fontsize=9)
            u = (np.bincount(codes[:, g], minlength=V) > 0).mean()
            ax.text(.98, .93, f"top-50 = {top.sum():.1f}% of mass\nutilisation {u*100:.0f}%",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                    bbox=dict(fc="white", ec="none", alpha=.75))
            ax.grid(alpha=.25, axis="y")
    fig.suptitle("Codebook usage: most- vs least-used entries per language (G=2 groups x V=320 entries)",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, .945])
    fig.savefig(f"{RES}/codebook_usage.png", dpi=160); plt.close(fig)


def fig_genre(D):
    """How much does acoustic domain alone move the residual, within one language?"""
    fig, ax = plt.subplots(figsize=(11, 5.2))
    rows, pos, ticks = [], 0, []
    for c in ["Sinhala_YT", "Tamil_YT"]:
        pf = D[c]["perfile"]
        for g, sub in pf.groupby("genre"):
            rows.append((f"{c.split('_')[0]}\n{g}", sub["mean_resid_cos"].values, COLOR[c]))
    for c in ["Sinhala_read", "Tamil_read", "English_CV"]:
        rows.append((LABEL[c].split(" (")[0] + "\n" + ("read" if "read" in c else "read CV"),
                     D[c]["perfile"]["mean_resid_cos"].values, COLOR[c]))
    for name, vals, col in rows:
        bp = ax.boxplot(vals, positions=[pos], widths=.62, patch_artist=True, showfliers=False)
        bp["boxes"][0].set(facecolor=col, alpha=.65); ticks.append(name); pos += 1
    ax.set_xticks(range(len(ticks))); ax.set_xticklabels(ticks, fontsize=7.6)
    ax.set_ylabel("per-file mean cosine residual"); ax.grid(alpha=.25, axis="y")
    ax.set_title("Residual varies more with acoustic domain (genre) than with language",
                 fontsize=12, weight="bold")
    fig.tight_layout(); fig.savefig(f"{RES}/residual_by_genre.png", dpi=160); plt.close(fig)


def interpret(summ, tests):
    L = []
    add = L.append
    add("=" * 96); add("  INTERPRETATION"); add("=" * 96)
    s = summ.set_index("corpus")
    for arm, members in ARMS.items():
        eng = "English_CV"
        add(f"\n--- ARM: {arm} ---")
        for c in members:
            if c == eng: continue
            for metric, name, higher_worse in [("resid_cos_mean", "cosine residual", True),
                                               ("entropy_mean", "assignment entropy", True),
                                               ("margin_mean", "assignment margin", False)]:
                r = s.loc[c, metric] / s.loc[eng, metric]
                add(f"  {c:<14} {name:<20} ratio vs English = {r:5.3f}x")
        for c in members:
            if c == eng: continue
            r = s.loc[c, "resid_cos_mean"] / s.loc[eng, "resid_cos_mean"]
            v = ("STRONG evidence of codebook gap. Extension is well-motivated." if r > 1.5 else
                 "MODERATE evidence. Extension may help but is not guaranteed." if r > 1.1 else
                 "NO meaningful gap detected (within ~10% of English control).")
            add(f"  -> {c} residual ratio {r:.3f}x : {v}")
        add(f"  perplexity (frame-matched, mean over groups): " +
            ", ".join(f"{c}={s.loc[c,'matched_perplexity_mean']:.1f}" for c in members))
        add(f"  utilisation (frame-matched): " +
            ", ".join(f"{c}={s.loc[c,'matched_utilisation_mean']*100:.1f}%" for c in members))
        add(f"  distinct recording sources in sample: " +
            ", ".join(f"{c}={int(s.loc[c,'n_sources'])}" for c in members))
        add(f"  perplexity (source- AND frame-matched): " +
            ", ".join(f"{c}={s.loc[c,'srcmatched_perplexity_mean']:.1f}" for c in members))
    return "\n".join(L)


def main():
    D = load()
    summ, budget = summarise(D)
    tests = file_level_tests(D)
    summ.to_csv(f"{RES}/summary_statistics.csv", index=False)
    tests.to_csv(f"{RES}/significance_tests.csv", index=False)

    cols = ["label", "n_files", "n_frames_speech", "hours", "resid_cos_mean", "resid_cos_median",
            "resid_cos_std", "resid_cos_p95", "entropy_mean", "margin_mean",
            "matched_perplexity_mean", "matched_utilisation_mean",
            "n_sources", "srcmatched_perplexity_mean", "srcmatched_utilisation_mean"]
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print("\n" + "=" * 96)
    print(f"  SUMMARY  (speech-like frames; usage stats at matched budget of {budget:,} frames)")
    print("=" * 96)
    print(summ[cols].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print("\n" + "=" * 96); print("  FILE-LEVEL SIGNIFICANCE (per-file means; Welch t-test)"); print("=" * 96)
    t = tests[tests.metric == "mean_resid_cos"]
    print(t[["arm", "a", "b", "mean_a", "mean_b", "ratio_a_over_b", "cohens_d", "welch_p"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    txt = interpret(summ, tests)
    print("\n" + txt)
    open(f"{RES}/interpretation.txt", "w").write(txt)

    fig_residual_distribution(D); fig_codebook_usage(D); fig_genre(D)
    print(f"\nfigures + CSVs -> {RES}/")


if __name__ == "__main__":
    main()
