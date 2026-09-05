#!/usr/bin/env python
"""Experiment B analysis. Pre-registered design in selection.py; nothing here re-picks groups.

Levels of aggregation: frame -> file -> language -> group. Frames within a file are
autocorrelated and files within a language share a speaker pool and recording setup,
so the language mean is the unit for every group-level test.
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selection import SELECTION, PAIRS, PROXIMITY, META, ALL, XLSR300_UNSEEN

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
NAME = {}   # filled from fleurs_languages.json
METRICS = [("infonce", "InfoNCE", "lower"), ("pos_neg_gap", "pos-neg gap", "higher"),
           ("residual", "cosine residual", "lower"), ("entropy", "entropy", "lower"),
           ("perplexity", "perplexity", "higher")]


# ------------------------------------------------------------------ power
def mde_paired(n, alpha=.05, power=.80):
    """Smallest Cohen's dz a paired test with n pairs can detect at the given power."""
    df, tc = n - 1, stats.t.ppf(1 - alpha / 2, n - 1)
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        ncp = mid * np.sqrt(n)
        p = (1 - stats.nct.cdf(tc, df, ncp)) + stats.nct.cdf(-tc, df, ncp)
        lo, hi = (lo, mid) if p > power else (mid, hi)
    return (lo + hi) / 2


def mde_unpaired(n1, n2, alpha=.05, power=.80):
    df, tc = n1 + n2 - 2, stats.t.ppf(1 - alpha / 2, n1 + n2 - 2)
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        ncp = mid * np.sqrt(n1 * n2 / (n1 + n2))
        p = (1 - stats.nct.cdf(tc, df, ncp)) + stats.nct.cdf(-tc, df, ncp)
        lo, hi = (lo, mid) if p > power else (mid, hi)
    return (lo + hi) / 2


def paired_test(d, label):
    """d = per-pair differences (unseen - seen) of language means."""
    n = len(d)
    m, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    dz = m / sd if sd > 0 else np.nan
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(.975, n - 1)
    ci = (m - tcrit * se, m + tcrit * se)
    t, p_t = stats.ttest_rel(d, np.zeros(n))
    try:
        w, p_w = stats.wilcoxon(d)
    except ValueError:
        w, p_w = np.nan, 1.0
    dz_min = mde_paired(n)
    return dict(label=label, n_pairs=n, mean_diff=m, sd_diff=sd, ci_lo=ci[0], ci_hi=ci[1],
                cohens_dz=dz, t=float(t), p_paired_t=float(p_t),
                wilcoxon_W=float(w), p_wilcoxon=float(p_w),
                mde_dz_80pct=dz_min, mde_raw_units=dz_min * sd)


def unpaired_test(x, y, label):
    n1, n2 = len(x), len(y)
    m1, m2 = float(np.mean(x)), float(np.mean(y))
    s1, s2 = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    d = (m2 - m1) / sp if sp > 0 else np.nan
    se = sp * np.sqrt(1 / n1 + 1 / n2)
    tcrit = stats.t.ppf(.975, n1 + n2 - 2)
    t, p_t = stats.ttest_ind(y, x, equal_var=False)
    u, p_u = stats.mannwhitneyu(y, x)
    dmin = mde_unpaired(n1, n2)
    return dict(label=label, n1=n1, n2=n2, mean_seen=m1, mean_unseen=m2,
                mean_diff=m2 - m1, ci_lo=(m2 - m1) - tcrit * se, ci_hi=(m2 - m1) + tcrit * se,
                cohens_d=d, welch_t=float(t), p_welch=float(p_t),
                mannwhitney_U=float(u), p_mannwhitney=float(p_u),
                mde_d_80pct=dmin, mde_raw_units=dmin * sp)


# ------------------------------------------------------------------ load
def language_means(df, ckpt):
    sub = df[df.checkpoint == ckpt]
    g = sub.groupby("language").agg(
        n_files=("file_id", "count"), n_frames=("n_frames", "sum"),
        **{m: (m, "mean") for m, _, _ in METRICS}).reset_index()
    g["group"] = g.language.map(lambda c: META[c]["group"])
    g["family"] = g.language.map(lambda c: META[c]["family"])
    g["proximity"] = g.language.map(lambda c: META[c].get("proximity", "-"))
    g["name"] = g.language.map(lambda c: NAME.get(c, c))
    return g.set_index("language")


def report_checkpoint(df, ckpt, out, drop=()):
    lm = language_means(df, ckpt)
    lm = lm.drop(index=[d for d in drop if d in lm.index])
    pairs = [(s, u) for s, u, _, _ in PAIRS if s in lm.index and u in lm.index]
    seen = [s for s, g in zip(lm.index, lm.group) if g == "SEEN"]
    unseen = [s for s, g in zip(lm.index, lm.group) if g == "UNSEEN"]
    res = {"checkpoint": ckpt, "dropped": list(drop), "paired": {}, "unpaired": {},
           "proximity": {}}
    for m, disp, _ in METRICS:
        d = np.array([lm.loc[u, m] - lm.loc[s, m] for s, u in pairs])
        res["paired"][m] = paired_test(d, f"{disp} (unseen - seen)")
        res["unpaired"][m] = unpaired_test(lm.loc[seen, m].values,
                                           lm.loc[unseen, m].values, disp)
        # CHANGE 2: UNSEEN split by proximity, each against the full SEEN group
        for rating in ("CLOSE", "DISTANT"):
            sel = [u for u in unseen if META[u].get("proximity") == rating]
            if len(sel) >= 2:
                res["proximity"].setdefault(m, {})[rating] = unpaired_test(
                    lm.loc[seen, m].values, lm.loc[sel, m].values,
                    f"{disp}: SEEN vs UNSEEN-{rating}")
    out[ckpt + ("_noXhosa" if drop else "")] = res
    return lm, res


# ------------------------------------------------------------------ dose-response
def xlsr53_tier():
    """Ordinal pre-training dose for XLSR-53: MLS(3) >> CommonVoice(2) > BABEL(1).
    XLSR-53 never published per-language hours, so corpus membership is the dose."""
    x = json.load(open(f"{RES}/xlsr53_languages.json"))
    iso = {n: i for n, i in zip(x["names"], [None] * len(x["names"]))}
    fl = {r["config"]: r for r in json.load(open(f"{RES}/fleurs_languages.json"))}
    name_of = {}
    for corpus, rank in [("MLS", 3), ("CommonVoice", 2), ("BABEL", 1)]:
        for n in x["by_corpus"][corpus]:
            name_of.setdefault(n, []).append((corpus, rank))
    out = {}
    for cfg, *_ in ALL:
        r = fl[cfg]
        hits = [v for n, v in name_of.items() if _iso_of(n, x) == r["iso"]]
        flat = [t for h in hits for t in h]
        if flat:
            best = max(flat, key=lambda t: t[1])
            out[cfg] = dict(tier=best[1], corpus="+".join(sorted({c for c, _ in flat})))
    return out


def _iso_of(name, x53):
    from build_language_lists import XLSR53_ISO
    return XLSR53_ISO.get(name)


def dose_response(df, lmeans):
    fl = {r["config"]: r for r in json.load(open(f"{RES}/fleurs_languages.json"))}
    out = {}
    # (a) XLS-R 300m: exact published hours vs that checkpoint's own fit
    lm = lmeans["facebook/wav2vec2-xls-r-300m"]
    rows = [(c, fl[c]["xlsr300_hours"], lm.loc[c, "infonce"], lm.loc[c, "pos_neg_gap"])
            for c in lm.index if fl[c]["xlsr300_hours"]]
    h = np.array([r[1] for r in rows]); n1 = np.array([r[2] for r in rows])
    g1 = np.array([r[3] for r in rows])
    for nm, y, hyp in [("infonce", n1, "more hours -> lower InfoNCE (rho < 0)"),
                       ("pos_neg_gap", g1, "more hours -> larger gap (rho > 0)")]:
        rho, p = stats.spearmanr(h, y)
        rl, pl = stats.spearmanr(np.log10(h), y)
        out[f"xlsr300m_hours_vs_{nm}"] = dict(
            n=len(rows), spearman_rho=float(rho), p=float(p),
            spearman_rho_log10h=float(rl), p_log10h=float(pl),
            hours_min=float(h.min()), hours_max=float(h.max()), hypothesis=hyp)
    out["xlsr300m_dose_table"] = [dict(language=NAME.get(c, c), hours=hh,
                                       infonce=float(a), pos_neg_gap=float(b))
                                  for (c, hh, a, b) in rows]
    # (b) XLSR-53: ordinal corpus tier, SEEN languages only
    tiers = xlsr53_tier()
    lm53 = lmeans["facebook/wav2vec2-large-xlsr-53"]
    rows = [(c, tiers[c]["tier"], tiers[c]["corpus"], lm53.loc[c, "infonce"])
            for c in lm53.index if c in tiers and META[c]["group"] in ("SEEN", "REFERENCE")]
    if len(rows) >= 4:
        t = np.array([r[1] for r in rows], float); y = np.array([r[3] for r in rows])
        rho, p = stats.spearmanr(t, y)
        out["xlsr53_tier_vs_infonce"] = dict(
            n=len(rows), spearman_rho=float(rho), p=float(p),
            note="dose = corpus tier (MLS 3 > CommonVoice 2 > BABEL 1); "
                 "XLSR-53 published no per-language hours",
            table=[dict(language=NAME.get(c, c), tier=int(tt), corpus=cc, infonce=float(yy))
                   for c, tt, cc, yy in rows])
    return out


def variance_explained(lm, metric):
    """eta^2 for group membership vs language family on the same language means."""
    y = lm[metric].values
    sst = float(((y - y.mean()) ** 2).sum())
    out = {}
    for factor in ("group", "family"):
        ssb = 0.0
        for _, sub in lm.groupby(factor):
            ssb += len(sub) * (sub[metric].mean() - y.mean()) ** 2
        out[factor] = float(ssb / sst) if sst > 0 else np.nan
    grp = [sub[metric].values for _, sub in lm.groupby("group")]
    fam = [sub[metric].values for _, sub in lm.groupby("family") if len(sub) > 1]
    out["anova_group_p"] = float(stats.f_oneway(*grp).pvalue) if len(grp) > 1 else np.nan
    out["anova_family_p"] = float(stats.f_oneway(*fam).pvalue) if len(fam) > 1 else np.nan
    return out


# ------------------------------------------------------------------ figures
def figures(lmeans, results):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"SEEN": "#00798C", "UNSEEN": "#D1495B", "REFERENCE": "#4A4A4A"}
    PC = {"CLOSE": "#E8A33D", "DISTANT": "#8E2C48"}
    lm53 = lmeans["facebook/wav2vec2-large-xlsr-53"]
    lm30 = lmeans["facebook/wav2vec2-xls-r-300m"]

    # 1-2. per-language metric relative to English, sorted
    for metric, disp, better in [("infonce", "InfoNCE", "lower"),
                                 ("pos_neg_gap", "positive-negative gap", "higher")]:
        fig, ax = plt.subplots(figsize=(11, 6.2))
        base = lm53.loc["en_us", metric]
        d = lm53.drop(index=["en_us"]).copy()
        d["rel"] = d[metric] / base
        d = d.sort_values("rel")
        cols = [C[g] for g in d.group]
        ax.barh(range(len(d)), d["rel"], color=cols, height=.7)
        ax.axvline(1.0, color="#333", lw=1.8)
        for i, (nm, v, g) in enumerate(zip(d["name"], d["rel"], d.group)):
            ax.text(v + .004, i, f"{v:.3f}", va="center", fontsize=8.5)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([f"{n}" for n in d["name"]], fontsize=10)
        ax.set_xlabel(f"{disp} relative to English  ({better} = better codebook fit)")
        ax.set_xlim(min(d["rel"]) * .96, max(d["rel"]) * 1.04)
        ax.grid(alpha=.25, axis="x")
        h = [plt.Rectangle((0, 0), 1, 1, color=C[k]) for k in ("SEEN", "UNSEEN")]
        ax.legend(h, ["SEEN by XLSR-53", "UNSEEN by XLSR-53"], frameon=False, fontsize=10)
        p = results["facebook/wav2vec2-large-xlsr-53"]["paired"][metric]
        ax.set_title(f"XLSR-53: {disp} by language, sorted\n"
                     f"paired Wilcoxon p={p['p_wilcoxon']:.3f}, "
                     f"Cohen's dz={p['cohens_dz']:+.2f}, MDE(80% power)={p['mde_dz_80pct']:.2f}",
                     fontsize=11.5, weight="bold")
        fig.tight_layout(); fig.savefig(f"{RES}/lang_{metric}_xlsr53.png", dpi=160); plt.close(fig)

    # 3. paired slopegraph - the primary analysis made visible
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, metric, disp in [(axes[0], "infonce", "InfoNCE (lower = better)"),
                             (axes[1], "pos_neg_gap", "pos-neg gap (higher = better)")]:
        for s, u, fam, desc in PAIRS:
            ys, yu = lm53.loc[s, metric], lm53.loc[u, metric]
            ax.plot([0, 1], [ys, yu], "-o", color="#888", ms=5, lw=1.3, alpha=.85)
            ax.text(-.04, ys, lm53.loc[s, "name"], ha="right", va="center", fontsize=8.5,
                    color=C["SEEN"])
            ax.text(1.04, yu, lm53.loc[u, "name"], ha="left", va="center", fontsize=8.5,
                    color=C["UNSEEN"])
        ax.set_xticks([0, 1]); ax.set_xticklabels(["SEEN", "UNSEEN"], fontsize=11)
        ax.set_xlim(-.45, 1.45); ax.set_title(disp, fontsize=11, weight="bold")
        ax.grid(alpha=.25, axis="y")
    fig.suptitle("The 9 family-matched pairs on XLSR-53 (primary analysis)\n"
                 "crossing lines = no consistent direction",
                 fontsize=12.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(f"{RES}/paired_slopegraph.png", dpi=160); plt.close(fig)

    # 4. dose-response
    fl = {r["config"]: r for r in json.load(open(f"{RES}/fleurs_languages.json"))}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, metric, disp in [(axes[0], "infonce", "InfoNCE"),
                             (axes[1], "pos_neg_gap", "pos-neg gap")]:
        xs, ys, nms, cs = [], [], [], []
        for c in lm30.index:
            h = fl[c]["xlsr300_hours"]
            if not h: continue
            xs.append(h); ys.append(lm30.loc[c, metric]); nms.append(lm30.loc[c, "name"])
            cs.append(C[lm30.loc[c, "group"]])
        xs, ys = np.array(xs), np.array(ys)
        ax.scatter(xs, ys, c=cs, s=52, zorder=3)
        for x, y, n in zip(xs, ys, nms):
            ax.annotate(n, (x, y), fontsize=7.6, xytext=(4, 4), textcoords="offset points")
        lx = np.log10(xs)
        b, a = np.polyfit(lx, ys, 1)
        xx = np.linspace(lx.min(), lx.max(), 50)
        ax.plot(10 ** xx, a + b * xx, "--", color="#333", lw=1.5, zorder=2)
        rho, p = stats.spearmanr(xs, ys)
        ax.set_xscale("log"); ax.set_xlabel("XLS-R pre-training hours for that language (log)")
        ax.set_ylabel(disp); ax.grid(alpha=.25)
        ax.set_title(f"{disp} vs pre-training hours\nSpearman rho={rho:+.2f}, p={p:.3f}",
                     fontsize=11, weight="bold")
    fig.suptitle("Dose-response on XLS-R 0.3B: does more pre-training data improve codebook fit?",
                 fontsize=12.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, .90])
    fig.savefig(f"{RES}/dose_response.png", dpi=160); plt.close(fig)

    # 5. checkpoint comparison + proximity subgroups
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, (ck, lm, ttl) in zip(axes, [
            ("facebook/wav2vec2-large-xlsr-53", lm53, "XLSR-53  (split is real here)"),
            ("facebook/wav2vec2-xls-r-300m", lm30, "XLS-R 0.3B  (negative control)")]):
        pos, ticks = 0, []
        for lab, sel, col in [("SEEN", [c for c in lm.index if lm.loc[c, "group"] == "SEEN"], C["SEEN"]),
                              ("UNSEEN\nCLOSE", [c for c in lm.index if META[c].get("proximity") == "CLOSE"], PC["CLOSE"]),
                              ("UNSEEN\nDISTANT", [c for c in lm.index if META[c].get("proximity") == "DISTANT"], PC["DISTANT"])]:
            v = lm.loc[sel, "infonce"].values
            bp = ax.boxplot(v, positions=[pos], widths=.6, patch_artist=True, showfliers=False)
            bp["boxes"][0].set(facecolor=col, alpha=.62)
            ax.scatter(np.random.default_rng(0).normal(pos, .055, len(v)), v,
                       color="#222", s=22, zorder=4)
            ticks.append(f"{lab}\n(n={len(v)})"); pos += 1
        ax.axhline(lm.loc["en_us", "infonce"], color="#333", ls=":", lw=1.4)
        ax.text(2.42, lm.loc["en_us", "infonce"], " English", fontsize=8.5, va="center")
        ax.set_xticks(range(3)); ax.set_xticklabels(ticks, fontsize=9.5)
        ax.set_ylabel("InfoNCE (lower = better fit)"); ax.grid(alpha=.25, axis="y")
        ax.set_title(ttl, fontsize=11, weight="bold")
    fig.suptitle("Membership vs proximity, on both checkpoints", fontsize=12.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, .91])
    fig.savefig(f"{RES}/checkpoint_and_proximity.png", dpi=160); plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    global NAME
    NAME.update({r["config"]: r["name"] for r in json.load(open(f"{RES}/fleurs_languages.json"))})
    df = pd.read_parquet(f"{RES}/per_file_metrics.parquet")
    CK = {"xlsr53": "facebook/wav2vec2-large-xlsr-53",
          "xlsr300m": "facebook/wav2vec2-xls-r-300m"}
    results, lmeans = {}, {}
    for tag, full in CK.items():
        lm, res = report_checkpoint(df, tag, results)
        results[full] = results.pop(tag)
        lmeans[full] = lm
    # CHANGE 3: negative-control arm re-run without Xhosa
    _, res_nx = report_checkpoint(df, "xlsr300m", results, drop=tuple(XLSR300_UNSEEN))
    results["facebook/wav2vec2-xls-r-300m_noXhosa"] = results.pop("xlsr300m_noXhosa")
    _, res53_nx = report_checkpoint(df, "xlsr53", results, drop=tuple(XLSR300_UNSEEN))
    results["facebook/wav2vec2-large-xlsr-53_noXhosa"] = results.pop("xlsr53_noXhosa")

    dose = dose_response(df, lmeans)
    var = {ck: {m: variance_explained(lm[lm.group != "REFERENCE"], m)
                for m, _, _ in METRICS} for ck, lm in lmeans.items()}
    json.dump(dict(group_tests=results, dose_response=dose, variance_explained=var),
              open(f"{RES}/statistics.json", "w"), indent=1, default=float)

    pd.set_option("display.width", 250)
    for ck, lm in lmeans.items():
        out = lm.reset_index()[["name", "group", "proximity", "family", "n_files",
                                "n_frames", "infonce", "pos_neg_gap", "residual",
                                "entropy", "perplexity"]]
        out.to_csv(f"{RES}/language_means_{ck.split('/')[-1]}.csv", index=False)
        print(f"\n{'='*118}\n  LANGUAGE MEANS - {ck}\n{'='*118}")
        print(out.sort_values(["group", "infonce"]).to_string(
            index=False, float_format=lambda v: f"{v:,.4f}"))
    return results, dose, var, lmeans


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------ duration confound
def duration_diagnostics(df, ckpt, lm):
    """Characterise the duration confound ON FLEURS ITSELF, and test whether it
    contaminates the paired comparison. Reported unconditionally (pre-registered)."""
    sub = df[df.checkpoint == ckpt]
    out = {}

    # (3) per-language rho(duration, InfoNCE) - the confound's actual size here,
    # rather than extrapolated from Stage A's VAD-segmented YouTube corpus.
    per = {}
    for cfg, g in sub.groupby("language"):
        if g.duration.nunique() < 5:
            continue                      # cropped arm: duration is constant by design
        r1, p1 = stats.spearmanr(g.duration, g.infonce)
        r2, p2 = stats.spearmanr(g.duration, g.pos_neg_gap)
        per[cfg] = dict(name=NAME.get(cfg, cfg), n=len(g),
                        rho_dur_infonce=float(r1), p_infonce=float(p1),
                        rho_dur_gap=float(r2), p_gap=float(p2),
                        mean_duration=float(g.duration.mean()))
    if per:
        rr = np.array([v["rho_dur_infonce"] for v in per.values()])
        out["per_language"] = per
        out["rho_summary"] = dict(
            n_languages=len(rr), median=float(np.median(rr)), mean=float(rr.mean()),
            min=float(rr.min()), max=float(rr.max()),
            n_negative=int((rr < 0).sum()), n_significant=int(
                sum(v["p_infonce"] < .05 for v in per.values())))

    # (2) within-pair duration deltas vs within-pair InfoNCE deltas across the 9 pairs
    dur = sub.groupby("language").duration.mean()
    rows = []
    for s, u, fam, desc in PAIRS:
        if s not in lm.index or u not in lm.index:
            continue
        rows.append(dict(pair=desc.split(" - ")[0].strip(), family=fam,
                         seen=NAME.get(s, s), unseen=NAME.get(u, u),
                         dur_seen=float(dur[s]), dur_unseen=float(dur[u]),
                         dur_delta=float(dur[s] - dur[u]),
                         infonce_delta=float(lm.loc[u, "infonce"] - lm.loc[s, "infonce"]),
                         gap_delta=float(lm.loc[u, "pos_neg_gap"] - lm.loc[s, "pos_neg_gap"])))
    if len(rows) >= 4:
        dd = np.array([r["dur_delta"] for r in rows])
        ni = np.array([r["infonce_delta"] for r in rows])
        gg = np.array([r["gap_delta"] for r in rows])
        r1, p1 = stats.spearmanr(dd, ni)
        r2, p2 = stats.spearmanr(dd, gg)
        out["pair_duration_deltas"] = rows
        out["pair_delta_correlation"] = dict(
            n_pairs=len(rows),
            spearman_dur_vs_infonce=float(r1), p_infonce=float(p1),
            spearman_dur_vs_gap=float(r2), p_gap=float(p2),
            mean_abs_dur_delta=float(np.abs(dd).mean()),
            mean_signed_dur_delta=float(dd.mean()),
            n_seen_longer=int((dd > 0).sum()), n_unseen_longer=int((dd < 0).sum()),
            interpretation=("strong correlation would mean the paired test is "
                            "confounded by duration and the cropped arm is primary"))
    return out


def proximity_sensitivity(lm, metric="infonce"):
    """Pre-registered: repeat the subgroup split with Burmese moved CLOSE."""
    from selection import PROXIMITY_SENSITIVITY
    seen = [c for c in lm.index if lm.loc[c, "group"] == "SEEN"]
    out = {}
    for label, override in [("as_registered", {}), ("burmese_as_CLOSE", PROXIMITY_SENSITIVITY)]:
        rate = {c: override.get(c, META[c].get("proximity")) for c in lm.index
                if META[c].get("proximity")}
        grp = {}
        for r in ("CLOSE", "DISTANT"):
            sel = [c for c, v in rate.items() if v == r]
            vals = lm.loc[sel, metric].values
            grp[r] = dict(n=len(sel), languages=[NAME.get(c, c) for c in sel],
                          values=[float(v) for v in vals], mean=float(vals.mean()),
                          seen_mean=float(lm.loc[seen, metric].mean()),
                          diff_vs_seen=float(vals.mean() - lm.loc[seen, metric].mean()))
            # descriptive only for DISTANT (n=4, MDE d>=1.85); test kept for CLOSE
            if r == "CLOSE" and len(sel) >= 3:
                grp[r]["test"] = unpaired_test(lm.loc[seen, metric].values, vals,
                                               f"SEEN vs UNSEEN-CLOSE ({metric})")
        out[label] = grp
    return out
