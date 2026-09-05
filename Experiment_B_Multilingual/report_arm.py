#!/usr/bin/env python
"""Per-arm report. Each arm is analysed independently and printed in isolation so the
arms cannot anchor each other's reading."""
import json, sys, os
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse_b as A
from selection import PAIRS, META, ARM_ROLES, PROXIMITY_SENSITIVITY

RES = "results"
CK = {"xlsr53": "facebook/wav2vec2-large-xlsr-53",
      "xlsr300m": "facebook/wav2vec2-xls-r-300m"}


def verdict(p, dz, mde):
    if p < .05:
        return f"DISTINGUISHABLE from zero (p={p:.4f})"
    return (f"NOT distinguishable from zero (p={p:.4f}); |dz|={abs(dz):.2f} is "
            f"{'below' if abs(dz) < mde else 'above'} the MDE of {mde:.2f}")


def run_arm(tag, label):
    df = pd.read_parquet(f"{RES}/per_file_metrics{tag}.parquet")
    res, lmeans = {}, {}
    print("\n" + "#" * 100)
    print(f"#  ARM: {label}     ({tag.lstrip('_')})")
    print("#" * 100)

    for ck, full in CK.items():
        lm, r = A.report_checkpoint(df, ck, res)
        lmeans[full] = lm
        res[full] = res.pop(ck)

    lm53 = lmeans[CK["xlsr53"]]
    r53 = res[CK["xlsr53"]]

    # ---- language means
    out = lm53.reset_index()[["name", "group", "proximity", "family", "n_files",
                              "n_frames", "infonce", "pos_neg_gap", "residual",
                              "entropy", "perplexity"]]
    print("\n--- XLSR-53 language means (unit: file -> language) ---")
    print(out.sort_values(["group", "infonce"]).to_string(
        index=False, float_format=lambda v: f"{v:,.4f}"))

    # ---- primary: paired
    print("\n--- PRIMARY: paired tests on the 9 within-pair differences (unseen - seen) ---")
    print(f"{'metric':<16}{'mean diff':>11}{'95% CI':>22}{'dz':>8}{'Wilcoxon p':>12}"
          f"{'t-test p':>10}{'MDE dz':>8}")
    for m in ("infonce", "pos_neg_gap"):
        p = r53["paired"][m]
        ci = f"[{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}]"
        print(f"{m:<16}{p['mean_diff']:>+11.4f}{ci:>22}{p['cohens_dz']:>+8.2f}"
              f"{p['p_wilcoxon']:>12.4f}{p['p_paired_t']:>10.4f}{p['mde_dz_80pct']:>8.2f}")
    for m in ("infonce", "pos_neg_gap"):
        p = r53["paired"][m]
        print(f"  {m}: {verdict(p['p_wilcoxon'], p['cohens_dz'], p['mde_dz_80pct'])}")
        print(f"     MDE in raw units = {p['mde_raw_units']:.4f}  "
              f"(observed |diff| = {abs(p['mean_diff']):.4f})")

    # ---- per-pair detail
    print("\n--- the 9 pair differences ---")
    print(f"{'pair':<26}{'seen':>9}{'unseen':>9}{'diff':>9}   direction")
    for s, u, fam, desc in PAIRS:
        a, b = lm53.loc[s, "infonce"], lm53.loc[u, "infonce"]
        d = b - a
        print(f"{lm53.loc[s,'name']+'/'+lm53.loc[u,'name']:<26}{a:>9.4f}{b:>9.4f}{d:>+9.4f}"
              f"   {'unseen worse' if d > 0 else 'unseen better'}")

    # ---- secondary: unpaired
    print("\n--- SECONDARY: unpaired 9 v 9 ---")
    for m in ("infonce", "pos_neg_gap"):
        u = r53["unpaired"][m]
        print(f"{m:<14} seen={u['mean_seen']:.4f} unseen={u['mean_unseen']:.4f} "
              f"diff={u['mean_diff']:+.4f} CI[{u['ci_lo']:+.4f},{u['ci_hi']:+.4f}] "
              f"d={u['cohens_d']:+.2f} Welch p={u['p_welch']:.4f} "
              f"MWU p={u['p_mannwhitney']:.4f} MDE d={u['mde_d_80pct']:.2f}")

    # ---- duration diagnostics
    dd = A.duration_diagnostics(df, "xlsr53", lm53)
    print("\n--- duration confound, measured on FLEURS itself ---")
    if "rho_summary" in dd:
        s = dd["rho_summary"]
        print(f"  per-language Spearman(duration, InfoNCE) over {s['n_languages']} languages:")
        print(f"    median {s['median']:+.3f}   mean {s['mean']:+.3f}   "
              f"range [{s['min']:+.3f}, {s['max']:+.3f}]")
        print(f"    negative in {s['n_negative']}/{s['n_languages']}, "
              f"p<.05 in {s['n_significant']}/{s['n_languages']}")
    else:
        print("  (duration constant by construction in this arm)")
    if "pair_delta_correlation" in dd:
        c = dd["pair_delta_correlation"]
        print(f"  Spearman(pair duration delta, pair InfoNCE delta) over {c['n_pairs']} pairs: "
              f"rho={c['spearman_dur_vs_infonce']:+.3f} p={c['p_infonce']:.4f}")
        print(f"    -> {'CONFOUNDED: cropped arm should be primary' if c['p_infonce'] < .05 else 'no evidence the paired test is duration-driven'}")
    return res, lmeans, dd, df


if __name__ == "__main__":
    tag = sys.argv[1]
    label = ARM_ROLES[tag.lstrip("_")]
    run_arm(tag, label)
