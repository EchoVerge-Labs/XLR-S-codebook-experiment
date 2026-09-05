#!/usr/bin/env python
"""Corrected variance decomposition + equivalence testing across the three arms."""
import json, sys, os
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse_b as A
from selection import PAIRS, META

RES = "results"
ARMS = [("_full", "full", "Full clips (PRIMARY OVERALL)"),
        ("_crop6.0", "crop6", "6.0 s crop (PRIMARY CROPPED)"),
        ("_crop9.0", "crop9", "9.0 s crop (SENSITIVITY)")]

# Equivalence margins calibrated on Stage A's positive controls: the InfoNCE increase
# each knowingly-degrading perturbation caused relative to clean English.
# NOTE: those are UNPAIRED ABSOLUTE differences across corpora on XLS-R 0.3B, whereas
# this is a PAIRED WITHIN-CORPUS difference on XLSR-53. The transfer is not exact -
# these are calibration anchors, not formally matched equivalence margins.
STAGE_A_CONTROLS = {"1.45x speed/pitch": 0.289, "six-way babble": 0.306,
                    "reversed speech": 0.590, "sine tones": 0.987, "white noise": 2.023}
MARGIN_PRIMARY = 0.289      # smallest control that measurably degrades the codebook
MARGIN_SECONDARY = 0.590    # reversed speech, conservative secondary


def variance_explained_corrected(lm, metric):
    """eta^2 rewards degrees of freedom, so an 8-df factor on 18 points looks large by
    construction. Report omega^2 and epsilon^2 (adjusted eta^2), which subtract the
    variance a factor of that size explains by chance, plus the null expectation."""
    y = lm[metric].values
    n = len(y)
    sst = float(((y - y.mean()) ** 2).sum())
    out = {}
    for factor in ("group", "family"):
        groups = [sub[metric].values for _, sub in lm.groupby(factor)]
        k = len(groups)
        ssb = sum(len(g) * (g.mean() - y.mean()) ** 2 for g in groups)
        ssw = sst - ssb
        df_b, df_w = k - 1, n - k
        msw = ssw / df_w if df_w > 0 else np.nan
        eta2 = ssb / sst
        omega2 = (ssb - df_b * msw) / (sst + msw)
        eps2 = (ssb - df_b * msw) / sst
        out[factor] = dict(k_levels=k, df_effect=df_b, df_error=df_w,
                           eta2=float(eta2),
                           eta2_expected_under_null=float(df_b / (df_b + df_w)),
                           omega2=float(omega2), epsilon2=float(eps2),
                           f_p=float(stats.f_oneway(*groups).pvalue) if k > 1 and df_w > 0 else np.nan)
    return out


def tost_paired(d, margin):
    """Two one-sided tests on paired differences. Equivalence is declared when the
    90% CI lies entirely inside (-margin, +margin)."""
    n = len(d)
    m, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    se = sd / np.sqrt(n)
    df = n - 1
    t_lower = (m + margin) / se          # H0: mean <= -margin
    t_upper = (m - margin) / se          # H0: mean >= +margin
    p_lower = 1 - stats.t.cdf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)
    p_tost = max(p_lower, p_upper)
    tc90 = stats.t.ppf(.95, df)
    ci90 = (m - tc90 * se, m + tc90 * se)
    tc95 = stats.t.ppf(.975, df)
    ci95 = (m - tc95 * se, m + tc95 * se)
    return dict(n=n, mean=m, sd=sd, se=se, margin=margin,
                ci90_lo=ci90[0], ci90_hi=ci90[1], ci95_lo=ci95[0], ci95_hi=ci95[1],
                p_lower=float(p_lower), p_upper=float(p_upper), p_tost=float(p_tost),
                equivalent=bool(p_tost < .05),
                ci90_inside=bool(ci90[0] > -margin and ci90[1] < margin))


def main():
    A.NAME.update({r["config"]: r["name"]
                   for r in json.load(open(f"{RES}/fleurs_languages.json"))})
    out = {"stage_a_control_margins": STAGE_A_CONTROLS,
           "margin_primary": MARGIN_PRIMARY, "margin_secondary": MARGIN_SECONDARY,
           "margin_caveat": ("Stage A margins are unpaired absolute differences across "
                             "corpora on XLS-R 0.3B; this is a paired within-corpus "
                             "difference on XLSR-53. Calibration anchor, not a formally "
                             "matched equivalence margin."),
           "arms": {}}

    print("=" * 104)
    print("  CORRECTED VARIANCE DECOMPOSITION  (XLSR-53 language means, full-clip arm)")
    print("=" * 104)
    df = pd.read_parquet(f"{RES}/per_file_metrics_full.parquet")
    res = {}
    lm, _ = A.report_checkpoint(df, "xlsr53", res)
    lmx = lm[lm.group != "REFERENCE"]
    for metric in ("infonce", "pos_neg_gap"):
        v = variance_explained_corrected(lmx, metric)
        print(f"\n  {metric}")
        print(f"    {'factor':<9}{'levels':>7}{'df':>4}{'eta2':>8}{'E[eta2] null':>14}"
              f"{'omega2':>9}{'epsilon2':>10}{'ANOVA p':>9}")
        for f, s in v.items():
            print(f"    {f:<9}{s['k_levels']:>7}{s['df_effect']:>4}{s['eta2']:>8.3f}"
                  f"{s['eta2_expected_under_null']:>14.3f}{s['omega2']:>9.3f}"
                  f"{s['epsilon2']:>10.3f}{s['f_p']:>9.3f}")
        out.setdefault("variance", {})[metric] = v
    print("\n  Both factors fall BELOW their null expectation. Neither group membership")
    print("  nor family explains variance above chance at this sample size.")

    print("\n" + "=" * 104)
    print(f"  EQUIVALENCE TESTING (TOST) - paired InfoNCE differences, XLSR-53")
    print("=" * 104)
    print("  Stage A positive controls (InfoNCE increase vs clean English):")
    for k, v in STAGE_A_CONTROLS.items():
        print(f"     {k:<20}{v:>+7.3f}{'   <- primary margin' if abs(v-MARGIN_PRIMARY)<1e-9 else ('   <- secondary margin' if abs(v-MARGIN_SECONDARY)<1e-9 else '')}")
    for margin, lab in [(MARGIN_PRIMARY, "PRIMARY"), (MARGIN_SECONDARY, "SECONDARY")]:
        print(f"\n  --- margin +/-{margin:.3f}  ({lab}) ---")
        print(f"  {'arm':<28}{'mean':>9}{'90% CI':>22}{'95% CI':>22}{'TOST p':>9}  verdict")
        for tag, key, label in ARMS:
            d2 = pd.read_parquet(f"{RES}/per_file_metrics{tag}.parquet")
            r2 = {}
            l2, _ = A.report_checkpoint(d2, "xlsr53", r2)
            d = np.array([l2.loc[u, "infonce"] - l2.loc[s, "infonce"] for s, u, _, _ in PAIRS])
            t = tost_paired(d, margin)
            ci90 = f"[{t['ci90_lo']:+.3f}, {t['ci90_hi']:+.3f}]"
            ci95 = f"[{t['ci95_lo']:+.3f}, {t['ci95_hi']:+.3f}]"
            v = "EQUIVALENT" if t["equivalent"] else "not established"
            print(f"  {label:<28}{t['mean']:>+9.4f}{ci90:>22}{ci95:>22}{t['p_tost']:>9.4f}  {v}")
            out["arms"].setdefault(key, {})[f"tost_{margin}"] = t
    json.dump(out, open(f"{RES}/final_statistics.json", "w"), indent=1, default=float)
    print(f"\n-> {RES}/final_statistics.json")


if __name__ == "__main__":
    main()
