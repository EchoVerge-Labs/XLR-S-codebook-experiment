#!/usr/bin/env python
"""Step 6 - plain-language summary of Experiment B."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selection import PAIRS, PROXIMITY, META
import analyse_b as A

RES = A.RES
X53, X300 = "facebook/wav2vec2-large-xlsr-53", "facebook/wav2vec2-xls-r-300m"


def verdict(p, dz, mde, better_when):
    sig = p < .05
    if sig:
        return ("DISTINGUISHABLE FROM ZERO", "worse" if dz > 0 else "better")
    return ("NOT distinguishable from zero", None)


def main():
    L = []
    a = L.append
    st = json.load(open(f"{RES}/statistics.json"))
    g = st["group_tests"]

    a("=" * 100)
    a("  EXPERIMENT B - does presence in XLSR-53's pre-training list predict codebook fit?")
    a("=" * 100)

    a("\n--- 1. PRIMARY: paired test on 9 family-matched pairs (XLSR-53) ---")
    a(f"{'metric':<18}{'mean diff':>11}{'95% CI':>22}{'dz':>8}{'Wilcoxon p':>12}"
      f"{'paired-t p':>12}{'MDE dz':>9}")
    for m, disp, _ in A.METRICS:
        p = g[X53]["paired"][m]
        ci = f"[{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}]"
        a(f"{disp:<18}{p['mean_diff']:>+11.4f}{ci:>22}{p['cohens_dz']:>+8.2f}"
          f"{p['p_wilcoxon']:>12.4f}{p['p_paired_t']:>12.4f}{p['mde_dz_80pct']:>9.2f}")
    p = g[X53]["paired"]["infonce"]
    a(f"\n  differences are UNSEEN minus SEEN; positive InfoNCE difference = unseen fits worse.")
    a(f"  headline (InfoNCE): dz = {p['cohens_dz']:+.2f}, Wilcoxon p = {p['p_wilcoxon']:.3f}")
    a(f"  the design could detect dz >= {p['mde_dz_80pct']:.2f} at 80% power "
      f"(= {p['mde_raw_units']:.4f} InfoNCE units); observed effect is "
      f"{'LARGER' if abs(p['cohens_dz'])>p['mde_dz_80pct'] else 'SMALLER'} than that.")

    a("\n--- 2. SECONDARY: unpaired 9 vs 9 (XLSR-53) ---")
    a(f"{'metric':<18}{'SEEN':>10}{'UNSEEN':>10}{'diff':>10}{'d':>8}{'Welch p':>10}{'MWU p':>10}{'MDE d':>8}")
    for m, disp, _ in A.METRICS:
        u = g[X53]["unpaired"][m]
        a(f"{disp:<18}{u['mean_seen']:>10.4f}{u['mean_unseen']:>10.4f}{u['mean_diff']:>+10.4f}"
          f"{u['cohens_d']:>+8.2f}{u['p_welch']:>10.4f}{u['p_mannwhitney']:>10.4f}"
          f"{u['mde_d_80pct']:>8.2f}")

    a("\n--- 3. UNSEEN split by proximity to a pre-trained relative (XLSR-53) ---")
    a("  reported separately and NOT collapsed: if the isolated group fares worse, the")
    a("  finding is that phonological proximity - not list membership - predicts fit.")
    for m in ("infonce", "pos_neg_gap"):
        disp = dict((x, y) for x, y, _ in A.METRICS)[m]
        a(f"  {disp}:")
        for rating in ("CLOSE", "DISTANT"):
            r = g[X53]["proximity"].get(m, {}).get(rating)
            if not r: continue
            a(f"     SEEN(n={r['n1']}) vs UNSEEN-{rating}(n={r['n2']}): "
              f"{r['mean_seen']:.4f} vs {r['mean_unseen']:.4f}  "
              f"diff={r['mean_diff']:+.4f}  d={r['cohens_d']:+.2f}  "
              f"Welch p={r['p_welch']:.3f}  MWU p={r['p_mannwhitney']:.3f}  "
              f"(MDE d>={r['mde_d_80pct']:.2f})")

    a("\n--- 4. NEGATIVE CONTROL: same split applied to XLS-R 0.3B ---")
    a("  almost every selected language is in XLS-R's 128, so the split should do nothing.")
    for key, lab in [(X300, "all 18 languages"), (X300 + "_noXhosa", "excluding Xhosa")]:
        if key not in g: continue
        p = g[key]["paired"]["infonce"]; u = g[key]["unpaired"]["infonce"]
        a(f"     {lab:<20} paired dz={p['cohens_dz']:+.2f} (p={p['p_wilcoxon']:.3f}), "
          f"unpaired d={u['cohens_d']:+.2f} (p={u['p_welch']:.3f})")
    if X53 + "_noXhosa" in g:
        p = g[X53 + "_noXhosa"]["paired"]["infonce"]
        a(f"     [XLSR-53 excluding Xhosa]  paired dz={p['cohens_dz']:+.2f} "
          f"(p={p['p_wilcoxon']:.3f})")

    a("\n--- 5. DOSE-RESPONSE ---")
    d = st["dose_response"]
    for k in ("xlsr300m_hours_vs_infonce", "xlsr300m_hours_vs_pos_neg_gap"):
        if k not in d: continue
        r = d[k]
        a(f"  {k}: n={r['n']}  rho={r['spearman_rho']:+.3f} (p={r['p']:.3f}), "
          f"log10 hours rho={r['spearman_rho_log10h']:+.3f} (p={r['p_log10h']:.3f})")
        a(f"     hours span {r['hours_min']:.0f}-{r['hours_max']:.0f}h; expected: {r['hypothesis']}")
    if "xlsr53_tier_vs_infonce" in d:
        r = d["xlsr53_tier_vs_infonce"]
        a(f"  XLSR-53 corpus tier vs InfoNCE: n={r['n']} rho={r['spearman_rho']:+.3f} "
          f"(p={r['p']:.3f})  [{r['note']}]")

    a("\n--- 6. IS FAMILY A BETTER PREDICTOR THAN MEMBERSHIP? (eta-squared on language means) ---")
    for ck in (X53, X300):
        v = st["variance_explained"][ck]
        a(f"  {ck.split('/')[-1]}:")
        for m, disp, _ in A.METRICS:
            a(f"     {disp:<18} group eta2={v[m]['group']:.3f} (p={v[m]['anova_group_p']:.3f})   "
              f"family eta2={v[m]['family']:.3f} (p={v[m]['anova_family_p']:.3f})")

    txt = "\n".join(L)
    print(txt)
    open(f"{RES}/experiment_b_report.txt", "w").write(txt)


if __name__ == "__main__":
    main()
