import json, sys, os
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse_b as A
from selection import PAIRS, META, PROXIMITY_SENSITIVITY, XLSR300_UNSEEN
RES="results"; CK={"xlsr53":"facebook/wav2vec2-large-xlsr-53","xlsr300m":"facebook/wav2vec2-xls-r-300m"}

def extras(tag):
    A.NAME.update({r["config"]:r["name"] for r in json.load(open(f"{RES}/fleurs_languages.json"))})
    df=pd.read_parquet(f"{RES}/per_file_metrics{tag}.parquet")
    res={}; lm={}
    for ck,full in CK.items():
        l,r=A.report_checkpoint(df,ck,res); lm[full]=l; res[full]=res.pop(ck)
    lm53,lm30=lm[CK["xlsr53"]],lm[CK["xlsr300m"]]

    print("\n--- PROXIMITY SUBGROUPS (XLSR-53, InfoNCE) ---")
    seen=[c for c in lm53.index if lm53.loc[c,"group"]=="SEEN"]
    sv=lm53.loc[seen,"infonce"].values
    print(f"  SEEN (n=9)  mean {sv.mean():.4f}  sd {sv.std(ddof=1):.4f}  "
          f"range [{sv.min():.4f}, {sv.max():.4f}]")
    for rating in ("CLOSE","DISTANT"):
        sel=[c for c in lm53.index if META[c].get("proximity")==rating]
        v=lm53.loc[sel,"infonce"].values
        print(f"\n  UNSEEN-{rating} (n={len(sel)})  mean {v.mean():.4f}  diff vs SEEN {v.mean()-sv.mean():+.4f}")
        for c in sorted(sel,key=lambda x: lm53.loc[x,"infonce"]):
            z=(lm53.loc[c,"infonce"]-sv.mean())/sv.std(ddof=1)
            pct=100*(sv<lm53.loc[c,"infonce"]).mean()
            print(f"      {lm53.loc[c,'name']:<12}{lm53.loc[c,'infonce']:>8.4f}   z vs SEEN {z:+.2f}   "
                  f"above {pct:.0f}% of SEEN languages")
        if rating=="CLOSE":
            t=A.unpaired_test(sv,v,"SEEN vs CLOSE")
            print(f"      test: d={t['cohens_d']:+.2f} Welch p={t['p_welch']:.4f} "
                  f"MWU p={t['p_mannwhitney']:.4f} MDE d={t['mde_d_80pct']:.2f}")
        else:
            print(f"      DESCRIPTIVE ONLY - n=4, MDE d>={A.mde_unpaired(9,4):.2f}; exploratory and underpowered")

    print("\n--- Burmese-as-CLOSE sensitivity ---")
    for lab,ov in [("as registered",{}),("Burmese moved to CLOSE",PROXIMITY_SENSITIVITY)]:
        rate={c:ov.get(c,META[c].get("proximity")) for c in lm53.index if META[c].get("proximity")}
        s=", ".join(f"{r}: n={sum(1 for v in rate.values() if v==r)}, "
                    f"mean={lm53.loc[[c for c,v in rate.items() if v==r],'infonce'].mean():.4f}"
                    for r in ("CLOSE","DISTANT"))
        print(f"  {lab:<24} {s}")

    print("\n--- NEGATIVE CONTROL: same split on XLS-R 300m (almost everything seen) ---")
    for drop,lab in [((),"all 18"),(tuple(XLSR300_UNSEEN),"excl. Xhosa")]:
        r2={}
        l2,rr=A.report_checkpoint(df,"xlsr300m",r2,drop=drop)
        p=rr["paired"]["infonce"]; u=rr["unpaired"]["infonce"]
        print(f"  {lab:<12} paired: n={p['n_pairs']} diff={p['mean_diff']:+.4f} "
              f"CI[{p['ci_lo']:+.4f},{p['ci_hi']:+.4f}] dz={p['cohens_dz']:+.2f} "
              f"Wilcoxon p={p['p_wilcoxon']:.4f} | unpaired d={u['cohens_d']:+.2f} p={u['p_welch']:.4f}")

    print("\n--- DOSE-RESPONSE ---")
    d=A.dose_response(df,lm)
    for k in ("xlsr300m_hours_vs_infonce","xlsr300m_hours_vs_pos_neg_gap"):
        v=d[k]
        print(f"  {k}: n={v['n']} rho={v['spearman_rho']:+.3f} p={v['p']:.4f}  "
              f"(log10 hours rho={v['spearman_rho_log10h']:+.3f} p={v['p_log10h']:.4f})")
        print(f"     hours span {v['hours_min']:.0f}-{v['hours_max']:.0f}h; hypothesis: {v['hypothesis']}")
    if "xlsr53_tier_vs_infonce" in d:
        v=d["xlsr53_tier_vs_infonce"]
        print(f"  XLSR-53 ordinal corpus tier (SEEN only): n={v['n']} rho={v['spearman_rho']:+.3f} p={v['p']:.4f}")
        for r in sorted(v["table"],key=lambda r:-r["tier"]):
            print(f"     {r['language']:<12} tier {r['tier']} ({r['corpus']:<18}) InfoNCE {r['infonce']:.4f}")

    print("\n--- VARIANCE EXPLAINED (XLSR-53 language means) ---")
    lmx=lm53[lm53.group!="REFERENCE"]
    for m in ("infonce","pos_neg_gap"):
        v=A.variance_explained(lmx,m)
        print(f"  {m:<14} eta2(group)={v['group']:.4f} (ANOVA p={v['anova_group_p']:.4f})   "
              f"eta2(family)={v['family']:.4f} (p={v['anova_family_p']:.4f})")
    return lm,res,d

if __name__=="__main__":
    extras(sys.argv[1])
