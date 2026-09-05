import numpy as np, json, pandas as pd, os
RES=os.environ.get("XLSR_RESULTS","results"); base="English_CV"
order=[base,"Sinhala_YT","Tamil_YT","Sinhala_read","Tamil_read",
       "Ctl_SpeedPitch","Ctl_Babble","Ctl_Reversed","Ctl_Tones","Ctl_WhiteNoise"]
def stats(c):
    z=np.load(f"{RES}/frames_{c}.npz"); fid,rms=z["file_id"],z["rms"]
    p90=np.zeros(fid.max()+1,dtype=np.float32)
    for f in range(fid.max()+1): p90[f]=np.quantile(rms[fid==f],0.90)
    m=rms>0.10*p90[fid]; cd=z["codes"][m]; perp=[]
    for g in range(2):
        cnt=np.bincount(cd[:,g],minlength=320).astype(float); p=cnt/cnt.sum(); nz=p[p>0]
        perp.append(np.exp(-(nz*np.log(nz)).sum()))
    return dict(corpus=c,resid=float(z["resid_cos"][m].mean()),ent=float(z["entropy"][m].mean()),
                margin=float(z["margin"][m].mean()),perp=float(np.mean(perp)))
S={c:stats(c) for c in order}; b=S[base]
mk=json.load(open(f"{RES}/masked_summary.json")) if os.path.exists(f"{RES}/masked_summary.json") else {}
print(f"{'condition':<18}{'resid':>8}{'xEN':>7}{'entropy':>9}{'xEN':>7}{'perplex':>9}{'xEN':>7}{'NCE':>8}{'xEN':>7}{'posgap':>9}{'xEN':>7}")
print("-"*98)
for c in order:
    s=S[c]; tag=" *" if c.startswith("Ctl") else ""
    n=mk.get(c,{}); nb=mk.get(base,{})
    nce=f"{n['nce_mean']:>8.3f}{n['nce_mean']/nb['nce_mean']:>7.3f}" if n else f"{'-':>8}{'-':>7}"
    pg=f"{n['pos_gap_mean']:>9.3f}{n['pos_gap_mean']/nb['pos_gap_mean']:>7.3f}" if n else f"{'-':>9}{'-':>7}"
    print(f"{c+tag:<18}{s['resid']:>8.4f}{s['resid']/b['resid']:>7.3f}{s['ent']:>9.4f}"
          f"{s['ent']/b['ent']:>7.3f}{s['perp']:>9.1f}{s['perp']/b['perp']:>7.3f}{nce}{pg}")
print("\n* = positive control (knowingly out-of-distribution audio).  NCE chance level = ln(101) = 4.615")
pd.DataFrame([{**S[c],**{f"masked_{k}":v for k,v in mk.get(c,{}).items()}} for c in order]).to_csv(f"{RES}/sensitivity_ladder.csv",index=False)
