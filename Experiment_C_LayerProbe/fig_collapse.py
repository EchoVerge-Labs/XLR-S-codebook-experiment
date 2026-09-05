import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=json.load(open('results/collapse_rate.json'))
sw=json.load(open('results/probe_results_xlsr300m.json'))
L=[0,4,8,12,15,17,20,22]; langs=['sinhala','tamil','english']
COL={"sinhala":"#D1495B","tamil":"#EDAE49","english":"#00798C"}
NAME={"sinhala":"Sinhala","tamil":"Tamil","english":"English"}
fig,axes=plt.subplots(1,2,figsize=(14,5.6))
ax=axes[0]
for l in langs:
    y=[100*d[l][str(x)]['rate'] for x in L]
    ax.plot(L,y,"-o",color=COL[l],ms=6,lw=2,label=f"{NAME[l]} ({sum(d[l][str(x)]['n_collapsed'] for x in L)}/72)")
ax.axvspan(14.5,17.5,color="#8E2C48",alpha=.09,zorder=0)
ax.text(16,ax.get_ylim()[1]*0.93,"best-layer\nregion",ha="center",fontsize=8.5,color="#8E2C48")
ax.set_xlabel("transformer layer"); ax.set_ylabel("CTC collapse rate (% of 9 inits)")
ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=9.5)
ax.set_title("Collapse rate is concentrated by layer AND language\n"
             "Si+Ta 8.3% vs English 0.0% (p=0.010); L15/17 22.2% vs elsewhere 3.7% (p=0.002)",
             fontsize=11,weight="bold")
ax=axes[1]
for l in langs:
    a=[sw['layers'][l][str(x)]['ctc']['cer_mean'] for x in L]
    b=[d[l][str(x)]['healthy_cer_mean'] for x in L]
    ax.plot(L,np.array(b)-np.array(a),"-o",color=COL[l],ms=6,lw=2,label=NAME[l])
ax.axhline(0,color="#333",lw=1.4); ax.axvspan(14.5,17.5,color="#8E2C48",alpha=.09,zorder=0)
ax.set_xlabel("transformer layer"); ax.set_ylabel("CER difference (9 healthy draws - 3-seed sweep)")
ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=9.5)
ax.set_title("But the cleaning does not shift the curve\n"
             "differences at L15/L17 are +0.003 or less",fontsize=11,weight="bold")
fig.suptitle("CTC collapse rate as its own data series",fontsize=13,weight="bold")
fig.text(.5,.015,"Left: collapse rate measured directly from 9 independent inits per cell, judged from training loss only. "
         "Right: whether discarding collapsed runs biases the reported curve - the surviving-run CER at 9 draws against the\n"
         "3-seed sweep value. English L4 (+0.073) is ordinary seed variance, not cleaning: no English cell collapsed at all.",
         ha="center",fontsize=8.4,style="italic")
fig.tight_layout(rect=[0,.07,1,.92]); fig.savefig("results/fig5_collapse_rate.png",dpi=170)
print("wrote results/fig5_collapse_rate.png")
