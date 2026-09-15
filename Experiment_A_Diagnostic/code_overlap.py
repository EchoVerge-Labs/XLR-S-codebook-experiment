#!/usr/bin/env python
"""Which codes does each language actually use, and are they the same ones?

POST-HOC. Added after the pre-registered diagnostic, in response to review.

The codebook hypothesis is mechanistic: a language absent from pre-training is said to
have sounds with no nearby codevectors. Sections on InfoNCE answer that with an
aggregate fit, which leaves the direct form of the claim untested - if Sinhala and Tamil
really lacked units, they should either use a DIFFERENT region of the codebook or be
squeezed onto a SMALLER one. Both are visible in the code assignments we already stored.

Three measures per group (G=2, V=320), all on the frame-level assignments:
  support   - how many distinct codes the condition ever uses
  Jaccard   - overlap of that support with English's support
  JS        - Jensen-Shannon divergence between the code-usage distributions

Support depends on sample size, so every condition is subsampled to the same number of
gated frames and the estimate is averaged over repeated draws. Frames are gated exactly
as in the main analysis: below 10% of each file's 90th-percentile RMS is dropped, so
silence cannot drive the comparison.

The positive controls calibrate all three: they are built from the SAME English clips,
so anything they shift is a property of the corruption, not of the language.
"""
import os, json, argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
G, V = 2, 320
CONDITIONS = ["English_CV", "Sinhala_YT", "Tamil_YT", "Sinhala_read", "Tamil_read",
              "Ctl_SpeedPitch", "Ctl_Babble", "Ctl_Reversed", "Ctl_Tones", "Ctl_WhiteNoise"]
REFERENCE = "English_CV"


def load_gated(resdir, cond):
    d = np.load(f"{resdir}/frames_{cond}.npz", allow_pickle=True)
    codes, rms, fid = d["codes"], d["rms"], d["file_id"]
    keep = np.zeros(len(rms), bool)
    for f in np.unique(fid):
        m = fid == f
        keep[m] = rms[m] >= 0.10 * np.percentile(rms[m], 90)
    return codes[keep]


def js_divergence(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: float((a[a > 0] * np.log(a[a > 0] / b[a > 0])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=f"{ROOT}/results")
    ap.add_argument("--frames", type=int, default=30000, help="matched frames per condition")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    X = {c: load_gated(a.results, c) for c in CONDITIONS
         if os.path.exists(f"{a.results}/frames_{c}.npz")}
    smallest = min(len(v) for v in X.values())
    if a.frames > smallest:
        raise SystemExit(f"--frames {a.frames} exceeds the smallest condition ({smallest})")

    rng = np.random.default_rng(0)
    acc = {c: {k: [] for k in ("support", "jaccard", "mass_in_ref", "js")} for c in X}
    for _ in range(a.reps):
        hist = {}
        for c, v in X.items():
            s = v[rng.choice(len(v), a.frames, replace=False)]
            hist[c] = np.stack([np.bincount(s[:, g], minlength=V) for g in range(G)])
        ref = hist[REFERENCE]
        ref_support = [set(np.where(ref[g] > 0)[0]) for g in range(G)]
        ref_p = ref / ref.sum(1, keepdims=True)
        for c, h in hist.items():
            sup = [set(np.where(h[g] > 0)[0]) for g in range(G)]
            p = h / h.sum(1, keepdims=True)
            acc[c]["support"].append(np.mean([len(s) for s in sup]))
            acc[c]["jaccard"].append(np.mean(
                [len(sup[g] & ref_support[g]) / len(sup[g] | ref_support[g]) for g in range(G)]))
            acc[c]["mass_in_ref"].append(np.mean(
                [h[g][list(ref_support[g])].sum() / h[g].sum() for g in range(G)]))
            acc[c]["js"].append(np.mean([js_divergence(p[g], ref_p[g]) for g in range(G)]))

    out = {"config": dict(frames_per_condition=a.frames, reps=a.reps, reference=REFERENCE,
                          groups=G, codes_per_group=V,
                          gated_frames={c: int(len(v)) for c, v in X.items()}),
           "conditions": {c: {k: dict(mean=float(np.mean(v)), sd=float(np.std(v)))
                              for k, v in d.items()} for c, d in acc.items()}}
    path = a.out or f"{a.results}/code_overlap.json"
    json.dump(out, open(path, "w"), indent=1)

    print(f"matched at {a.frames:,} gated frames, {a.reps} draws, reference {REFERENCE}\n")
    print(f"{'condition':17s}{'support/320':>14}{'Jaccard':>16}{'mass in ref':>13}{'JS':>16}")
    for c in CONDITIONS:
        if c not in acc:
            continue
        r = {k: (np.mean(v), np.std(v)) for k, v in acc[c].items()}
        print(f"{c:17s}{r['support'][0]:8.1f}+-{r['support'][1]:<5.1f}"
              f"{r['jaccard'][0]:10.3f}+-{r['jaccard'][1]:<5.3f}"
              f"{r['mass_in_ref'][0]:13.4f}{r['js'][0]:11.4f}+-{r['js'][1]:<5.4f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
