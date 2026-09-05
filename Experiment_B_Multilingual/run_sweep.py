#!/usr/bin/env python
"""Experiment B sweep - the Experiment A instrument applied across languages and checkpoints.

The measurement functions are imported from Experiment A rather than reimplemented, so the
numbers are directly comparable with that experiment:
    stage_a_diagnostic.Diagnostic.run_file  -> frame metrics (residual, entropy, codes, rms)
    masked_probe.probe_file                 -> masked contrastive metrics (InfoNCE, gap)
masked_probe's module constants supply the validated protocol unchanged:
    MASK_PROB=0.065, MASK_SPAN=10, N_NEG=100, TEMP=0.1
Experiment A selects its checkpoint through a module-level constant, so that constant is
rebound before each Diagnostic is constructed; nothing else is altered.
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd, torch

STAGE_A = os.path.expanduser("~/EchoVerge-LABS/Experiment_A_Diagnostic")
sys.path.insert(0, STAGE_A)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stage_a_diagnostic as SA
import masked_probe as MP
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForPreTraining
from selection import ALL, META, CONFIGS

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = f"{ROOT}/results"
CKPTS = {"xlsr53": "facebook/wav2vec2-large-xlsr-53",
         "xlsr300m": "facebook/wav2vec2-xls-r-300m"}
REPEATS, SPEECH_Q, SPEECH_FRAC, V, G = 3, 0.90, 0.10, 320, 2


def perplexity(codes):
    """exp(entropy) of code usage, averaged over the two quantiser groups."""
    out = []
    for g in range(G):
        cnt = np.bincount(codes[:, g], minlength=V).astype(np.float64)
        p = cnt / max(cnt.sum(), 1)
        nz = p[p > 0]
        out.append(float(np.exp(-(nz * np.log(nz)).sum())))
    return float(np.mean(out))


def run_checkpoint(tag, ckpt, manifest, limit=None):
    SA.MODEL = ckpt                       # Experiment A picks its checkpoint from this global
    diag = SA.Diagnostic(dtype=torch.bfloat16)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(ckpt)
    model = diag.model                    # same frozen eval-mode model for both probes

    rows, per_lang_codes, failures = [], {}, []
    for cfg, *_ in ALL:
        if cfg not in manifest: continue
        recs = manifest[cfg]["records"][:limit] if limit else manifest[cfg]["records"]
        codes_acc, t0 = [], time.time()
        for r in recs:
            try:
                o = diag.run_file(r["path"])
            except Exception as e:
                failures.append(dict(cfg=cfg, path=r["path"], stage="frames", err=repr(e)[:140]))
                continue
            # per-file energy gate, identical to Experiment A
            rms = o["rms"]
            m = rms > SPEECH_FRAC * np.quantile(rms, SPEECH_Q)
            if m.sum() < 10:
                m = np.ones_like(m, dtype=bool)
            codes = o["codes"][m]
            codes_acc.append(codes)

            nce, gap, acc = [], [], []
            for k in range(REPEATS):
                try:
                    p = MP.probe_file(model, fe, r["path"], "cuda",
                                      seed=1000 * r["file_id"] + k)
                except Exception as e:
                    failures.append(dict(cfg=cfg, path=r["path"], stage="masked",
                                         err=repr(e)[:140]))
                    continue
                nce.append(p["nce"]); gap.append(p["pos_gap"]); acc.append(p["acc"])
            if not nce:
                continue
            nce, gap, acc = map(np.concatenate, (nce, gap, acc))
            rows.append(dict(
                language=cfg, group=META[cfg]["group"], family=META[cfg]["family"],
                sub_branch=META[cfg]["sub_branch"], checkpoint=tag,
                file_id=r["file_id"], n_frames=int(m.sum()),
                n_frames_all=int(len(rms)), duration=r["duration"],
                infonce=float(nce.mean()), pos_neg_gap=float(gap.mean()),
                masked_acc=float(acc.mean()),
                residual=float(o["resid_cos"][m].mean()),
                entropy=float(o["entropy"][m].mean()),
                perplexity=perplexity(codes),
            ))
        per_lang_codes[cfg] = np.concatenate(codes_acc) if codes_acc else np.zeros((0, G), np.int16)
        n = sum(1 for x in rows if x["language"] == cfg)
        print(f"  [{tag}] {cfg:<13} {n:>4} files  "
              f"{per_lang_codes[cfg].shape[0]:>7,} speech frames  {time.time()-t0:>5.0f}s",
              flush=True)
    del diag, model
    torch.cuda.empty_cache()
    return rows, per_lang_codes, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ckpts", nargs="*", default=list(CKPTS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--manifest", default="fleurs_manifest.json")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="truncate every clip to this length, matching context across "
                         "languages (duration-matched sensitivity arm)")
    a = ap.parse_args()

    if a.max_seconds:
        # Experiment A truncates via these module constants; setting them equalises the
        # amount of context every masked position can draw on.
        SA.MAX_SECONDS = a.max_seconds
        MP.MAX_S = a.max_seconds
        print(f"duration-matched arm: every clip truncated to {a.max_seconds}s")
    manifest = json.load(open(f"{RES}/{a.manifest}"))
    all_rows, all_fail = [], []
    for tag in a.ckpts:
        print(f"\n=== {tag} : {CKPTS[tag]} ===")
        rows, codes, fails = run_checkpoint(tag, CKPTS[tag], manifest, a.limit)
        all_rows += rows; all_fail += fails
        np.savez_compressed(f"{RES}/codes_{tag}{a.tag}.npz", **codes)

    df = pd.DataFrame(all_rows)
    df.to_parquet(f"{RES}/per_file_metrics{a.tag}.parquet", index=False)
    df.to_csv(f"{RES}/per_file_metrics{a.tag}.csv", index=False)
    json.dump(all_fail, open(f"{RES}/sweep_failures{a.tag}.json", "w"), indent=1)
    print(f"\nwrote {len(df):,} per-file rows, {len(all_fail)} failures -> {RES}/")


if __name__ == "__main__":
    main()
