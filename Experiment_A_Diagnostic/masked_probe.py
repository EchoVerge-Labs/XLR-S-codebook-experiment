#!/usr/bin/env python
"""
Masked contrastive probe - the diagnostic that matches how the codebook is actually used.

wav2vec2's objective only ever asks project_hid(c_t) to align with project_q(q_t) at
MASKED positions: given surrounding context, identify this frame's quantized target
among distractors.  Measured without masking, the two projections are near-orthogonal
for every input (including white noise), which is why the unmasked residual has almost
no dynamic range.

Here we reproduce the pretraining condition (mask_prob .065, span 10, 100 negatives)
and report, at masked positions:
    contrastive accuracy  - is the true code the most similar of 1 + 100 candidates?
    InfoNCE loss          - the pretraining loss itself
    pos - mean(neg) sim   - how far the true code stands above the distractors
Higher loss / lower accuracy = the codebook serves this speech less well.
"""
import os, json, argparse, warnings
import numpy as np, torch, soundfile as sf, torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices, _sample_negative_indices)

warnings.filterwarnings("ignore")
ROOT = os.path.expanduser("~/EchoVerge-LABS/Experiment_A_Diagnostic")
MODEL = os.environ.get("XLSR_MODEL", "facebook/wav2vec2-xls-r-300m")  # XLS-R 0.3B
RES, SR = os.environ.get("XLSR_RESULTS", f"{ROOT}/results"), 16000
MASK_PROB, MASK_SPAN, N_NEG, TEMP, MAX_S = 0.065, 10, 100, 0.1, 30.0


def load16k(p):
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    t = torch.from_numpy(x.mean(1))
    if sr != SR:
        t = torchaudio.functional.resample(t, sr, SR)
    return t[: int(MAX_S * SR)]


@torch.no_grad()
def probe_file(model, fe, path, dev, seed):
    w = load16k(path)
    if w.numel() < SR // 2:
        raise ValueError("too short")
    iv = fe(w.numpy(), sampling_rate=SR, return_tensors="pt").input_values.to(dev)
    T = int(model._get_feat_extract_output_lengths(iv.shape[-1]))
    if T < MASK_SPAN * 3:
        raise ValueError("too few frames")
    np.random.seed(seed)
    mask = _compute_mask_indices((1, T), MASK_PROB, MASK_SPAN, min_masks=2)
    if mask.sum() < 2:
        raise ValueError("empty mask")
    neg = _sample_negative_indices((1, T), N_NEG, mask)
    mask_t = torch.tensor(mask, dtype=torch.bool, device=dev)
    neg_t = torch.tensor(neg, dtype=torch.long, device=dev)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(iv, mask_time_indices=mask_t)
    hid = out.projected_states.float()                 # (1,T,768) project_hid(context)
    qp = out.projected_quantized_states.float()        # (1,T,768) project_q(q)

    # (n_neg, 1, T, D) gathered negatives, exactly as the pretraining loss does
    negs = qp.view(-1, qp.size(-1))[neg_t.view(-1)].view(1, T, N_NEG, -1).permute(2, 0, 1, 3)
    logits = Wav2Vec2ForPreTraining.compute_contrastive_logits(
        qp.unsqueeze(0), negs, hid, TEMP)            # (1+n_neg, 1, T)
    logits = logits[:, 0, :][:, mask_t[0]]            # (1+n_neg, n_masked)
    tgt = torch.zeros(logits.shape[1], dtype=torch.long, device=dev)
    nce = torch.nn.functional.cross_entropy(logits.t().float(), tgt, reduction="none")
    # strict win: when the codebook collapses, negatives duplicate the positive and
    # argmax ties resolve to index 0, which would fake a high accuracy. Require the
    # positive to beat every distractor outright, and track the tie rate separately.
    acc = (logits[0] > logits[1:].max(0).values).float()
    tie = (logits[0] <= logits[1:].max(0).values).float() * \
          (logits[0] >= logits[1:].max(0).values).float()
    sims = logits * TEMP                              # back to cosine similarities
    pos_gap = sims[0] - sims[1:].mean(0)
    return dict(nce=nce.cpu().numpy(), acc=acc.cpu().numpy(), tie=tie.cpu().numpy(),
                pos_gap=pos_gap.cpu().numpy(), n_masked=int(mask_t.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=3, help="mask draws per file")
    a = ap.parse_args()

    man = json.load(open(f"{RES}/staging_manifest.json"))
    corpora = a.corpora or sorted(man.keys())
    dev = "cuda"
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)
    model = Wav2Vec2ForPreTraining.from_pretrained(MODEL).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    summary = {}
    for c in corpora:
        recs = man[c][: a.limit] if a.limit else man[c]
        nce, acc, gap, tie, per_file, fails = [], [], [], [], [], 0
        for i, r in enumerate(recs):
            fn, fa, fg, ft = [], [], [], []
            for k in range(a.repeats):
                try:
                    o = probe_file(model, fe, r["path"], dev, seed=1000 * i + k)
                except Exception:
                    fails += 1; continue
                fn.append(o["nce"]); fa.append(o["acc"]); fg.append(o["pos_gap"]); ft.append(o["tie"])
            if not fn:
                continue
            fn, fa, fg, ft = (np.concatenate(fn), np.concatenate(fa),
                              np.concatenate(fg), np.concatenate(ft))
            nce.append(fn); acc.append(fa); gap.append(fg); tie.append(ft)
            per_file.append(dict(path=r["path"], genre=r.get("genre"), src=r.get("src"),
                                 nce=float(fn.mean()), acc=float(fa.mean()),
                                 pos_gap=float(fg.mean())))
        nce, acc, gap, tie = (np.concatenate(nce), np.concatenate(acc),
                              np.concatenate(gap), np.concatenate(tie))
        summary[c] = dict(n_files=len(per_file), n_masked_frames=int(len(nce)), fails=fails,
                          nce_mean=float(nce.mean()), nce_median=float(np.median(nce)),
                          nce_p95=float(np.quantile(nce, .95)),
                          acc_mean=float(acc.mean()), tie_rate=float(tie.mean()),
                          pos_gap_mean=float(gap.mean()))
        np.savez_compressed(f"{RES}/masked_{c}.npz", nce=nce, acc=acc, pos_gap=gap, tie=tie)
        json.dump(per_file, open(f"{RES}/masked_perfile_{c}.json", "w"))
        print(f"  {c:<16} files={len(per_file):>4} masked_frames={len(nce):>7,} "
              f"acc={acc.mean()*100:5.1f}% tie={tie.mean()*100:4.1f}% "
              f"NCE={nce.mean():.4f}  pos-neg gap={gap.mean():+.4f}",
              flush=True)
    json.dump(summary, open(f"{RES}/masked_summary.json", "w"), indent=1)
    print(f"\n-> {RES}/masked_summary.json")


if __name__ == "__main__":
    main()
