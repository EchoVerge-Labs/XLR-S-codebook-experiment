#!/usr/bin/env python
"""
Experiment A - Quantization residual diagnostic for XLS-R (wav2vec2-large-xlsr-53).

Question: does the frozen XLSR-53 codebook represent Sinhala/Tamil speech as well
as it represents English?

--------------------------------------------------------------------------------
A note on the residual definition
--------------------------------------------------------------------------------
The brief asked for r_t = ||z_t - q_t||^2, the classic VQ residual between a
continuous feature and the codebook entry it maps to.  That quantity is not
defined for wav2vec2: the CNN output z_t is 512-d, while the quantizer output
q_t is 768-d (2 groups x 384-d codevectors).  wav2vec2's quantizer is not a
reconstructing VQ - the codevectors are free parameters in their own space,
selected by a linear projection of z, and never asked to reconstruct z.  So
there is no shared space in which z_t - q_t can be subtracted.

We therefore measure codebook fit with two well-defined families that together
answer the same question:

  (A) CONTRASTIVE-SPACE RESIDUAL - the geometry wav2vec2 pretraining actually
      optimises.  The InfoNCE objective pulls project_hid(c_t) toward
      project_q(q_t) in a shared 768-d space, using cosine similarity:
          resid_cos  = 1 - cos( project_hid(c_t), project_q(q_t) )     [headline]
          resid_l2sq = || project_hid(c_t) - project_q(q_t) ||^2
      A high value means: the code this frame was assigned sits far from where
      the model's own contextual representation of the frame lives.

  (B) z-SPACE ASSIGNMENT QUALITY - how cleanly z_t lands on a codebook entry.
      The quantizer scores z against every entry via weight_proj; the softmax
      posterior over the 320 entries of each group is the model's code-match
      distribution:
          post_entropy  = H(softmax(logits))      (high = ambiguous match)
          post_maxprob  = max softmax prob        (low  = ambiguous match)
          post_margin   = top1 logit - top2 logit (low  = ambiguous match)

Plus codebook usage statistics (perplexity, utilisation) over the whole corpus.

--------------------------------------------------------------------------------
Design controls
--------------------------------------------------------------------------------
* Two arms.  The requested arm compares in-the-wild YouTube Sinhala/Tamil against
  read Common Voice English - those differ in acoustic domain as well as language,
  which on its own can move the residual.  A second, domain-matched arm compares
  read Sinhala (OpenSLR30) and read Tamil (OpenSLR65) against the same read
  English, isolating the language effect.
* Usage perplexity and utilisation grow with the number of frames observed, so
  they are additionally reported at a frame budget matched across corpora.
* Statistics are aggregated per FILE, not per frame: frames within a file are
  strongly autocorrelated, so file means are the honest unit of analysis for
  comparing corpora.
* Frames are tagged with relative energy so results can be recomputed on
  speech-like frames only (read corpora carry more leading/trailing silence).
"""
import os, sys, json, time, argparse, warnings
import numpy as np
import torch, soundfile as sf, torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForPreTraining

warnings.filterwarnings("ignore", category=UserWarning)
ROOT    = os.path.expanduser("~/EchoVerge-LABS/Experiment_A_Diagnostic")
RESULTS = os.path.join(ROOT, "results")
MODEL   = os.environ.get("XLSR_MODEL", "facebook/wav2vec2-xls-r-300m")  # XLS-R 0.3B
RESULTS = os.environ.get("XLSR_RESULTS", RESULTS)
SR, MAX_SECONDS, STRIDE = 16000, 30.0, 320     # 320 samples/frame = 20 ms hop

ARMS = {
    "in_the_wild": ["Sinhala_YT",   "Tamil_YT",   "English_CV"],
    "read_matched": ["Sinhala_read", "Tamil_read", "English_CV"],
}


def load_audio(path):
    """Load -> mono -> float32 -> 16 kHz -> first MAX_SECONDS. Raises on bad files."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)                                     # mono
    if x.size == 0:
        raise ValueError("empty audio")
    t = torch.from_numpy(x)
    if sr != SR:
        t = torchaudio.functional.resample(t, sr, SR)
    t = t[: int(MAX_SECONDS * SR)]
    if t.numel() < SR // 2:                                # < 0.5 s is not usable
        raise ValueError(f"too short ({t.numel()/SR:.2f}s)")
    if not torch.isfinite(t).all():
        raise ValueError("non-finite samples")
    return t


def frame_rms(wave, n_frames):
    """Per-frame RMS on the (utterance-normalised) waveform, aligned to the conv hop."""
    n = n_frames * STRIDE
    w = torch.nn.functional.pad(wave, (0, max(0, n - wave.numel())))[:n]
    return w.view(n_frames, STRIDE).pow(2).mean(dim=1).sqrt()


class Diagnostic:
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        self.device, self.dtype = device, dtype
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)
        self.model = Wav2Vec2ForPreTraining.from_pretrained(MODEL).to(device).eval()
        for p in self.model.parameters():          # frozen: inference only
            p.requires_grad_(False)
        q = self.model.quantizer
        self.G, self.V = q.num_groups, q.num_vars
        print(f"  model on {device} | dtype {dtype} | codebook G={self.G} V={self.V} "
              f"| codevectors {tuple(q.codevectors.shape)}")

    @torch.no_grad()
    def run_file(self, path):
        wave = load_audio(path)
        iv = self.fe(wave.numpy(), sampling_rate=SR, return_tensors="pt").input_values
        iv = iv.to(self.device)
        m = self.model
        with torch.autocast("cuda", dtype=self.dtype, enabled=(self.dtype != torch.float32)):
            base = m.wav2vec2(iv, output_hidden_states=False, return_dict=True)
            ctx, ext = base.last_hidden_state, base.extract_features   # (1,T,1024) (1,T,512)
            hid = m.project_hid(ctx)                                   # (1,T,768)
            qfeat, _ = m.quantizer(ext)                                # eval -> hard argmax
            qp = m.project_q(qfeat)                                    # (1,T,768)
            logits = m.quantizer.weight_proj(ext)                      # (1,T,G*V)

        # metrics in fp32 regardless of the compute dtype
        hid, qp, logits = hid.float()[0], qp.float()[0], logits.float()[0]
        T = hid.shape[0]
        logits = logits.view(T, self.G, self.V)

        diff = hid - qp
        resid_l2sq = diff.pow(2).sum(-1)
        resid_cos = 1.0 - torch.nn.functional.cosine_similarity(hid, qp, dim=-1)
        resid_rel = resid_l2sq / hid.pow(2).sum(-1).clamp_min(1e-9)     # scale-free

        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()
        entropy = -(p * logp).sum(-1)                                   # (T,G) nats
        top2 = logits.topk(2, dim=-1)
        maxprob = p.max(-1).values                                      # (T,G)
        margin = top2.values[..., 0] - top2.values[..., 1]              # (T,G)
        codes = logits.argmax(-1)                                       # (T,G)

        rms = frame_rms(iv[0].detach().float().cpu(), T)
        return dict(
            resid_cos=resid_cos.cpu().numpy(), resid_l2sq=resid_l2sq.cpu().numpy(),
            resid_rel=resid_rel.cpu().numpy(), entropy=entropy.cpu().numpy(),
            maxprob=maxprob.cpu().numpy(), margin=margin.cpu().numpy(),
            codes=codes.cpu().numpy().astype(np.int16), rms=rms.numpy(),
            n_frames=T, duration=wave.numel() / SR,
        )

    def run_corpus(self, name, records, limit=None):
        recs = records[:limit] if limit else records
        acc = {k: [] for k in ["resid_cos", "resid_l2sq", "resid_rel", "entropy",
                               "maxprob", "margin", "codes", "rms", "file_id"]}
        per_file, failures = [], []
        t0 = time.time()
        for i, r in enumerate(recs):
            try:
                o = self.run_file(r["path"])
            except Exception as e:
                failures.append(dict(path=r["path"], src=r.get("src"), error=repr(e)))
                continue
            fid = len(per_file)
            for k in ["resid_cos", "resid_l2sq", "resid_rel", "entropy",
                      "maxprob", "margin", "codes", "rms"]:
                acc[k].append(o[k])
            acc["file_id"].append(np.full(o["n_frames"], fid, dtype=np.int32))
            per_file.append(dict(
                file_id=fid, path=r["path"], genre=r.get("genre"), src=r.get("src"),
                n_frames=o["n_frames"], duration=o["duration"],
                mean_resid_cos=float(o["resid_cos"].mean()),
                mean_resid_l2sq=float(o["resid_l2sq"].mean()),
                mean_resid_rel=float(o["resid_rel"].mean()),
                mean_entropy=float(o["entropy"].mean()),
                mean_maxprob=float(o["maxprob"].mean()),
                mean_margin=float(o["margin"].mean()),
            ))
            if (i + 1) % 50 == 0:
                el = time.time() - t0
                print(f"    {name}: {i+1}/{len(recs)}  {el:.0f}s  "
                      f"({el/(i+1):.2f}s/file)  fails={len(failures)}", flush=True)
        out = {k: np.concatenate(v) for k, v in acc.items() if v}
        print(f"  {name}: {len(per_file)} files, {out['resid_cos'].size:,} frames, "
              f"{len(failures)} failed, {time.time()-t0:.0f}s")
        for f in failures[:5]:
            print("     SKIPPED", os.path.basename(f["path"]), f["error"][:90])
        return out, per_file, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="files per corpus (debug)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", nargs="*", default=None, help="corpora to run (default: all in manifest)")
    a = ap.parse_args()
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32

    man = json.load(open(f"{RESULTS}/staging_manifest.json"))
    corpora = a.only or sorted(man.keys())
    print(f"corpora: {corpora}\n")

    d = Diagnostic(dtype=dtype)
    all_failures = {}
    for c in corpora:
        frames, per_file, fails = d.run_corpus(c, man[c], limit=a.limit)
        np.savez_compressed(f"{RESULTS}/frames_{c}{a.tag}.npz", **frames)
        # raw per-frame residuals, as requested
        np.save(f"{RESULTS}/residuals_{c}{a.tag}.npy", frames["resid_cos"])
        np.save(f"{RESULTS}/residuals_l2sq_{c}{a.tag}.npy", frames["resid_l2sq"])
        json.dump(per_file, open(f"{RESULTS}/perfile_{c}{a.tag}.json", "w"), indent=1)
        all_failures[c] = fails
    json.dump(all_failures, open(f"{RESULTS}/failures{a.tag}.json", "w"), indent=1)
    print(f"\nwrote per-frame arrays to {RESULTS}/")


if __name__ == "__main__":
    main()
