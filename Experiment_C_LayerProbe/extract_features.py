#!/usr/bin/env python
"""Extract all 24 transformer layers once and cache to disk as fp16 memmaps.

Layout per (language, split):
    cache/{tag}/{lang}/{split}/layer_{NN}.npy   (total_frames, 1024) fp16 - frame features
    cache/{tag}/{lang}/{split}/pooled_{NN}.npy  (n_utts, 1024)       fp16 - mean-pooled
    cache/{tag}/{lang}/{split}/index.json       utt -> [offset, length]

Frame features feed the CTC probe; the pooled arrays make the speaker probe trivial to
load. Extraction happens exactly once - the probes never touch the audio or the model.
"""
import os, json, sys, time, argparse
import numpy as np, torch, soundfile as sf, torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2Config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CHECKPOINT_PRIMARY, N_LAYERS, SR, MAX_CLIP_SECONDS, LANGUAGES

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_audio(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    t = torch.from_numpy(x.mean(axis=1))
    if sr != SR:
        t = torchaudio.functional.resample(t, sr, SR)
    return t[: int(MAX_CLIP_SECONDS * SR)]


class Extractor:
    def __init__(self, ckpt, random_init=False, device="cuda"):
        self.device = device
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(ckpt)
        if random_init:
            cfg = Wav2Vec2Config.from_pretrained(ckpt)
            torch.manual_seed(0)
            self.model = Wav2Vec2Model(cfg)          # architecture only, no learned weights
        else:
            self.model = Wav2Vec2Model.from_pretrained(ckpt)
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def layers(self, wave):
        iv = self.fe(wave.numpy(), sampling_rate=SR, return_tensors="pt").input_values.to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(iv, output_hidden_states=True)
        # hidden_states[0] is the feature-projection output; [1..24] are the transformer layers
        hs = out.hidden_states[1:N_LAYERS + 1]
        return [h[0].float().cpu().numpy().astype(np.float16) for h in hs]


def run_split(ex, lang, split, records, outdir, log):
    os.makedirs(outdir, exist_ok=True)
    if os.path.exists(f"{outdir}/index.json"):
        print(f"    {lang}/{split}: cached, skipping", flush=True)
        return
    # pass 1: frame counts (cheap - derived from the conv stack, no forward pass)
    lens, keep, fails = [], [], []
    for r in records:
        try:
            w = load_audio(r["path"])
            n = int(ex.model._get_feat_extract_output_lengths(w.numel()))
            if n < 4:
                raise ValueError(f"too short after conv stack ({n} frames)")
            lens.append(n); keep.append(r)
        except Exception as e:
            fails.append(dict(path=r["path"], utt=r.get("utt"), error=repr(e)[:160]))
    total = int(sum(lens))
    mm = [np.lib.format.open_memmap(f"{outdir}/layer_{i:02d}.npy", mode="w+",
                                    dtype=np.float16, shape=(total, ex.dim))
          for i in range(N_LAYERS)]
    pooled = [np.zeros((len(keep), ex.dim), np.float16) for _ in range(N_LAYERS)]
    index, off, t0 = {}, 0, time.time()
    for k, r in enumerate(keep):
        w = load_audio(r["path"])
        hs = ex.layers(w)
        n = hs[0].shape[0]
        for i in range(N_LAYERS):
            h = hs[i][:n]
            mm[i][off:off + n] = h
            pooled[i][k] = h.astype(np.float32).mean(0).astype(np.float16)
        index[r["utt"]] = [off, n]
        off += n
        if (k + 1) % 500 == 0:
            print(f"    {lang}/{split}: {k+1}/{len(keep)}  {time.time()-t0:.0f}s", flush=True)
    for i in range(N_LAYERS):
        mm[i].flush()
        np.save(f"{outdir}/pooled_{i:02d}.npy", pooled[i])
    json.dump(dict(index=index, total_frames=off, dim=ex.dim, n_utts=len(keep),
                   n_failed=len(fails)), open(f"{outdir}/index.json", "w"))
    json.dump(fails, open(f"{outdir}/failures.json", "w"), indent=1)
    log.append(dict(lang=lang, split=split, n_utts=len(keep), frames=off,
                    n_failed=len(fails), seconds=round(time.time() - t0, 1)))
    print(f"    {lang}/{split}: {len(keep)} utts, {off:,} frames, {len(fails)} FAILED, "
          f"{time.time()-t0:.0f}s", flush=True)
    for f in fails[:5]:
        print("       FAIL", f["utt"], f["error"][:90], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="xlsr300m")
    ap.add_argument("--ckpt", default=CHECKPOINT_PRIMARY)
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="cap per split (used to keep the floor control small)")
    a = ap.parse_args()

    ex = Extractor(a.ckpt, random_init=a.random_init)
    print(f"extractor: {a.ckpt} random_init={a.random_init} dim={ex.dim} "
          f"layers={N_LAYERS}", flush=True)
    log = []
    for lang in LANGUAGES:
        sp = json.load(open(f"{ROOT}/data/{lang}_split.json", encoding="utf-8"))
        # the union of both tasks' utterances; probes index into it by utt id
        pool = {r["utt"]: r for r in sp["ctc_train"] + sp["ctc_test"]}
        for split, recs in [("train", sp["ctc_train"]), ("test", sp["ctc_test"])]:
            if a.max_hours:
                cut, sec = [], 0.0
                for r in recs:
                    if sec >= a.max_hours * 3600:
                        break
                    cut.append(r); sec += r["duration"]
                recs = cut
            run_split(ex, lang, split, recs,
                      f"{ROOT}/cache/{a.tag}/{lang}/{split}", log)
    json.dump(log, open(f"{ROOT}/results/extraction_log_{a.tag}.json", "w"), indent=1)
    tf = sum(x["frames"] for x in log); nf = sum(x["n_failed"] for x in log)
    print(f"\ntotal {tf:,} frames cached, {nf} failures across all splits")


if __name__ == "__main__":
    main()
