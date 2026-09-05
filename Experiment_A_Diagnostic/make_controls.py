#!/usr/bin/env python
"""Positive controls: audio that is knowingly out-of-distribution for a speech model.

A null language result only means something if the measurement can move at all.
These conditions establish the dynamic range of the residual / entropy metrics,
so "Sinhala ~= English" can be read as a real null rather than a dead instrument.
All are derived from the SAME English Common Voice clips, so content and channel
are held constant and only the manipulation differs.
"""
import os, json, numpy as np, soundfile as sf, torch, torchaudio

ROOT  = os.path.expanduser("~/EchoVerge-LABS/Experiment_A_Diagnostic")
STAGE = f"{ROOT}/data/staged"
SR, N = 16000, 300

def load16k(p):
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    t = torch.from_numpy(x.mean(1))
    if sr != SR:
        t = torchaudio.functional.resample(t, sr, SR)
    return t[: 30 * SR]

def main():
    man = json.load(open(f"{ROOT}/results/staging_manifest.json"))
    src = [r["path"] for r in man["English_CV"]][:N]
    rng = np.random.default_rng(7)
    conds = ["Ctl_WhiteNoise", "Ctl_Reversed", "Ctl_SpeedPitch", "Ctl_Babble", "Ctl_Tones"]
    for c in conds:
        os.makedirs(f"{STAGE}/{c}", exist_ok=True)
    new = {c: [] for c in conds}

    for i, p in enumerate(src):
        w = load16k(p)
        n = w.numel()
        rms = w.pow(2).mean().sqrt().clamp_min(1e-6)
        out = {}
        # 1. white gaussian noise, matched in level - maximally out of distribution
        out["Ctl_WhiteNoise"] = torch.randn(n) * rms
        # 2. time-reversed speech - speech-like spectrum, impossible temporal dynamics
        out["Ctl_Reversed"] = torch.flip(w, [0])
        # 3. 1.45x speed/pitch shift - real speech pushed off its normal manifold
        out["Ctl_SpeedPitch"] = torchaudio.functional.resample(w, SR, int(SR * 1.45))
        # 4. six-way babble - dense overlapping speech, no single clean speaker
        mix = w.clone()
        for j in rng.choice(len(src), 5, replace=False):
            o = load16k(src[int(j)])
            o = o.repeat(n // max(o.numel(), 1) + 1)[:n]
            mix = mix + o
        out["Ctl_Babble"] = mix / mix.pow(2).mean().sqrt().clamp_min(1e-6) * rms
        # 5. random sine chirps - structured non-speech
        t = torch.arange(n) / SR
        tone = sum(torch.sin(2 * np.pi * (f + 300 * t) * t)
                   for f in rng.uniform(120, 3500, 4))
        out["Ctl_Tones"] = (tone / tone.abs().max().clamp_min(1e-6)) * rms

        for c, y in out.items():
            y = torch.nan_to_num(y).clamp(-1, 1)
            dst = f"{STAGE}/{c}/{i:04d}_ctl.wav"
            sf.write(dst, y.numpy(), SR)
            new[c].append(dict(idx=i, src=p, path=dst, genre="control",
                               bytes=os.path.getsize(dst)))
    man.update(new)
    json.dump(man, open(f"{ROOT}/results/staging_manifest.json", "w"), indent=1)
    print("added control corpora:", {k: len(v) for k, v in new.items()})

if __name__ == "__main__":
    main()
