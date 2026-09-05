#!/usr/bin/env python
"""Download FLEURS test parquets and extract a deterministic clip sample per language.

Each test split is a single parquet row group with the audio embedded as WAV bytes,
so partial reads are not possible - the file is fetched whole, the sampled clips are
written out, and the parquet is deleted to keep disk use bounded.
"""
import os, io, json, random, sys, subprocess
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
import soundfile as sf
from selection import ALL

ROOT = os.path.dirname(os.path.abspath(__file__))
PQ, AUD, RES = f"{ROOT}/data/parquet", f"{ROOT}/data/audio", f"{ROOT}/results"
N_CLIPS, SEED, SR = 500, 1234, 16000
URL = ("https://huggingface.co/datasets/google/fleurs/resolve/main/"
       "parquet-data/{cfg}/test-00000-of-00001.parquet")


def stage(cfg):
    out = f"{AUD}/{cfg}"
    done = f"{out}/.done"
    if os.path.exists(done):
        return cfg, json.load(open(done))
    os.makedirs(out, exist_ok=True)
    p = f"{PQ}/{cfg}.parquet"
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        # curl rather than urllib: the HF endpoint leaves stalled connections open, which
        # urlopen's timeout does not catch. --speed-limit aborts a stalled transfer and
        # -C - resumes the partial file, so a stall costs seconds instead of hanging.
        cmd = ["curl", "-sL", "--fail", "-C", "-",
               "--speed-limit", "50000", "--speed-time", "45",
               "--retry", "20", "--retry-delay", "3", "--retry-all-errors",
               "--max-time", "7200", "-A", "curl/8",
               "-o", p + ".tmp", URL.format(cfg=cfg)]
        for attempt in range(40):
            if subprocess.run(cmd).returncode == 0:
                break
            print(f"    {cfg}: stalled, resuming (attempt {attempt + 2})", flush=True)
        else:
            raise RuntimeError(f"download failed for {cfg}")
        os.replace(p + ".tmp", p)

    t = pq.ParquetFile(p).read(columns=["id", "num_samples", "audio", "language"])
    n = t.num_rows
    idx = sorted(random.Random(SEED).sample(range(n), min(N_CLIPS, n)))
    audio = t.column("audio").to_pylist()
    ids = t.column("id").to_pylist()
    recs, fails = [], []
    for k, i in enumerate(idx):
        try:
            b = audio[i]["bytes"]
            x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
            assert sr == SR, f"unexpected sr {sr}"
            dst = f"{out}/{k:04d}_{ids[i]}.wav"
            sf.write(dst, x.mean(axis=1), SR)
            recs.append(dict(file_id=k, fleurs_id=ids[i], path=dst,
                             duration=x.shape[0] / SR))
        except Exception as e:
            fails.append({"i": i, "err": repr(e)[:120]})
    meta = dict(config=cfg, n_available=n, n_staged=len(recs), n_failed=len(fails),
                hours=round(sum(r["duration"] for r in recs) / 3600, 3),
                records=recs, failures=fails)
    json.dump(meta, open(done, "w"))
    os.remove(p)
    print(f"  {cfg:<13} {len(recs):>4}/{n:<5} clips  {meta['hours']:.2f} h  fails={len(fails)}",
          flush=True)
    return cfg, meta


if __name__ == "__main__":
    os.makedirs(PQ, exist_ok=True); os.makedirs(AUD, exist_ok=True)
    cfgs = [c for c, *_ in ALL]
    manifest = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for cfg, meta in ex.map(stage, cfgs):
            manifest[cfg] = meta
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "records"}
               for k, v in manifest.items()},
              open(f"{RES}/fleurs_staging_summary.json", "w"), indent=1)
    json.dump(manifest, open(f"{RES}/fleurs_manifest.json", "w"))
    tot = sum(v["n_staged"] for v in manifest.values())
    print(f"\nstaged {tot} clips across {len(manifest)} languages, "
          f"{sum(v['hours'] for v in manifest.values()):.1f} h total")
