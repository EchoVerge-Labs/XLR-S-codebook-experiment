#!/usr/bin/env python
"""Build the CROPPED arm: every clip cut to exactly N seconds, short clips dropped.

N is the 25th percentile of the POOLED duration distribution over all staged
languages, rounded down to 0.5 s. Pooling means the window cannot depend on group
assignment. Cropped clips are written as ordinary WAVs so the sweep runs the identical
measurement code on them - only the input files differ between arms.
"""
import os, json, sys
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selection import CROP_PERCENTILE, CROP_ROUND_TO, META

ROOT = os.path.dirname(os.path.abspath(__file__))
RES, OUT, SR = f"{ROOT}/results", f"{ROOT}/data/audio_crop", 16000


def main(N_override=None, suffix=""):
    man = json.load(open(f"{RES}/fleurs_manifest.json"))
    pooled = np.array([r["duration"] for m in man.values() for r in m["records"]])
    raw = float(np.percentile(pooled, CROP_PERCENTILE))
    N = float(N_override) if N_override else np.floor(raw / CROP_ROUND_TO) * CROP_ROUND_TO
    print(f"pooled clips: {len(pooled):,}   P{CROP_PERCENTILE} = {raw:.3f}s"
          f"   ->  window N = {N:.1f}s\n")

    out_man, rows = {}, []
    for cfg, m in sorted(man.items()):
        d = f"{OUT}{suffix}/{cfg}"
        os.makedirs(d, exist_ok=True)
        recs, dropped = [], 0
        for r in m["records"]:
            if r["duration"] < N:
                dropped += 1
                continue
            x, sr = sf.read(r["path"], dtype="float32")
            assert sr == SR
            dst = f"{d}/{r['file_id']:04d}.wav"
            sf.write(dst, x[: int(round(N * SR))], SR)
            recs.append(dict(file_id=r["file_id"], path=dst, duration=float(N),
                             orig_duration=r["duration"]))
        out_man[cfg] = dict(config=cfg, n_staged=len(recs), n_dropped=dropped,
                            crop_seconds=float(N), records=recs)
        rows.append((cfg, META[cfg]["group"], len(m["records"]), len(recs), dropped,
                     100 * dropped / len(m["records"])))
    json.dump(out_man, open(f"{RES}/fleurs_manifest_crop{suffix}.json", "w"))
    json.dump(dict(window_seconds=float(N), percentile=CROP_PERCENTILE,
                   raw_percentile_seconds=raw, n_pooled_clips=int(len(pooled)),
                   per_language={c: dict(kept=k, dropped=dr) for c, _, _, k, dr, _ in rows}),
              open(f"{RES}/crop_summary{suffix}.json", "w"), indent=1)

    print(f"{'config':<13}{'group':<11}{'in':>5}{'kept':>7}{'dropped':>9}{'drop %':>9}")
    print("-" * 54)
    for c, g, n0, k, dr, pc in sorted(rows, key=lambda r: -r[5]):
        print(f"{c:<13}{g:<11}{n0:>5}{k:>7}{dr:>9}{pc:>8.1f}%")
    tot_d = sum(r[4] for r in rows); tot_n = sum(r[2] for r in rows)
    print("-" * 54)
    print(f"{'TOTAL':<24}{tot_n:>5}{sum(r[3] for r in rows):>7}{tot_d:>9}"
          f"{100*tot_d/tot_n:>8.1f}%")


if __name__ == "__main__":
    import sys
    N = float(sys.argv[1])
    main(N, suffix=f"{N:.1f}")
