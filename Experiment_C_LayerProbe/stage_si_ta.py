#!/usr/bin/env python
"""Stage Sinhala (OpenSLR-52) and Tamil (OpenSLR-65) locally from the Drive mount.

Selects the densest speakers first, then copies enough utterances per speaker to cover
the per-language budget with headroom. Exact durations are measured after staging;
the final trim to exactly 3 h / 0.5 h happens in the split builder.
"""
import os, csv, glob, json, shutil, collections, sys
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf

CD = os.path.expanduser("~/google-drive/Community Datasets")
ROOT = os.path.dirname(os.path.abspath(__file__))
N_SPK, HOURS_TARGET = 40, 4.6      # headroom over the 3.5 h budget


def sinhala_index():
    tsv = f"{CD}/OpenSLR52_Sinhala/audio/asr_sinhala/utt_spk_text.tsv"
    per = collections.defaultdict(list)
    for line in open(tsv, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 3:
            per[f[1]].append((f[0], f[2]))
    return per


def tamil_index():
    per = collections.defaultdict(list)
    for sub in ("female", "male"):
        idx = f"{CD}/OpenSLR65_Tamil/audio/{sub}/line_index.tsv"
        for line in open(idx, encoding="utf-8"):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2:
                per[f[0].split("_")[1]].append((f[0], f[1], sub))
    return per


def stage(lang, items, srcfn):
    """items: {speaker: [(utt_id, text, ...)]}"""
    out = f"{ROOT}/data/{lang}"
    os.makedirs(out, exist_ok=True)
    order = sorted(items, key=lambda s: -len(items[s]))[:N_SPK]
    per_spk_sec = HOURS_TARGET * 3600 / len(order)
    recs, fails = [], []

    def do(spk):
        d = f"{out}/{spk}"; os.makedirs(d, exist_ok=True)
        got, sec = [], 0.0
        for it in items[spk]:
            if sec >= per_spk_sec:
                break
            src = srcfn(it)
            dst = f"{d}/{it[0]}{os.path.splitext(src)[1]}"
            try:
                if not os.path.exists(dst) or os.path.getsize(dst) == 0:
                    shutil.copyfile(src, dst)
                info = sf.info(dst)
                got.append(dict(speaker=spk, utt=it[0], path=dst, text=it[1],
                                duration=info.duration, sr=info.samplerate))
                sec += info.duration
            except Exception as e:
                fails.append(dict(spk=spk, utt=it[0], err=repr(e)[:120]))
        return got

    with ThreadPoolExecutor(max_workers=16) as ex:
        for g in ex.map(do, order):
            recs += g
    tot = sum(r["duration"] for r in recs)
    print(f"{lang}: {len(recs)} utts, {len(order)} speakers, {tot/3600:.2f} h, "
          f"mean clip {tot/max(len(recs),1):.2f}s, {len(fails)} failures", flush=True)
    for f in fails[:5]:
        print("   FAIL", f, flush=True)
    json.dump(dict(language=lang, n_speakers=len(order), n_utts=len(recs),
                   hours=tot/3600, failures=fails, records=recs),
              open(f"{ROOT}/data/{lang}_staged.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "sinhala"):
        si = sinhala_index()
        stage("sinhala", si,
              lambda it: f"{CD}/OpenSLR52_Sinhala/audio/asr_sinhala/data/{it[0][:2]}/{it[0]}.flac")
    if which in ("both", "tamil"):
        ta = tamil_index()
        stage("tamil", ta,
              lambda it: f"{CD}/OpenSLR65_Tamil/audio/{it[2]}/{it[0]}.wav")
