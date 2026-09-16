#!/usr/bin/env python
"""Build a 6 h training split for Sinhala and English, holding everything else fixed.

POST-HOC. Added after review, not pre-registered.

The paper's limitation reads "whether the representation becomes limiting at larger
budgets is untested". This builds the second budget point that tests it.

What stays identical to the 3 h experiment, so the two points are comparable:
  - the test set, reused verbatim (same utterances, same 10 held-out speakers)
  - the 30 training speakers, so "more data" is not confounded with "more speakers"
  - the character vocabulary, so the output layer is unchanged
  - the text normalisation and the 20 s clip guard

What changes: each training speaker contributes more of their utterances. OpenSLR-52
holds a median of 394 utterances per speaker and the 3 h split used 54-117 of them, so
the extra audio comes from speakers the model already trains on.

Tamil is absent by necessity, not choice: OpenSLR-65 is 4,291 utterances from 50
speakers in total, and its 30 training speakers hold 3.48 h. A 6 h Tamil arm does not
exist in the corpus.
"""
import os, sys, json, csv, random, collections, subprocess
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_splits import norm_text
from config import MAX_CLIP_SECONDS

ROOT = os.path.dirname(os.path.abspath(__file__))
SLR52 = os.path.expanduser("~/google-drive/Community Datasets/OpenSLR52_Sinhala")
TARGET_HOURS = 6.0
SEED = 1234
csv.field_size_limit(10 ** 7)


def stage_sinhala(train_speakers, have_utts, want_utts):
    """Copy more utterances for speakers we already train on. The Drive mount is
    latency-bound, not bandwidth-bound, so copies run in parallel."""
    rows = []
    with open(f"{SLR52}/metadata/utt_spk_text.tsv", encoding="utf-8") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and p[1] in train_speakers and p[0] not in have_utts:
                rows.append((p[0], p[1], p[2]))
    rng = random.Random(SEED)
    rng.shuffle(rows)
    rows = rows[:want_utts]
    out = f"{ROOT}/data/sinhala_6h"
    os.makedirs(out, exist_ok=True)
    todo = [(u, f"{SLR52}/audio/asr_sinhala/data/{u[:2]}/{u}.flac", f"{out}/{u}.flac")
            for u, _, _ in rows if not os.path.exists(f"{out}/{u}.flac")]
    print(f"  staging {len(todo)} new Sinhala clips in parallel ...", flush=True)
    # null-delimited: the source path contains a space, which whitespace-splitting
    # xargs would tear in half
    with open("/tmp/_stage_list", "wb") as fh:
        for _, src, dst in todo:
            fh.write(src.encode() + b"\0" + dst.encode() + b"\0")
    subprocess.run("xargs -0 -n2 -P 24 cp < /tmp/_stage_list", shell=True, check=False)
    recs = []
    for u, spk, text in rows:
        p = f"{out}/{u}.flac"
        t = norm_text(text)
        if not t or not os.path.exists(p):
            continue
        try:
            d = sf.info(p).duration
        except Exception:
            continue
        recs.append(dict(speaker=spk, utt=u, path=p, text=t, duration=d))
    return recs


def more_english(train_speakers, have_utts):
    meta = {}
    with open(f"{ROOT}/data/cv_en_train.tsv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            meta[r["path"]] = r["sentence"]
    recs = []
    base = f"{ROOT}/data/english_cv"
    for spk in train_speakers:
        d = os.path.join(base, spk)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".mp3") or f[:-4] in have_utts:
                continue
            txt = meta.get(f)
            if not txt:
                continue
            p = os.path.join(d, f)
            t = norm_text(txt)
            if not t:
                continue
            try:
                dur = sf.info(p).duration
            except Exception:
                continue
            recs.append(dict(speaker=spk, utt=f[:-4], path=p, text=t, duration=dur))
    return recs


def build(lang, extra):
    sp = json.load(open(f"{ROOT}/data/{lang}_split.json", encoding="utf-8"))
    rng = random.Random(SEED)
    pool = [r for r in sp["ctc_train"]] + [r for r in extra if r["duration"] <= MAX_CLIP_SECONDS]
    rng.shuffle(pool)
    train, sec = [], 0.0
    for r in pool:
        if sec >= TARGET_HOURS * 3600:
            break
        train.append(r); sec += r["duration"]
    out = dict(sp)                      # same vocab, same speakers, same test set
    out["ctc_train"] = train
    out["stats"] = dict(sp["stats"], ctc_train=dict(
        n=len(train), hours=sec / 3600, speakers=len(sp["ctc_train_speakers"]),
        mean_clip=sec / max(len(train), 1)))
    json.dump(out, open(f"{ROOT}/data/{lang}_split6h.json", "w"), ensure_ascii=False)
    print(f"{lang}: {len(train)} utts, {sec/3600:.2f} h train "
          f"({len(sp['ctc_train_speakers'])} speakers), test reused "
          f"{len(sp['ctc_test'])} utts, vocab {len(sp['vocab'])}", flush=True)
    return sec / 3600


if __name__ == "__main__":
    si = json.load(open(f"{ROOT}/data/sinhala_split.json", encoding="utf-8"))
    have = {r["utt"] for r in si["ctc_train"] + si["ctc_test"]}
    extra_si = stage_sinhala(set(si["ctc_train_speakers"]), have, want_utts=4200)
    print(f"  staged {len(extra_si)} extra Sinhala utts "
          f"({sum(r['duration'] for r in extra_si)/3600:.2f} h)", flush=True)
    build("sinhala", extra_si)

    en = json.load(open(f"{ROOT}/data/english_split.json", encoding="utf-8"))
    have_en = {r["utt"] for r in en["ctc_train"] + en["ctc_test"]}
    extra_en = more_english(en["ctc_train_speakers"], have_en)
    print(f"  found {len(extra_en)} extra English utts "
          f"({sum(r['duration'] for r in extra_en)/3600:.2f} h)", flush=True)
    build("english", extra_en)
