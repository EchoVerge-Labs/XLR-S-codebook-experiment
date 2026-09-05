#!/usr/bin/env python
"""Split for the Common Voice Tamil arm, matched to the OpenSLR Tamil arm.

Same budget (3.00 h train / 0.50 h test), same speaker-disjoint construction, same
normalisation, same clip cap. The one thing that cannot be matched is speaker count:
Common Voice Tamil's train split contains 22 speakers against OpenSLR's 40, and there
is no way to manufacture more. That mismatch is recorded and reported.
"""
import os, sys, csv, json, random, collections
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_splits import norm_text, take_hours
from config import TRAIN_HOURS, TEST_HOURS, CTC_TEST_SPEAKER_FRACTION, SPEAKER_TEST_UTT_FRACTION

csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
SEED = 1234


def main():
    meta = {}
    with open(f"{ROOT}/data/cv_ta_train.tsv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            meta[r["path"]] = r["sentence"]
    recs = []
    base = f"{ROOT}/data/tamil_cv"
    for s in sorted(os.listdir(base)):
        d = os.path.join(base, s)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".mp3"):
                continue
            t = norm_text(meta.get(f, ""))
            if not t:
                continue
            p = os.path.join(d, f)
            try:
                info = sf.info(p)
            except Exception:
                continue
            recs.append(dict(speaker=s, utt=f[:-4], path=p, text=t, duration=info.duration))

    rng = random.Random(SEED)
    by = collections.defaultdict(list)
    for r in recs:
        by[r["speaker"]].append(r)
    spk_dur = {s: sum(x["duration"] for x in v) for s, v in by.items()}
    speakers = sorted(spk_dur, key=lambda s: -spk_dur[s])
    n_test = max(1, round(len(speakers) * CTC_TEST_SPEAKER_FRACTION))
    test_spk, train_spk = speakers[-n_test:], speakers[:-n_test]
    ctc_test, sec_te = take_hours([r for s in test_spk for r in by[s]], TEST_HOURS, rng)
    ctc_train, sec_tr = take_hours([r for s in train_spk for r in by[s]], TRAIN_HOURS, rng)
    pool = ctc_train + ctc_test
    spk_tr, spk_te = [], []
    for s in speakers:
        g = [r for r in pool if r["speaker"] == s]
        if len(g) < 4:
            continue
        rng.shuffle(g)
        k = max(1, int(round(len(g) * SPEAKER_TEST_UTT_FRACTION)))
        spk_te += g[:k]; spk_tr += g[k:]
    chars = sorted({c for r in pool for c in r["text"]})
    stats = dict(language="tamil_cv", n_speakers_pool=len(speakers),
                 ctc_train=dict(n=len(ctc_train), hours=sec_tr / 3600, speakers=len(train_spk),
                                mean_clip=sec_tr / max(len(ctc_train), 1)),
                 ctc_test=dict(n=len(ctc_test), hours=sec_te / 3600, speakers=len(test_spk),
                               mean_clip=sec_te / max(len(ctc_test), 1)),
                 char_vocab_size=len(chars),
                 speaker_disjoint_ctc=bool(not set(train_spk) & set(test_spk)))
    json.dump(dict(stats=stats, vocab=["<blank>"] + chars, speakers=speakers,
                   ctc_train_speakers=train_spk, ctc_test_speakers=test_spk,
                   ctc_train=ctc_train, ctc_test=ctc_test,
                   speaker_train=spk_tr, speaker_test=spk_te),
              open(f"{ROOT}/data/tamil_cv_split.json", "w"), ensure_ascii=False)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
