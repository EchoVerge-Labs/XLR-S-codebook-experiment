#!/usr/bin/env python
"""Build the matched, budget-exact splits for all three languages.

Two tasks need two different splits and they cannot be the same one:
  CTC     - speaker-disjoint train/test (standard for an ASR proxy)
  SPEAKER - train/test necessarily SHARE speakers (you classify among known speakers),
            so the split is utterance-disjoint within each speaker

Both are carved from one 3.5 h per-language pool so the cached features serve both.
"""
import os, json, csv, random, unicodedata, collections, sys
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (TRAIN_HOURS, TEST_HOURS, N_SPEAKERS, MAX_CLIP_SECONDS,
                    CTC_TEST_SPEAKER_FRACTION, SPEAKER_TEST_UTT_FRACTION, LANGUAGES)

ROOT = os.path.dirname(os.path.abspath(__file__))
SEED = 1234
csv.field_size_limit(10 ** 7)


def norm_text(t):
    """Identical normalisation for every language: NFC, lowercase, collapse whitespace,
    strip punctuation that is not part of any of these writing systems."""
    t = unicodedata.normalize("NFC", t).lower().strip()
    drop = '.,!?;:"“”‘’()[]{}<>/\\|@#$%^&*_+=~`–—…'
    t = "".join(" " if c in drop else c for c in t)
    return " ".join(t.split())


def load_english():
    """CV English: speaker = client_id, transcript from train.tsv."""
    meta = {}
    with open(f"{ROOT}/data/cv_en_train.tsv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            meta[r["path"]] = r["sentence"]
    recs = []
    base = f"{ROOT}/data/english_cv"
    for spk in sorted(os.listdir(base)):
        d = os.path.join(base, spk)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".mp3"):
                continue
            txt = meta.get(f)
            if not txt:
                continue
            p = os.path.join(d, f)
            try:
                info = sf.info(p)
            except Exception:
                continue
            recs.append(dict(speaker=spk, utt=f[:-4], path=p,
                             text=norm_text(txt), duration=info.duration))
    return recs


def load_staged(lang):
    m = json.load(open(f"{ROOT}/data/{lang}_staged.json", encoding="utf-8"))
    out = []
    for r in m["records"]:
        t = norm_text(r["text"])
        if t:
            out.append(dict(speaker=r["speaker"], utt=r["utt"], path=r["path"],
                            text=t, duration=r["duration"]))
    return out


def take_hours(recs, hours, rng):
    """Greedy fill to a duration budget, shuffled so it is not length-ordered."""
    recs = recs[:]
    rng.shuffle(recs)
    out, sec, budget = [], 0.0, hours * 3600
    for r in recs:
        if r["duration"] > MAX_CLIP_SECONDS:
            continue
        if sec >= budget:
            break
        out.append(r); sec += r["duration"]
    return out, sec


def build(lang, recs):
    rng = random.Random(SEED)
    by_spk = collections.defaultdict(list)
    for r in recs:
        by_spk[r["speaker"]].append(r)
    # keep the N_SPEAKERS with most audio, so the budget is reachable
    spk_dur = {s: sum(x["duration"] for x in v) for s, v in by_spk.items()}
    speakers = sorted(spk_dur, key=lambda s: -spk_dur[s])[:N_SPEAKERS]
    n_test_spk = max(1, round(len(speakers) * CTC_TEST_SPEAKER_FRACTION))
    # hold out the SMALLEST speakers for the CTC test set, preserving train capacity
    test_spk = speakers[-n_test_spk:]
    train_spk = speakers[:-n_test_spk]

    ctc_test, sec_te = take_hours([r for s in test_spk for r in by_spk[s]], TEST_HOURS, rng)
    ctc_train, sec_tr = take_hours([r for s in train_spk for r in by_spk[s]], TRAIN_HOURS, rng)

    # speaker-ID pool = everything the CTC split uses, re-partitioned within speaker
    pool = ctc_train + ctc_test
    spk_train, spk_test = [], []
    for s, group in collections.defaultdict(list, {
            s: [r for r in pool if r["speaker"] == s] for s in speakers}).items():
        if len(group) < 4:
            continue
        g = group[:]; rng.shuffle(g)
        k = max(1, int(round(len(g) * SPEAKER_TEST_UTT_FRACTION)))
        spk_test += g[:k]; spk_train += g[k:]

    chars = sorted({c for r in pool for c in r["text"]})
    vocab = ["<blank>"] + chars
    stats = dict(
        language=lang, n_speakers_pool=len(speakers),
        ctc_train=dict(n=len(ctc_train), hours=sec_tr / 3600, speakers=len(train_spk),
                       mean_clip=sec_tr / max(len(ctc_train), 1)),
        ctc_test=dict(n=len(ctc_test), hours=sec_te / 3600, speakers=len(test_spk),
                      mean_clip=sec_te / max(len(ctc_test), 1)),
        speaker_train=dict(n=len(spk_train)), speaker_test=dict(n=len(spk_test)),
        utts_per_speaker=dict(
            min=min(len([r for r in pool if r["speaker"] == s]) for s in speakers),
            median=float(np.median([len([r for r in pool if r["speaker"] == s])
                                    for s in speakers])),
            max=max(len([r for r in pool if r["speaker"] == s]) for s in speakers)),
        char_vocab_size=len(chars), vocab_size_with_blank=len(vocab),
        speaker_disjoint_ctc=bool(set(train_spk) & set(test_spk) == set()),
    )
    split = dict(stats=stats, vocab=vocab,
                 speakers=speakers, ctc_train_speakers=train_spk, ctc_test_speakers=test_spk,
                 ctc_train=ctc_train, ctc_test=ctc_test,
                 speaker_train=spk_train, speaker_test=spk_test)
    json.dump(split, open(f"{ROOT}/data/{lang}_split.json", "w"), ensure_ascii=False)
    return stats


if __name__ == "__main__":
    allstats = {}
    for lang in LANGUAGES:
        recs = load_english() if lang == "english" else load_staged(lang)
        print(f"{lang}: {len(recs)} candidate utts, "
              f"{sum(r['duration'] for r in recs)/3600:.2f} h available", flush=True)
        allstats[lang] = build(lang, recs)
    json.dump(allstats, open(f"{ROOT}/results/data_statistics.json", "w"), indent=1)

    print("\n" + "=" * 104)
    print("  MATCHED DATA STATISTICS")
    print("=" * 104)
    print(f"{'language':<10}{'CTC train':>22}{'CTC test':>22}{'spk pool':>10}"
          f"{'utts/spk':>18}{'char vocab':>12}")
    for l, s in allstats.items():
        tr, te = s["ctc_train"], s["ctc_test"]
        ups = s["utts_per_speaker"]
        print(f"{l:<10}{tr['n']:>6} utt {tr['hours']:>5.2f}h {tr['speakers']:>2}spk"
              f"{te['n']:>6} utt {te['hours']:>5.2f}h {te['speakers']:>2}spk"
              f"{s['n_speakers_pool']:>10}"
              f"{ups['min']:>6}/{ups['median']:>5.0f}/{ups['max']:<5.0f}"
              f"{s['char_vocab_size']:>12}")
    print("\nmean clip duration (s):", {l: round(s["ctc_train"]["mean_clip"], 2)
                                        for l, s in allstats.items()})
    print("speaker-disjoint CTC split:", {l: s["speaker_disjoint_ctc"]
                                          for l, s in allstats.items()})
    print("speaker-ID sets:", {l: (s["speaker_train"]["n"], s["speaker_test"]["n"])
                               for l, s in allstats.items()})
