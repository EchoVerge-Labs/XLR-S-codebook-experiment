#!/usr/bin/env python
"""Stream CV English train shards, keeping only clips from speakers that are dense in
train.tsv, until enough speakers have enough utterances for the speaker-ID probe.

CV train tars are speaker-interleaved (~5 clips per speaker per 4k scanned), so density
has to be accumulated across shards rather than found in one.
"""
import tarfile, urllib.request, csv, collections, os, json, sys
csv.field_size_limit(10**7)
OUT = "data/english_cv"; os.makedirs(OUT, exist_ok=True)
# CV train shards are ordered by speaker density: shard 0 holds only speakers with
# <=5 clips, shard 14 has median 187 clips/speaker, shard 27 is one mega-speaker.
# Fetch from the dense middle rather than scanning from the start.
NEED_SPK, NEED_UTT = 40, 70
SHARDS = [14, 13, 15, 12, 16]
URL = ("https://huggingface.co/datasets/fsicoli/common_voice_17_0/resolve/main/"
       "audio/en/train/en_train_{i}.tar")

spk, dense = {}, collections.Counter()
with open("data/cv_en_train.tsv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        spk[r["path"]] = r["client_id"]; dense[r["client_id"]] += 1
keep = {s for s, n in dense.items() if n >= 100}
print(f"{len(keep):,} speakers with >=100 clips in train.tsv", flush=True)

have = collections.Counter()
for i in SHARDS:
    req = urllib.request.Request(URL.format(i=i), headers={"User-Agent": "curl/8"})
    try:
        r = urllib.request.urlopen(req, timeout=600)
        tf = tarfile.open(fileobj=r, mode="r|")
        for m in tf:
            if not m.isfile():
                continue
            b = m.name.split("/")[-1]
            s = spk.get(b)
            if s is None or s not in keep:
                continue
            d = f"{OUT}/{s[:12]}"
            os.makedirs(d, exist_ok=True)
            open(f"{d}/{b}", "wb").write(tf.extractfile(m).read())
            have[s] += 1
    except Exception as e:
        print(f"  shard {i} ended: {repr(e)[:90]}", flush=True)
    ok = sum(1 for v in have.values() if v >= NEED_UTT)
    print(f"shard {i} done | speakers>={NEED_UTT} utts: {ok} | total kept {sum(have.values()):,}",
          flush=True)
    if ok >= NEED_SPK:
        break
json.dump({s: n for s, n in have.items()}, open("data/english_speaker_counts.json", "w"))
print("DONE", sum(1 for v in have.values() if v >= NEED_UTT), "usable speakers")
