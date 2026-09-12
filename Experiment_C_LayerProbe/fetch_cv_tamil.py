#!/usr/bin/env python
"""Common Voice Tamil audio: the discriminating arm for language vs corpus.

The collapse finding is fully confounded - English is Common Voice, Sinhala and Tamil
are OpenSLR, so "Si/Ta collapse, English does not" is observationally identical to
"OpenSLR collapses, Common Voice does not". Common Voice Tamil breaks that tie for
Tamil: same language as the OpenSLR arm, same corpus as the English arm.
"""
import tarfile, urllib.request, csv, collections, os, json, subprocess
csv.field_size_limit(10 ** 7)
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = f"{ROOT}/data/tamil_cv"
PER_SPK = 170          # headroom over the 3.5 h budget without one speaker dominating
URL = ("https://huggingface.co/datasets/fsicoli/common_voice_17_0/resolve/main/"
       "audio/ta/train/ta_train_{i}.tar")

spk, txt = {}, {}
with open(f"{ROOT}/data/cv_ta_train.tsv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
        spk[r["path"]] = r["client_id"]; txt[r["path"]] = r["sentence"]
os.makedirs(OUT, exist_ok=True)
have = collections.Counter()
for i in (0, 1):
    tmp = f"{ROOT}/data/ta_train_{i}.tar"
    if not os.path.exists(tmp):
        subprocess.run(["curl", "-sL", "--fail", "-C", "-", "--speed-limit", "50000",
                        "--speed-time", "45", "--retry", "20", "--retry-delay", "3",
                        "--retry-all-errors", "-o", tmp, URL.format(i=i)], check=True)
    with tarfile.open(tmp) as tf:
        for m in tf:
            if not m.isfile():
                continue
            b = os.path.basename(m.name)
            s = spk.get(b)
            if s is None or have[s] >= PER_SPK:
                continue
            d = f"{OUT}/{s[:12]}"
            os.makedirs(d, exist_ok=True)
            open(f"{d}/{b}", "wb").write(tf.extractfile(m).read())
            have[s] += 1
    os.remove(tmp)
    print(f"shard {i}: {sum(have.values())} clips from {len(have)} speakers", flush=True)
    if len(have) >= 22 and min(have.values()) >= 120:
        break
json.dump({s: n for s, n in have.items()}, open(f"{ROOT}/data/cv_tamil_counts.json", "w"))
print("DONE", sum(have.values()), "clips,", len(have), "speakers")
