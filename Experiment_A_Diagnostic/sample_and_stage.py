"""Deterministically sample audio for each corpus arm and stage it to local disk.
The source data lives on an rclone FUSE mount, so reads are network-bound; staging
locally first keeps GPU inference from stalling on I/O."""
import os, random, json, shutil, sys, collections
from concurrent.futures import ThreadPoolExecutor

ROOT   = os.path.expanduser("~/EchoVerge-LABS/Experiment_A_Diagnostic")
CD     = os.path.expanduser("~/google-drive/Community Datasets")
STAGE  = os.path.join(ROOT, "data", "staged")
SEED, N_PER_LANG = 1234, 1000

def sample_youtube(tsv, lang, n):
    """Proportional-to-size sample across genre subfolders."""
    by_genre = collections.defaultdict(list)
    base = os.path.expanduser(f"~/google-drive/Pre Processed Data/{lang}")
    for line in open(tsv):
        sz, p = line.rstrip("\n").split("\t", 1)
        dur = (int(sz) - 44) / 32000.0
        if dur < 1.0:            # too short to yield a useful number of frames
            continue
        by_genre[os.path.relpath(p, base).split(os.sep)[0]].append(p)
    total = sum(len(v) for v in by_genre.values())
    rng, out = random.Random(SEED), []
    for g in sorted(by_genre):
        k = max(1, round(n * len(by_genre[g]) / total))
        out += [(p, g) for p in rng.sample(sorted(by_genre[g]), min(k, len(by_genre[g])))]
    rng.shuffle(out)
    return out[:n]

def sample_flat(paths, n, genre):
    rng = random.Random(SEED)
    return [(p, genre) for p in rng.sample(sorted(paths), min(n, len(paths)))]

def glob_wavs(*dirs):
    out = []
    for d in dirs:
        out += [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".wav")]
    return out

def build():
    C = {}
    C["Sinhala_YT"] = sample_youtube(f"{ROOT}/results/_inventory_sinhala.tsv", "Sinhala", N_PER_LANG)
    C["Tamil_YT"]   = sample_youtube(f"{ROOT}/results/_inventory_tamil.tsv",   "Tamil",   N_PER_LANG)
    eng = sorted(os.path.join(f"{ROOT}/data/english_cv", f)
                 for f in os.listdir(f"{ROOT}/data/english_cv") if f.endswith(".mp3"))
    C["English_CV"] = sample_flat(eng, N_PER_LANG, "read")
    C["Sinhala_read"] = sample_flat(glob_wavs(f"{CD}/OpenSLR30_Sinhala/audio"), N_PER_LANG, "read")
    C["Tamil_read"]   = sample_flat(glob_wavs(f"{CD}/OpenSLR65_Tamil/audio/female",
                                              f"{CD}/OpenSLR65_Tamil/audio/male"), N_PER_LANG, "read")
    return C

def stage(corpora):
    manifest = {}
    for corpus, items in corpora.items():
        dst_dir = os.path.join(STAGE, corpus)
        os.makedirs(dst_dir, exist_ok=True)
        recs, fails = [], []
        def cp(i_item):
            i, (src, genre) = i_item
            dst = os.path.join(dst_dir, f"{i:04d}_{genre}{os.path.splitext(src)[1]}")
            try:
                if not os.path.exists(dst) or os.path.getsize(dst) == 0:
                    shutil.copyfile(src, dst)
                return dict(idx=i, src=src, path=dst, genre=genre, bytes=os.path.getsize(dst))
            except Exception as e:
                fails.append((src, repr(e))); return None
        with ThreadPoolExecutor(max_workers=16) as ex:
            for r in ex.map(cp, enumerate(items)):
                if r: recs.append(r)
        manifest[corpus] = recs
        print(f"  {corpus:<16} staged {len(recs):>4}/{len(items)}  failed={len(fails)}")
        for s, e in fails[:3]: print("     FAIL", s, e)
    return manifest

if __name__ == "__main__":
    os.makedirs(STAGE, exist_ok=True)
    corpora = build()
    print("sampled:", {k: len(v) for k, v in corpora.items()})
    man = stage(corpora)
    json.dump(man, open(f"{ROOT}/results/staging_manifest.json", "w"), indent=1)
    print("-> results/staging_manifest.json")
