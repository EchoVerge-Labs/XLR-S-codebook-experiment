#!/usr/bin/env python
"""Fine-tuned ASR arm.

AMENDMENT, 2026-09-14. This arm is NOT part of the pre-registered probing protocol in
config.py; it was added after review, which asked whether the frozen-probe picture
survives fine-tuning and whether there is a downstream gap at all. It is reported as a
post-hoc addition, not as a pre-registered result.

Everything that defines the comparison is inherited unchanged from the probes so the two
are commensurable: the same speaker-disjoint splits, the same 3.00 h train / 0.50 h test
budget, the same character vocabularies, and the same CER (greedy decode, collapse
repeats, drop blank; total edit distance over total reference length).

What differs is the only thing under test: the encoder is trained rather than frozen.
"""
import os, sys, json, math, time, random, argparse
import numpy as np, torch, torch.nn as nn, soundfile as sf, torchaudio
from transformers import Wav2Vec2ForCTC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CHECKPOINT_PRIMARY, SR, MAX_CLIP_SECONDS, LANGUAGES
from probes import edit_distance, make_scheduler

ROOT = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"

# Standard wav2vec2 CTC fine-tuning recipe for a small budget. The CNN feature encoder
# stays frozen (as in the original wav2vec 2.0 fine-tuning) so only the transformer and
# the head adapt; lr is 100x below the linear probe's because the whole encoder moves.
FINETUNE = dict(
    lr=1e-4, epochs=20, batch_size=8, weight_decay=0.005, grad_clip=1.0,
    warmup_frac=0.1, mask_time_prob=0.05, mask_feature_prob=0.0, layerdrop=0.0,
)


def load_audio(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    t = torch.from_numpy(x.mean(axis=1))
    if sr != SR:
        t = torchaudio.functional.resample(t, sr, SR)
    t = t[: int(MAX_CLIP_SECONDS * SR)]
    # the XLS-R feature extractor normalises each utterance to zero mean / unit variance
    return ((t - t.mean()) / torch.sqrt(t.var() + 1e-7)).numpy()


def prepare(lang, max_hours=None, suffix=""):
    sp = json.load(open(f"{ROOT}/data/{lang}_split{suffix}.json", encoding="utf-8"))
    vocab = sp["vocab"]
    c2i = {c: i for i, c in enumerate(vocab)}

    def take(recs, budget=None):
        out, sec = [], 0.0
        for r in recs:
            y = [c2i[c] for c in r["text"] if c in c2i]
            if not y:
                continue
            if budget and sec >= budget * 3600:
                break
            out.append(dict(utt=r["utt"], path=r["path"], y=y, text=r["text"],
                            duration=r["duration"]))
            sec += r["duration"]
        return out

    tr = take(sp["ctc_train"], max_hours)
    te = take(sp["ctc_test"])
    for r in tr + te:
        r["wave"] = load_audio(r["path"])
    return vocab, tr, te


def batches(recs, bs, shuffle, rng):
    """Length-sorted buckets keep padding down; batch ORDER is shuffled, not content."""
    order = sorted(range(len(recs)), key=lambda i: recs[i]["duration"])
    bs_list = [order[i:i + bs] for i in range(0, len(order), bs)]
    if shuffle:
        rng.shuffle(bs_list)
    return [[recs[i] for i in b] for b in bs_list]


def collate(batch):
    lens = [len(r["wave"]) for r in batch]
    x = torch.zeros(len(batch), max(lens))
    m = torch.zeros(len(batch), max(lens), dtype=torch.long)
    for i, r in enumerate(batch):
        x[i, : lens[i]] = torch.from_numpy(r["wave"]); m[i, : lens[i]] = 1
    ys = torch.tensor([t for r in batch for t in r["y"]], dtype=torch.long)
    yl = torch.tensor([len(r["y"]) for r in batch], dtype=torch.long)
    return x.to(DEV), m.to(DEV), ys.to(DEV), yl.to(DEV)


@torch.no_grad()
def evaluate(model, te, vocab, bs):
    """Greedy decode, identical to the probes: argmax, collapse repeats, drop blank."""
    model.eval()
    err = ref = 0
    for b in batches(te, bs, False, None):
        x, m, _, _ = collate(b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, attention_mask=m).logits
        pred = logits.float().argmax(-1).cpu().numpy()
        out_lens = model._get_feat_extract_output_lengths(m.sum(-1)).cpu().numpy()
        for r, p, n in zip(b, pred, out_lens):
            seq, prev = [], -1
            for t in p[:n]:
                if t != prev and t != 0:
                    seq.append(int(t))
                prev = int(t)
            hyp = "".join(vocab[i] for i in seq)
            err += edit_distance(hyp, r["text"]); ref += len(r["text"])
    model.train()
    return err / max(ref, 1)


def run(lang, seed, hp, max_hours=None, tag="finetune", suffix=""):
    t0 = time.time()
    vocab, tr, te = prepare(lang, max_hours, suffix)
    print(f"[{lang}] seed {seed}: {len(tr)} train / {len(te)} test utts, "
          f"{sum(r['duration'] for r in tr)/3600:.2f} h, vocab {len(vocab)}", flush=True)
    torch.manual_seed(seed); np.random.seed(seed); rng = random.Random(seed)

    model = Wav2Vec2ForCTC.from_pretrained(
        CHECKPOINT_PRIMARY, vocab_size=len(vocab), ctc_loss_reduction="mean",
        pad_token_id=0, mask_time_prob=hp["mask_time_prob"],
        mask_feature_prob=hp["mask_feature_prob"], layerdrop=hp["layerdrop"],
        ignore_mismatched_sizes=True).to(DEV)
    model.freeze_feature_encoder()
    model.config.ctc_zero_infinity = True

    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    steps = hp["epochs"] * math.ceil(len(tr) / hp["batch_size"])
    sch = make_scheduler(opt, steps, hp["warmup_frac"])

    hist, first_loss = [], None
    for ep in range(hp["epochs"]):
        tot = n = 0
        for b in batches(tr, hp["batch_size"], True, rng):
            x, m, ys, yl = collate(b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, attention_mask=m)
                logp = torch.log_softmax(out.logits.float(), dim=-1).transpose(0, 1)
                il = model._get_feat_extract_output_lengths(m.sum(-1))
                loss = nn.functional.ctc_loss(logp, ys, il, yl, blank=0,
                                              reduction="mean", zero_infinity=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            opt.step(); sch.step()
            tot += float(loss); n += 1
        first_loss = first_loss if first_loss is not None else tot / n
        cer = evaluate(model, te, vocab, hp["batch_size"]) if (
            ep + 1) % 5 == 0 or ep == hp["epochs"] - 1 else None
        hist.append(dict(epoch=ep + 1, loss=tot / n, cer=cer))
        print(f"[{lang}] seed {seed} ep {ep+1}/{hp['epochs']} loss {tot/n:.4f}"
              + (f" CER {cer:.4f}" if cer is not None else "")
              + f"  {time.time()-t0:.0f}s", flush=True)

    final = [h["cer"] for h in hist if h["cer"] is not None]
    res = dict(language=lang, seed=seed, hp=hp, max_hours=max_hours,
               n_train=len(tr), n_test=len(te), vocab_size=len(vocab),
               train_hours=sum(r["duration"] for r in tr) / 3600,
               loss_first_epoch=first_loss, loss_final=hist[-1]["loss"],
               cer_final=final[-1], cer_best=min(final), history=hist,
               seconds=round(time.time() - t0, 1))
    del model; torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="*", default=LANGUAGES)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=FINETUNE["epochs"])
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--tag", default="finetune")
    ap.add_argument("--split-suffix", default="",
                    help='e.g. "6h" to read data/<lang>_split6h.json')
    a = ap.parse_args()

    hp = dict(FINETUNE); hp["epochs"] = a.epochs
    out_path = f"{ROOT}/results/{a.tag}.json"
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for lang in a.langs:
        results.setdefault(lang, {})
        for s in a.seeds:
            if str(s) in results[lang]:
                print(f"[{lang}] seed {s}: cached, skipping", flush=True)
                continue
            results[lang][str(s)] = run(lang, s, hp, a.max_hours, a.tag, a.split_suffix)
            json.dump(results, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
