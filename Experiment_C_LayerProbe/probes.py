#!/usr/bin/env python
"""Linear probes over cached layer features.

Both probes are deliberately a SINGLE linear layer. A deeper probe would measure the
probe's capacity rather than the representation's, and the whole comparison depends on
the probe being the same trivial thing at every layer and in every language.
"""
import os, json, sys, math, time
import numpy as np, torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (CTC_PROBE, SPEAKER_PROBE, WEIGHTED_SUM_PROBE, N_LAYERS,
                    SPEAKER_TRAIN_UTTS_PER_SPEAKER, STANDARDISE_FEATURES,
                    )

ROOT = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"


# --------------------------------------------------------------------- data
class LayerCache:
    """Layer features. One layer for one language is ~1.1 GB, so it is pulled fully into
    RAM rather than read through a memmap - random per-utterance reads from an 87 GB
    cache were the training bottleneck. Only one layer is held at a time."""

    def __init__(self, tag, lang, split, preload=True):
        d = f"{ROOT}/cache/{tag}/{lang}/{split}"
        self.dir = d
        meta = json.load(open(f"{d}/index.json"))
        self.index, self.dim = meta["index"], meta["dim"]
        self._frames = {}
        self.preload = preload

    def frames(self, layer):
        if layer not in self._frames:
            if self.preload:
                self._frames = {}                    # hold one layer at a time
                self._frames[layer] = np.load(f"{self.dir}/layer_{layer:02d}.npy")
            else:
                self._frames[layer] = np.load(f"{self.dir}/layer_{layer:02d}.npy",
                                              mmap_mode="r")
        return self._frames[layer]

    def pooled(self, layer):
        if not hasattr(self, "_pooled"):
            self._pooled = {}
        if layer not in self._pooled:
            self._pooled[layer] = np.load(f"{self.dir}/pooled_{layer:02d}.npy")
        return self._pooled[layer]

    def stats(self, layer):
        """Per-layer mean/std over training frames. Activation scale varies by an order
        of magnitude across depth, so without this a single learning rate measures
        activation magnitude rather than information content, and some layers simply
        diverge. Fitted on train and applied unchanged to test."""
        if not hasattr(self, "_stats"):
            self._stats = {}
        if not STANDARDISE_FEATURES:
            return (np.zeros(self.dim, np.float32), np.ones(self.dim, np.float32))
        if layer not in self._stats:
            f = self.frames(layer)
            sl = f[:: max(1, len(f) // 200000)].astype(np.float32)   # subsample for speed
            mu, sd = sl.mean(0), sl.std(0)
            # floor the std RELATIVE to the layer's typical scale: dividing a
            # near-constant dimension by ~0 produced values that overflowed fp16 and
            # collapsed training on every layer
            sd = np.maximum(sd, 0.05 * float(np.median(sd)) + 1e-6)
            self._stats[layer] = (mu, sd)
        return self._stats[layer]

    def get(self, layer, utt, norm=None):
        o, n = self.index[utt]
        x = np.asarray(self.frames(layer)[o:o + n]).astype(np.float32)
        if norm is not None:
            # keep fp32 once normalised - fp16 has too little headroom here
            return (x - norm[0]) / norm[1]
        return x.astype(np.float16)


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def make_scheduler(opt, total, warmup_frac):
    warm = max(1, int(total * warmup_frac))
    def fn(step):
        if step < warm:
            return step / warm
        return max(0.0, (total - step) / max(1, total - warm))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


# --------------------------------------------------------------------- CTC probe
def train_ctc(cache_tr, cache_te, recs_tr, recs_te, vocab, layer, seed,
              hp=CTC_PROBE, weighted=False, n_layers=N_LAYERS, max_hours=None):
    """CTC probes collapse into a degenerate solution on a minority of random
    initialisations - roughly one seed in three on some layers, with the surviving
    seeds agreeing to within 0.002 CER. A collapsed optimisation measures the
    initialisation, not the representation, so it is detected from the TRAINING
    loss (no test information) and restarted with a fresh init. The restart count
    is returned and reported."""
    out = _train_ctc_once(cache_tr, cache_te, recs_tr, recs_te, vocab, layer,
                          seed, hp, weighted, n_layers, max_hours)
    out["restarts"] = 0
    return out


def _train_ctc_once(cache_tr, cache_te, recs_tr, recs_te, vocab, layer, seed,
                    hp=CTC_PROBE, weighted=False, n_layers=N_LAYERS, max_hours=None):
    torch.manual_seed(seed); np.random.seed(seed)
    c2i = {c: i for i, c in enumerate(vocab)}
    dim = cache_tr.dim

    def encode(t):
        return [c2i[c] for c in t if c in c2i]

    tr = [r for r in recs_tr if r["utt"] in cache_tr.index and encode(r["text"])]
    if max_hours:
        cut, s = [], 0.0
        for r in tr:
            if s >= max_hours * 3600: break
            cut.append(r); s += r["duration"]
        tr = cut
    te = [r for r in recs_te if r["utt"] in cache_te.index and encode(r["text"])]

    head = nn.Linear(dim, len(vocab)).to(DEV)
    params = list(head.parameters())
    lw = None
    if weighted:
        lw = nn.Parameter(torch.zeros(n_layers, device=DEV)); params.append(lw)
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
    steps = hp["epochs"] * math.ceil(len(tr) / hp["batch_size"])
    sch = make_scheduler(opt, steps, hp["warmup_frac"])
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    norm = None if weighted else cache_tr.stats(layer)
    norms = [cache_tr.stats(l) for l in range(n_layers)] if weighted else None

    def batch_feats(cache, recs):
        if weighted:
            xs = [np.stack([cache.get(l, r["utt"], norms[l]) for l in range(n_layers)])
                  for r in recs]
        else:
            xs = [cache.get(layer, r["utt"], norm) for r in recs]
        lens = [x.shape[-2] for x in xs]
        m = max(lens)
        dt = np.float32
        if weighted:
            arr = np.zeros((len(xs), n_layers, m, dim), dt)
            for i, x in enumerate(xs): arr[i, :, :x.shape[1]] = x
        else:
            arr = np.zeros((len(xs), m, dim), dt)
            for i, x in enumerate(xs): arr[i, :x.shape[0]] = x
        return torch.from_numpy(arr).to(DEV).float(), torch.tensor(lens, device=DEV)

    order = np.arange(len(tr))
    epoch_loss = []
    for ep in range(hp["epochs"]):
        np.random.shuffle(order)
        head.train()
        acc_loss, nb = 0.0, 0
        for i in range(0, len(order), hp["batch_size"]):
            b = [tr[j] for j in order[i:i + hp["batch_size"]]]
            x, xl = batch_feats(cache_tr, b)
            if weighted:
                w = torch.softmax(lw, 0).view(1, -1, 1, 1)
                x = (x * w).sum(1)
            logits = head(x).log_softmax(-1).transpose(0, 1)
            tgt = [encode(r["text"]) for r in b]
            tl = torch.tensor([len(t) for t in tgt], device=DEV)
            flat = torch.tensor([c for t in tgt for c in t], device=DEV)
            loss = ctc(logits, flat, xl, tl)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, hp["grad_clip"])
            opt.step(); sch.step()
            acc_loss += float(loss.detach()); nb += 1
        epoch_loss.append(acc_loss / max(nb, 1))

    # a healthy probe drops its CTC loss substantially; a collapsed one plateaus
    loss_ratio = epoch_loss[-1] / max(epoch_loss[0], 1e-9)

    head.eval()
    errs = refs = 0
    with torch.no_grad():
        for i in range(0, len(te), hp["batch_size"]):
            b = te[i:i + hp["batch_size"]]
            x, xl = batch_feats(cache_te, b)
            if weighted:
                w = torch.softmax(lw, 0).view(1, -1, 1, 1)
                x = (x * w).sum(1)
            pred = head(x).argmax(-1).cpu().numpy()
            for k, r in enumerate(b):
                seq, prev, out = pred[k][: int(xl[k])], -1, []
                for s in seq:
                    if s != prev and s != 0:
                        out.append(vocab[s])
                    prev = s
                errs += edit_distance(out, list(r["text"])); refs += len(r["text"])
    cer = errs / max(refs, 1)
    return dict(cer=float(cer), n_train=len(tr), n_test=len(te),
                loss_ratio=float(loss_ratio), first_loss=float(epoch_loss[0]),
                final_loss=float(epoch_loss[-1]),
                weights=(torch.softmax(lw, 0).detach().cpu().numpy().tolist()
                         if weighted else None))


# --------------------------------------------------------------------- speaker probe
def train_speaker(cache_tr, cache_te, recs_tr, recs_te, speakers, layer, seed,
                  hp=SPEAKER_PROBE, weighted=False, n_layers=N_LAYERS):
    torch.manual_seed(seed); np.random.seed(seed)
    s2i = {s: i for i, s in enumerate(speakers)}
    # subsample to a fixed number of training utterances per speaker: with the full set
    # the probe is at ceiling on every layer and cannot resolve depth (see config)
    if SPEAKER_TRAIN_UTTS_PER_SPEAKER:
        import collections as _c, random as _r
        by = _c.defaultdict(list)
        for r in recs_tr:
            by[r["speaker"]].append(r)
        rng, sub = _r.Random(seed), []
        for s in sorted(by):
            g = by[s][:]; rng.shuffle(g)
            sub += g[:SPEAKER_TRAIN_UTTS_PER_SPEAKER]
        recs_tr = sub

    # The speaker split partitions utterances WITHIN speaker, so a speaker's utterances
    # are spread across both CTC caches. Looking them up in only one cache would train on
    # the 30 CTC-train speakers and test on the 10 CTC-test speakers - disjoint label
    # sets, and an accuracy of exactly zero. Both caches are searched.
    stores = [cache_tr, cache_te]
    rowmap = [{u: k for k, u in enumerate(c.index.keys())} for c in stores]
    # same standardisation rationale as the CTC probe, fitted on the train cache
    if not STANDARDISE_FEATURES:
        z = np.zeros(cache_tr.dim, np.float32); o = np.ones(cache_tr.dim, np.float32)
        pnorm = [(z, o)] * n_layers if weighted else (z, o)
    elif weighted:
        pnorm = []
        for l in range(n_layers):
            _q = cache_tr.pooled(l).astype(np.float32)
            pnorm.append((_q.mean(0), np.maximum(_q.std(0),
                                                 0.05 * float(np.median(_q.std(0))) + 1e-6)))
    elif True:
        _p = cache_tr.pooled(layer).astype(np.float32)
        _sd = np.maximum(_p.std(0), 0.05 * float(np.median(_p.std(0))) + 1e-6)
        pnorm = (_p.mean(0), _sd)

    def pack(recs):
        X, y = [], []
        for r in recs:
            for c, rm in zip(stores, rowmap):
                if r["utt"] in c.index:
                    if weighted:
                        X.append(np.stack([(c.pooled(l)[rm[r["utt"]]] - pnorm[l][0])
                                           / pnorm[l][1] for l in range(n_layers)]))
                    else:
                        X.append((c.pooled(layer)[rm[r["utt"]]] - pnorm[0])
                                 / pnorm[1])
                    y.append(s2i[r["speaker"]])
                    break
        return (torch.from_numpy(np.stack(X)).float(),
                torch.from_numpy(np.array(y)).long())

    Xtr, ytr = pack(recs_tr)
    Xte, yte = pack(recs_te)
    Xtr, ytr, Xte, yte = Xtr.to(DEV), ytr.to(DEV), Xte.to(DEV), yte.to(DEV)
    dim = Xtr.shape[-1]
    head = nn.Linear(dim, len(speakers)).to(DEV)
    params = list(head.parameters())
    lw = None
    if weighted:
        lw = nn.Parameter(torch.zeros(n_layers, device=DEV)); params.append(lw)
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
    steps = hp["epochs"] * math.ceil(len(Xtr) / hp["batch_size"])
    sch = make_scheduler(opt, steps, hp["warmup_frac"])
    lossf = nn.CrossEntropyLoss()

    for ep in range(hp["epochs"]):
        perm = torch.randperm(len(Xtr), device=DEV)
        for i in range(0, len(perm), hp["batch_size"]):
            b = perm[i:i + hp["batch_size"]]
            x = Xtr[b]
            if weighted:
                x = (x * torch.softmax(lw, 0).view(1, -1, 1)).sum(1)
            loss = lossf(head(x), ytr[b])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, hp["grad_clip"])
            opt.step(); sch.step()

    head.eval()
    with torch.no_grad():
        x = Xte
        if weighted:
            x = (x * torch.softmax(lw, 0).view(1, -1, 1)).sum(1)
        acc = (head(x).argmax(-1) == yte).float().mean().item()
    return dict(accuracy=float(acc), n_train=int(len(Xtr)), n_test=int(len(Xte)),
                n_classes=len(speakers), chance=1.0 / len(speakers),
                weights=(torch.softmax(lw, 0).detach().cpu().numpy().tolist()
                         if weighted else None))
