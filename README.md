# XLS-R codebook experiments

Does XLS-R's frozen acoustic codebook under-represent Sinhala and Tamil, and if not,
where does the low-resource penalty actually live?

Three experiments, run on an NVIDIA DGX Spark (GB10, ARM64, CUDA 13.0). Each one is
pre-registered before it runs, validated against a control that is known to fail, and
reported with the confounds it cannot separate.

| | Question | Answer |
|---|---|---|
| **A** — [`Experiment_A_Diagnostic/`](Experiment_A_Diagnostic/) | Is the codebook a worse fit for Sinhala/Tamil than for English? | No. Equivalence-bounded null; the sign is reversed. |
| **B** — [`Experiment_B_Multilingual/`](Experiment_B_Multilingual/) | Does presence in the pre-training language list predict codebook fit, across 18 languages? | No. Equivalent to zero in three context regimes. |
| **C** — [`Experiment_C_LayerProbe/`](Experiment_C_LayerProbe/) | If not the codebook, where by depth? | In progress. |

## Experiment A — quantization residual diagnostic

Measures how well the frozen quantizer represents each language, on 4,999 clips
(Sinhala and Tamil YouTube, read OpenSLR, Common Voice English) plus five deliberately
corrupted positive controls.

The headline finding is methodological as much as empirical. The residual the brief
asked for, `‖z − q‖²`, is **not computable** for wav2vec2 — `z` is 512-d and `q` is
768-d, and the quantizer never reconstructs `z`. The natural substitute, an unmasked
contrastive distance, turns out to be a **dead instrument**: white noise scores only
1.11× English on it, so no audio whatsoever could clear the "gap detected" threshold.

Measuring at *masked* positions instead — the only place wav2vec2's objective aligns
those projections — gives an instrument with real range: white noise collapses to a
+0.010 positive/negative gap against +0.45 for speech. On that instrument Sinhala and
Tamil are fit **better** than English, not worse.

A second finding fell out of the checkpoint used: XLSR-53 pre-trained on Tamil but
**not** Sinhala, yet the two are statistically indistinguishable.

## Experiment B — multilingual seen/unseen sweep

Scales A's accidental natural experiment to 18 languages in 9 **family-matched pairs**,
split on whether XLSR-53 saw them in pre-training. Several pairs are close relatives
across the membership line (Zulu/Xhosa, Estonian/Finnish, Tamil/Malayalam, Polish/Czech).

The result is not merely non-significant but **statistically equivalent to zero**,
against a margin calibrated on Experiment A's positive controls: the paired InfoNCE
difference is +0.033 / +0.040 / +0.041 across full-clip, 6 s-crop and 9 s-crop arms, with
every 90% CI inside ±0.289 — the penalty a 1.45× speed shift inflicts on this codebook.

Supporting: a negative control on XLS-R 0.3B returns dz = −0.01, and the dose–response
between pre-training hours and fit is flat within the low-resource range these languages
occupy (ρ = −0.014 over 33–321 h).

## Experiment C — layer-wise probing

Localises the penalty by depth: linear CTC and speaker-ID probes on each of 24
transformer layers, 3 h of labelled audio per language matched exactly, 3 seeds per cell.

## Reproducing

```bash
python -m venv ~/venv && ~/venv/bin/pip install \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu130
~/venv/bin/pip install transformers soundfile numpy scipy pandas pyarrow matplotlib
```

ARM64 note: the `cu121` wheels commonly cited do not exist for aarch64 and would not
support GB10 (sm_121) if they did. Use the `cu130` index.

Audio corpora, extracted features and staged copies are not versioned — Experiment C's
feature cache alone is 87 GB. Every script regenerates what it needs.

## Conventions

Each experiment holds its pre-registered configuration in the repository, including the
decision criteria fixed before any result existed, and records amendments with the
evidence that motivated them rather than editing them silently.
