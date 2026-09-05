"""
Experiment C - layer-wise probing. PRE-REGISTERED CONFIGURATION.

Fixed before any probe was trained. Stage A and Experiment B both returned
equivalence-bounded nulls on the acoustic codebook, so the low-resource penalty is
somewhere else; this experiment localises it by depth.

Everything below that defines a comparison, a criterion, or a hyperparameter is fixed
here in advance. Nothing in the analysis re-derives these from the results.
"""

# ----------------------------------------------------------------- model
CHECKPOINT_PRIMARY = "facebook/wav2vec2-xls-r-300m"      # 24 layers, frozen
CHECKPOINT_SECONDARY = "facebook/wav2vec2-large-xlsr-53"  # if time allows
N_LAYERS = 24            # transformer layers; layer 0 = CNN/feature-projection output
SR = 16000
FEATURE_DTYPE = "float16"

# ----------------------------------------------------------------- data budget
# MATCHING IS THE EXPERIMENT. Every language gets exactly the same budget.
TRAIN_HOURS = 3.0
TEST_HOURS = 0.5
N_SPEAKERS = 40          # bounded by Tamil OpenSLR-65 (50 speakers, 7.08 h total)
MAX_CLIP_SECONDS = 20.0  # long-clip guard; report how many clips are truncated

LANGUAGES = ["sinhala", "tamil", "english"]
CORPORA = {
    "sinhala": "OpenSLR-52 (crowdsourced read Sinhala, 16 kHz flac)",
    "tamil":   "OpenSLR-65 (crowdsourced read Tamil, 48 kHz wav)",
    "english": "Common Voice 17 English, train split (read prompted sentences, 48 kHz mp3)",
}

# Two tasks need two different splits, and they cannot be the same split:
#   CTC      - train/test SPEAKER-DISJOINT (standard for an ASR proxy)
#   SPEAKER  - train/test share speakers by necessity (you classify among known
#              speakers), so the split is utterance-disjoint WITHIN each speaker
CTC_TEST_SPEAKER_FRACTION = 0.25     # ~10 of 40 speakers held out for the CTC test set
SPEAKER_TEST_UTT_FRACTION = 0.15     # within-speaker utterance holdout for speaker ID

# ----------------------------------------------------------------- probes
# Deliberately LINEAR. A deeper probe measures the probe's capacity, not the
# representation's. Identical hyperparameters for every layer, language and task -
# any variation would invalidate the cross-layer and cross-language comparison.
SEEDS = [0, 1, 2]

# AMENDED after a pilot, before any cross-language comparison was run. The original
# lr=3e-4 / 12 epochs left the probe badly undertrained: English layer 12 gave CER 0.81,
# versus 0.29 at lr=3e-3 / 25 epochs on identical data. A probe that does not learn
# measures nothing and would have compressed every cross-layer difference toward the
# ceiling. Pilot used one layer of one language; the amended values are applied
# identically to every layer, language and seed.
CTC_PROBE = dict(
    kind="linear",                # nn.Linear(hidden, vocab), CTC loss, blank=0
    lr=1e-2, epochs=25, batch_size=32, optimizer="adamw", weight_decay=0.01,
    grad_clip=5.0, scheduler="linear_warmup", warmup_frac=0.1,
)
# Features are used RAW (no per-layer standardisation). Standardising was tried and
# rejected: it fixed nothing that the batch/LR change did not fix, and at every LR from
# 1e-4 to 3e-2 it left Sinhala flat at ~0.82 on every layer while English trained fine.
STANDARDISE_FEATURES = False

# CTC blank-collapse guard: the initial blank logit is offset downward so no seed
# starts in the all-blank local minimum. Without it seed 0 collapsed on several
# layers (Sinhala L15: 0.839 vs 0.278/0.276 for seeds 1 and 2) while gradient
# clipping at 1.0 and 0.5 made no difference.
# Divergence is detected from training loss alone (no test information): a healthy
# probe drops its CTC loss well below its first-epoch value, a collapsed one
# plateaus. Collapsed runs are restarted with a fresh init, up to this many times,
# and the restart count is reported. Lowering the initial blank bias was tried
# first and only changed WHICH seeds collapsed, so it was rejected.
CTC_MAX_RESTARTS = 3
# A collapsed run is identified WITHIN its (language, layer) cell: its final/first
# training-loss ratio is several times that of its healthy siblings on identical
# data (e.g. Sinhala L15: 0.27 collapsed vs 0.04 healthy). A global threshold does
# not work - a healthy English L4 run sits at 0.25, overlapping Sinhala's collapsed
# value - so the rule is relative, and uses training loss only.
CTC_DIVERGENCE_OUTLIER_FACTOR = 3.0

# Full pilot record (English/Sinhala, layers 6/10/14/18, seed 0) - the settings were
# chosen for STABILITY ACROSS LAYERS, not for any cross-language outcome:
#   bs=8,  lr=3e-4  English L12 CER 0.81            - undertrained, probe barely learns
#   bs=8,  lr=1e-3  English L12 CER 0.82            - undertrained
#   bs=8,  lr=3e-3  English L12 CER 0.29            - trains, but see below
#   bs=8,  lr=3e-3  Sinhala L2/6/10 0.45/0.31/0.28 but L14/L18 0.83/0.80  - DIVERGES
#   bs=32, lr=3e-3  Sinhala 0.83/0.26/0.83/0.81     - still diverges on some layers
#   bs=32, lr=3e-3  English 0.81 everywhere         - undertrained at this batch size
#   bs=32, lr=1e-2  Sinhala 0.32/0.28/0.29/0.30     - STABLE across depth
#   bs=32, lr=1e-2  English 0.39/0.31/0.25/0.26     - STABLE across depth
# A probe that diverges on some layers and not others manufactures curve structure that
# has nothing to do with the representation, which is exactly what this experiment
# measures. Stability across layers was therefore the selection criterion.
CTC_PILOT_RECORD = {
    "bs8_lr3e-4_english_L12": 0.8115, "bs8_lr1e-3_english_L12": 0.8158,
    "bs8_lr3e-3_english_L12": 0.2880,
    "bs8_lr3e-3_sinhala_L2_6_10_14_18_22": [0.4543, 0.3073, 0.2779, 0.8270, 0.8044, 0.3365],
    "bs32_lr3e-3_sinhala_L6_10_14_18": [0.831, 0.256, 0.829, 0.812],
    "bs32_lr1e-2_sinhala_L6_10_14_18": [0.319, 0.278, 0.285, 0.297],
    "bs32_lr3e-3_english_L6_10_14_18": [0.807, 0.810, 0.808, 0.823],
    "bs32_lr1e-2_english_L6_10_14_18": [0.393, 0.311, 0.247, 0.256],
}
# AMENDED after the same pilot. With the full within-speaker training set the probe sat
# at 0.93-1.00 accuracy at EVERY layer in all three languages - a ceiling, which cannot
# reproduce the expected early-peak profile because it cannot resolve layers at all.
# Subsampling to 3 training utterances per speaker restores dynamic range (English:
# 0.972 at L0 falling to 0.144 at L22) without changing the task. Applied identically
# to all languages. The remaining ceiling across layers 0-16 is a stated limitation.
SPEAKER_TRAIN_UTTS_PER_SPEAKER = 3
SPEAKER_PROBE = dict(
    kind="linear",                # mean-pool over time -> nn.Linear(hidden, n_speakers)
    lr=1e-3, epochs=30, batch_size=64, optimizer="adamw", weight_decay=0.01,
    grad_clip=5.0, scheduler="linear_warmup", warmup_frac=0.1,
)
# Matches the amended CTC settings. The first run of this probe used the stale
# pre-amendment values and was undertrained: its learned layer weights stayed at the
# uniform initialisation (centroid 11.50 = exactly the middle of 0-23 for every language
# and both tasks), which is not a layer-importance profile at all.
WEIGHTED_SUM_PROBE = dict(
    kind="softmax_weighted_sum",  # learnable weights over all 24 layers -> linear head
    lr=1e-2, epochs=25, batch_size=32, optimizer="adamw", weight_decay=0.01,
    grad_clip=5.0, scheduler="linear_warmup", warmup_frac=0.1,
)

# ----------------------------------------------------------------- primary comparison
# CER IS NOT COMPARABLE ACROSS SCRIPTS. Sinhala, Tamil and English differ in alphabet
# size and orthographic depth, so an absolute CER difference between them means nothing.
# Only the SHAPE of the per-layer curve is compared, after normalising each language by
# its own best layer. Character vocabulary size is reported alongside every result.
PRIMARY_COMPARISON = "normalised per-layer CER curve shape across languages"
NORMALISATION = "per-language CER(layer) / min_over_layers CER(layer)"

# ----------------------------------------------------------------- decision criteria
# "Uniform vs concentrated" is defined HERE, before any curve exists.
#   best_layer        - argmin of the per-layer CER curve
#   peakedness        - max(normalised CER) / min(normalised CER) = worst/best ratio
#   centroid          - sum(w_l * l) with w_l proportional to (1/normalised CER),
#                       i.e. a depth centre of mass for where usable information sits
CRITERIA = dict(
    best_layer_shift_material=3,     # |best_layer(X) - best_layer(EN)| >= 3 layers
    centroid_shift_material=2.0,     # |centroid(X) - centroid(EN)| >= 2.0 layers
    peakedness_ratio_material=1.25,  # peakedness differing by >= 25% relatively
)
# Verdict rule, fixed in advance:
#   CONCENTRATED - best_layer OR centroid shifts by at least the material amount above,
#                  i.e. the languages draw their usable information from different depths
#   UNIFORM      - no criterion is met: the curves have the same shape and the penalty
#                  (if any) is a level difference, not a depth difference
# A level difference in raw CER is NOT interpretable across scripts and is not used.

# ----------------------------------------------------------------- data scaling
# Separates "needs more data" from "representation ceiling". Run on each language's own
# best layer. A SHIFTED curve (same shape, more data for the same CER) is a data-
# efficiency story; a curve that PLATEAUS below English is a representation-ceiling one.
SCALING_HOURS = [0.5, 1.0, 2.0, 3.0]

# ----------------------------------------------------------------- instrument validation
# Experiment C's equivalent of Stage A's white-noise control. Run BEFORE interpreting
# anything. If the floor control produces structured curves, the probe is broken.
FLOOR_CONTROL = dict(
    model="randomly initialised wav2vec2 of identical architecture (same config, seed 0)",
    expectation="flat, poor layer curves for both tasks",
)
# Expected-profile check: the literature places speaker information EARLY in the stack
# and phonetic content MID-stack. If speaker-ID does not peak earlier than CTC in all
# three languages, stop and debug before interpreting.
EXPECTED_PROFILE = "speaker-ID best layer < CTC best layer, in every language"

# ----------------------------------------------------------------- known limits
LIMITS = [
    "Read speech only. Stage A (except its YouTube arm), Experiment B and Experiment C "
    "are all read-speech-anchored, while the project's target data is scraped in-the-"
    "wild audio. This is a scope limit of the whole evidential base, not of this "
    "experiment alone.",
    "3 h of labelled audio per language is a small probe budget by design. It can "
    "compare curve SHAPE across depth; it cannot establish absolute achievable CER, and "
    "it cannot separate a representation limit from a data limit except through the "
    "scaling curve.",
    "CER is script-dependent and is never compared across languages in absolute terms.",
    "n = 3 languages. Nothing here supports an inferential claim about languages in "
    "general.",
    "English comes from a different corpus family (Common Voice) than Sinhala and Tamil "
    "(OpenSLR crowdsourced read speech). Recording conditions are not identical, so a "
    "level difference could reflect corpus rather than language.",
]
