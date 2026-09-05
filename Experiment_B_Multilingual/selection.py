"""Pre-registered design for Experiment B. Fixed BEFORE any metric was computed.

Family-matched SEEN/UNSEEN selection against XLSR-53's pre-training set, which the
1:1 family matching turns into a matched-pairs design. PAIRS is the primary analysis
unit; within-pair linguistic variance drops out, lowering the minimum detectable
effect at n=9.

PROXIMITY splits UNSEEN by whether a close relative was itself in pre-training. This
is the design's main threat: Malayalam is UNSEEN but Tamil is SEEN and both are South
Dravidian, so transfer from Tamil could mask a genuine membership effect. Splitting on
proximity separates "membership does not matter" from "phonological proximity, not
list membership, is what matters".

Sub-branch is read at fine granularity (Burmish vs Kuki-Chin, Malayic vs Javanese),
not at the coarse Tibeto-Burman / Malayo-Polynesian level.
"""

# (fleurs_config, group, family, sub_branch, script, region)
SELECTION = [
    ("bn_in",       "SEEN",   "Indo-European", "Indo-Aryan",      "Bengali",    "South Asia"),
    ("fa_ir",       "SEEN",   "Indo-European", "Iranian",         "Arabic",     "Middle East"),
    ("pl_pl",       "SEEN",   "Indo-European", "West Slavic",     "Latin",      "Europe"),
    ("et_ee",       "SEEN",   "Uralic",        "Finnic",          "Latin",      "Europe"),
    ("kk_kz",       "SEEN",   "Turkic",        "Kipchak",         "Cyrillic",   "Central Asia"),
    ("ta_in",       "SEEN",   "Dravidian",     "South Dravidian", "Tamil",      "South Asia"),
    ("id_id",       "SEEN",   "Austronesian",  "Malayic",         "Latin",      "SE Asia"),
    ("cmn_hans_cn", "SEEN",   "Sino-Tibetan",  "Sinitic",         "Han",        "East Asia"),
    ("zu_za",       "SEEN",   "Atlantic-Congo","Nguni",           "Latin",      "Southern Africa"),

    ("mr_in",       "UNSEEN", "Indo-European", "Indo-Aryan",      "Devanagari", "South Asia"),
    ("el_gr",       "UNSEEN", "Indo-European", "Hellenic",        "Greek",      "Europe"),
    ("cs_cz",       "UNSEEN", "Indo-European", "West Slavic",     "Latin",      "Europe"),
    ("fi_fi",       "UNSEEN", "Uralic",        "Finnic",          "Latin",      "Europe"),
    ("uz_uz",       "UNSEEN", "Turkic",        "Karluk",          "Latin",      "Central Asia"),
    ("ml_in",       "UNSEEN", "Dravidian",     "South Dravidian", "Malayalam",  "South Asia"),
    ("jv_id",       "UNSEEN", "Austronesian",  "Javanese",        "Latin",      "SE Asia"),
    ("my_mm",       "UNSEEN", "Sino-Tibetan",  "Burmish",         "Burmese",    "SE Asia"),
    ("xh_za",       "UNSEEN", "Atlantic-Congo","Nguni",           "Latin",      "Southern Africa"),
]

REFERENCE = ("en_us", "REFERENCE", "Indo-European", "Germanic", "Latin", "Global")
ALL = SELECTION + [REFERENCE]

# ---- primary analysis unit: 9 family-matched pairs (seen_config, unseen_config, family)
PAIRS = [
    ("bn_in",       "mr_in", "Indo-European",  "Bengali / Marathi      - both Indo-Aryan"),
    ("pl_pl",       "cs_cz", "Indo-European",  "Polish / Czech         - both West Slavic"),
    ("fa_ir",       "el_gr", "Indo-European",  "Persian / Greek        - Indo-European, different branches"),
    ("et_ee",       "fi_fi", "Uralic",         "Estonian / Finnish     - both Finnic"),
    ("kk_kz",       "uz_uz", "Turkic",         "Kazakh / Uzbek         - Turkic, Kipchak vs Karluk"),
    ("ta_in",       "ml_in", "Dravidian",      "Tamil / Malayalam      - both South Dravidian"),
    ("id_id",       "jv_id", "Austronesian",   "Indonesian / Javanese  - Malayo-Polynesian, different branches"),
    ("cmn_hans_cn", "my_mm", "Sino-Tibetan",   "Mandarin / Burmese     - Sinitic vs Burmish"),
    ("zu_za",       "xh_za", "Atlantic-Congo", "Zulu / Xhosa           - both Nguni"),
]

# ---- CHANGE 2: for each UNSEEN language, its nearest relative inside XLSR-53's 51.
# rating: CLOSE  = a seen language shares its fine-grained sub-branch
#         DISTANT = same family only, no seen language in the sub-branch
PROXIMITY = {
    "mr_in": ("CLOSE",   "Bengali (bn), Assamese (as)",
              "Indo-Aryan: same sub-branch (Marathi Southern, Bengali/Assamese Eastern)"),
    "cs_cz": ("CLOSE",   "Polish (pl)",
              "West Slavic: same sub-branch, closely related"),
    "fi_fi": ("CLOSE",   "Estonian (et)",
              "Finnic: same sub-branch, closely related"),
    "ml_in": ("CLOSE",   "Tamil (ta)",
              "South Dravidian: same sub-branch, very closely related"),
    "xh_za": ("CLOSE",   "Zulu (zu)",
              "Nguni: same sub-branch, largely mutually intelligible"),
    "el_gr": ("DISTANT", "none in branch",
              "Hellenic: no Hellenic language in XLSR-53; Indo-European family only"),
    "uz_uz": ("DISTANT", "Kazakh (kk), Kyrgyz (ky), Turkish (tr), Tatar (tt)",
              "Karluk: no Karluk in XLSR-53; Turkic family only (Kipchak/Oghuz present)"),
    "my_mm": ("DISTANT", "Hakh-Chin (cnh)",
              "Burmish: nearest is Kuki-Chin (Hakh-Chin), a different Tibeto-Burman "
              "branch with ~2 h of pre-training data; Sinitic present but distant"),
    "jv_id": ("DISTANT", "Indonesian (id), Tagalog (tl), Cebuano (ceb)",
              "Javanese is its own primary branch; Indonesian is Malayic, "
              "Tagalog/Cebuano are Philippine"),
}

# ---- CHANGE 3: Xhosa is absent from XLS-R 300m's 128 as well as XLSR-53's 51, so the
# negative-control arm is reported both with and without it.
XLSR300_UNSEEN = ["xh_za"]

CONFIGS = [c for c, *_ in ALL]
META = {c: dict(group=g, family=f, sub_branch=s, script=sc, region=r)
        for c, g, f, s, sc, r in ALL}
for c, (rating, closest, why) in PROXIMITY.items():
    META[c].update(proximity=rating, closest_seen=closest, proximity_note=why)


# ============================================================================
# PRE-REGISTERED ANALYSIS PLAN  -  fixed before any group mean was computed.
# Written 2026-09-04, after staging began but before any metric existed.
# ============================================================================

# Both arms are run unconditionally. Neither is a "correction" of the other:
# FLEURS is built from FLoRes, the same source sentences translated across
# languages, so a language's mean clip duration largely reflects how long that
# language takes to express the same content - syllable structure, agglutination,
# speech rate. That is a real typological property, not a recording artefact.
#   FULL    : how well the codebook handles each language AS ACTUALLY SPOKEN
#   CROPPED : how well it handles each language AT MATCHED CONTEXT
# If the two arms agree the result is robust; if they disagree, that disagreement
# is itself the finding and is reported as such.
ANALYSIS_ARMS = {
    "full":    dict(crop_seconds=None,
                    question="each language as actually spoken (unmodified Experiment A protocol)"),
    "cropped": dict(crop_seconds="P25_POOLED",
                    question="each language at matched context per clip"),
}

# Fixed-window length: the 25th percentile of the POOLED duration distribution over
# all 19 staged languages, rounded down to 0.5 s. Pooled, so the value cannot depend
# on group assignment. Clips shorter than the window are DROPPED, not padded; the
# drop count per language is reported.
CROP_PERCENTILE = 25
CROP_ROUND_TO = 0.5

# The paired test runs on 9 within-pair differences, so the contaminating quantity is
# duration differing WITHIN a pair, not between groups. After the sweep, Spearman
# correlation between each pair's duration delta and its InfoNCE delta is reported
# unconditionally. A strong correlation promotes the cropped arm to primary.
REPORT_PAIR_DURATION_CORRELATION = True

# The duration confound is characterised on FLEURS itself, per language, rather than
# extrapolated from Experiment A's VAD-segmented YouTube corpus where clip length also
# tracks segment quality.
REPORT_PER_LANGUAGE_DURATION_RHO = True

# n=4 cannot support a significance test (MDE d >= 1.85). The DISTANT subgroup is
# reported descriptively - the four language values against the SEEN distribution,
# group mean shown, no p-value as a headline - and labelled exploratory.
DISTANT_SUBGROUP_REPORTING = "descriptive_only"

# Burmese sits in DISTANT on the judgement that Hakh-Chin (~2 h, Kuki-Chin) is not
# meaningful transfer. Pre-registered sensitivity: repeat the subgroup analysis with
# Burmese moved to CLOSE. Both assignments reported.
PROXIMITY_SENSITIVITY = {"my_mm": "CLOSE"}


# ============================================================================
# AMENDMENT - recorded 2026-09-04, BEFORE any cropped-arm metric was computed.
# Adding a second window is not a forking path: both windows are fixed here in
# advance, both are reported unconditionally, neither can be dropped.
# ============================================================================
#
# PRIMARY/SENSITIVITY REVERSAL. The registered P25 rule lands at 9.0 s. At that
# window the clip-drop rate is a FUNCTION OF each language's mean duration - the
# very confound the crop exists to neutralise. Polish loses 47.6% of its clips
# and Persian 4.0%, so surviving Polish clips are its longest while surviving
# Persian clips are near-representative: selection on the confound, applied
# unequally by language. The cure partially reintroduces the disease. At 6.0 s
# the differential falls to 10.4 points.
#
#   full   (no crop) : PRIMARY OVERALL - unmodified Experiment A protocol, comparable
#                      with Experiment A, "each language as actually spoken"
#   crop6.0          : PRIMARY CROPPED  - "each language at matched context"
#   crop9.0          : SENSITIVITY      - the registered P25 rule, reported alongside
#
# The P25 rule is not abandoned; it is demoted to sensitivity and still reported.
ARM_ROLES = {
    "full":    "PRIMARY OVERALL (unmodified Experiment A protocol)",
    "crop6.0": "PRIMARY CROPPED (matched context, low differential loss)",
    "crop9.0": "SENSITIVITY (pre-registered P25 rule)",
}
CROP_WINDOWS = [6.0, 9.0]
CROP_PRIMARY = 6.0

# Reported alongside the results, unconditionally:
#  1. per-language retention rate for BOTH windows
#  2. within-pair retention delta at 9.0 s; flag pairs differing by >20 points
#  3. mean duration for SEEN, UNSEEN-CLOSE (n=5), UNSEEN-DISTANT (n=4) separately
RETENTION_FLAG_THRESHOLD_PP = 20.0
