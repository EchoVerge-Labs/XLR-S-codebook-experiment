#!/usr/bin/env python
"""Single per-file table across all three arms, plus the run config that produced it."""
import json, sys, os, hashlib, subprocess
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/EchoVerge-LABS/Experiment_A_Diagnostic"))
import selection as S
from final_stats import STAGE_A_CONTROLS, MARGIN_PRIMARY, MARGIN_SECONDARY
import masked_probe as MP  # noqa - for protocol constants

RES = "results"
ARMS = [("_full", "full"), ("_crop6.0", "crop6"), ("_crop9.0", "crop9")]

frames = []
for tag, arm in ARMS:
    d = pd.read_parquet(f"{RES}/per_file_metrics{tag}.parquet")
    d.insert(0, "arm", arm)
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
df.to_parquet(f"{RES}/per_file_metrics.parquet", index=False)
df.to_csv(f"{RES}/per_file_metrics.csv", index=False)

man = json.load(open(f"{RES}/fleurs_manifest.json"))
cfg = dict(
    experiment="Experiment B - multilingual seen/unseen sweep",
    checkpoints={"xlsr53": "facebook/wav2vec2-large-xlsr-53",
                 "xlsr300m": "facebook/wav2vec2-xls-r-300m"},
    dataset=dict(name="google/fleurs", split="test", languages=len(man),
                 clips_per_language=500, sampling_seed=1234,
                 total_clips=sum(v["n_staged"] for v in man.values())),
    measurement=dict(source="imported verbatim from Experiment_A_Diagnostic",
                     mask_prob=MP.MASK_PROB, mask_span=MP.MASK_SPAN,
                     n_negatives=MP.N_NEG, temperature=MP.TEMP,
                     max_seconds=MP.MAX_S, repeats_per_file=3,
                     mask_seed_formula="1000*file_id + repeat_index",
                     dtype="bfloat16", energy_gate="rms > 0.10 * per-file p90(rms)"),
    arms={"full": dict(crop_seconds=None, role=S.ARM_ROLES["full"]),
          "crop6": dict(crop_seconds=6.0, role=S.ARM_ROLES["crop6.0"]),
          "crop9": dict(crop_seconds=9.0, role=S.ARM_ROLES["crop9.0"])},
    crop_rule=dict(percentile=S.CROP_PERCENTILE, round_to=S.CROP_ROUND_TO,
                   windows=S.CROP_WINDOWS, primary=S.CROP_PRIMARY,
                   short_clips="dropped, not padded"),
    pairs=[dict(seen=s, unseen=u, family=f, description=d) for s, u, f, d in S.PAIRS],
    proximity={c: dict(rating=v[0], closest_seen=v[1], rationale=v[2])
               for c, v in S.PROXIMITY.items()},
    proximity_sensitivity=S.PROXIMITY_SENSITIVITY,
    distant_subgroup_reporting=S.DISTANT_SUBGROUP_REPORTING,
    equivalence=dict(stage_a_control_margins=STAGE_A_CONTROLS,
                     margin_primary=MARGIN_PRIMARY, margin_secondary=MARGIN_SECONDARY,
                     caveat=("Stage A margins are unpaired absolute differences across "
                             "corpora on XLS-R 0.3B; this is a paired within-corpus "
                             "difference on XLSR-53. Calibration anchor only.")),
    language_metadata=f"{RES}/fleurs_languages.json",
    pretraining_lists={"xlsr53": f"{RES}/xlsr53_languages.json",
                       "xlsr300m": f"{RES}/xlsr_300m_languages.json"},
)
json.dump(cfg, open(f"{RES}/run_config.json", "w"), indent=1, ensure_ascii=False)
print(f"per_file_metrics.parquet: {len(df):,} rows  arms={sorted(df.arm.unique())}  "
      f"langs={df.language.nunique()}  ckpts={df.checkpoint.nunique()}")
print("columns:", list(df.columns))
print(df.groupby(["arm", "checkpoint"]).size().to_string())
print("-> results/run_config.json")
