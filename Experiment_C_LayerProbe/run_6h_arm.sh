#!/bin/bash
# 6 h budget arm: stage the extra Sinhala audio, then fine-tune Sinhala and English
# at 6.00 h with the same recipe and seeds as the 3 h runs.
set -u
cd "$(dirname "$0")"
echo "=== staging  $(date '+%H:%M:%S')"
~/venv/bin/python build_6h_split.py || { echo "STAGING FAILED"; exit 1; }
echo "=== fine-tuning  $(date '+%H:%M:%S')"
~/venv/bin/python finetune_ctc.py --langs sinhala english --seeds 0 1 2 \
    --epochs 30 --split-suffix 6h --tag finetune_6h
echo "=== done  $(date '+%H:%M:%S')"
