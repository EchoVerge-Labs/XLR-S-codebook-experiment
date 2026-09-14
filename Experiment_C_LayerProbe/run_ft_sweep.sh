#!/bin/bash
# Wait for the in-flight Tamil seed-0 run (PID below), then finish the sweep.
# finetune_ctc.py skips (lang, seed) cells already present in results/finetune.json.
while kill -0 795862 2>/dev/null; do sleep 30; done
exec ~/venv/bin/python finetune_ctc.py --langs tamil sinhala english --seeds 0 1 2 --epochs 30 --tag finetune
