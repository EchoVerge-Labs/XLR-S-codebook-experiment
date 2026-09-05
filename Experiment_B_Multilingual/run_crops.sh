#!/bin/bash
# Cropped arms, run sequentially so they do not contend for the GPU and their
# logs do not interleave. 6.0 s is the primary cropped arm, 9.0 s the registered
# P25 sensitivity (see the AMENDMENT block in selection.py).
cd ~/EchoVerge-LABS/Experiment_B_Multilingual || exit 1
echo "########## CROP 6.0s (PRIMARY CROPPED) ##########"
~/venv/bin/python -u run_sweep.py --manifest fleurs_manifest_crop6.0.json --tag _crop6.0 || exit 1
echo "########## CROP 9.0s (SENSITIVITY) ##########"
~/venv/bin/python -u run_sweep.py --manifest fleurs_manifest_crop9.0.json --tag _crop9.0 || exit 1
echo "BOTH CROP ARMS DONE"
