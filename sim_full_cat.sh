#!/bin/bash
set -e

/home/nguerrav/miniconda3/envs/etc_4fs/bin/python3 simulate_catalog.py
if [ $? -eq 0 ]; then
    echo "simulate_catalog.py completed successfully"
    /home/nguerrav/miniconda3/envs/etc_4fs/bin/python3 rebin_and_get_SNR.py
else
    echo "simulate_catalog.py failed with exit code $?"
    exit 1
fi
