#!/bin/bash
set -e

/home/nguerrav/miniconda3/envs/etc_4fs/bin/python3 simulate_quasars_no_abs.py
if [ $? -eq 0 ]; then
    echo "simulate_quasars_no_abs.py completed successfully"
    /home/nguerrav/miniconda3/envs/etc_4fs/bin/python3 simulate_catalog.py
else
    echo "simulate_quasars_no_abs.py failed with exit code $?"
    exit 1
fi
