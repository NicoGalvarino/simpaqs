#!/bin/bash
set -e

# /home/nguerrav/miniconda3/envs/etc_4fs/bin/python3 simulate_quasars_no_abs.py
# if [ $? -eq 0 ]; then
    # echo "simulate_quasars_no_abs.py completed successfully"

/home/nguerrav/miniconda3/envs/etc_4fs/bin/python3 simulate_catalog.py --output /data2/home2/nguerrav/QSO_simpaqs/QSOs_L1_output_with_fobs_sim682_nexp
if [ $? -eq 0 ]; then
    echo "simulate_catalog.py completed successfully"
    
    /home/nguerrav/miniconda3/envs/etc_4fs/bin/python3 rebin_and_get_SNR.py
    if [ $? -eq 0 ]; then
        echo "All scripts completed successfully"
    else
        echo "rebin_and_get_SNR.py failed with exit code $?"
        exit 1
    fi
else
    echo "simulate_catalog.py failed with exit code $?"
    exit 1
fi
