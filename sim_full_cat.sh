# !/bin/bash
# set -e

# /home/nguerrav/miniconda3/envs/etc_4fs/bin/python3 simulate_catalog.py
# if [ $? -eq 0 ]; then
#     echo "simulate_catalog.py completed successfully"
#     /home/nguerrav/miniconda3/envs/etc_4fs/bin/python3 rebin_and_get_SNR.py
# else
#     echo "simulate_catalog.py failed with exit code $?"
#     exit 1
# fi

#!/bin/bash
# filepath: /home/nguerrav/4MOST_like_data/simpaqs/sim_full_cat.sh

set -e

# Run insert_absorber.py first
/home/nguerrav/miniconda3/envs/etc_4fs/bin/python3 insert_absorber.py
if [ $? -eq 0 ]; then
    echo "insert_absorber.py completed successfully"
    
    # Run simulate_catalog.py
    /home/nguerrav/miniconda3/envs/etc_4fs/bin/python3 simulate_catalog.py
    if [ $? -eq 0 ]; then
        echo "simulate_catalog.py completed successfully"
        
        # Run rebin_and_get_SNR.py
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
else
    echo "insert_absorber.py failed with exit code $?"
    exit 1
fi
