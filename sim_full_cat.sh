
# python3 simulate_quasars_no_abs.py --dir /nfs/home/nguerrav/QSO_simpaqs/QSO_templates/ --input_cat /nfs/home/nguerrav/QSO_simpaqs/ByCycle_Final_cat_with_qselfie_682.fits

python3 simulate_catalog.py --output /nfs/home/nguerrav/QSO_simpaqs/QSOs_L1_output/ --input_cat /nfs/home/nguerrav/QSO_simpaqs/QSO_templates/ByCycle_Final_cat_with_qselfie_682_fobs_qso_templates.fits --temp-dir /nfs/home/nguerrav/QSO_simpaqs/QSO_templates/

python3 rebin_and_get_SNR.py --input-cat ByCycle_Final_cat_with_qselfie_682_fobs_qso_templates.fits --cat-path /nfs/home/nguerrav/QSO_simpaqs/QSO_templates/ --l1-spec-path /nfs/home/nguerrav/QSO_simpaqs/QSOs_L1_output/ --rebinned-spec-path /nfs/home/nguerrav/QSO_simpaqs/QSOs_rebinned_03ang/ --etc-grid-path /nfs/home/nguerrav/QSO_simpaqs/wavelen_grid_03ang.npy
