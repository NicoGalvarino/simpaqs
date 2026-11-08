
python3 simulate_quasars_no_abs.py --dir /Volumes/T7_blu/4MOST_ByCycle/mock_data/QSO_templates/ --input_cat /Volumes/T7_blu/4MOST_ByCycle/mock_data/ByCycle_Final_cat_with_qselfie_682.fits

python3 simulate_catalog.py --output /Volumes/T7_blu/4MOST_ByCycle/mock_data/QSOs_L1_output/ --input_cat /Volumes/T7_blu/4MOST_ByCycle/mock_data/QSO_templates/ByCycle_Final_cat_with_qselfie_682_fobs_qso_templates.fits --temp-dir /Volumes/T7_blu/4MOST_ByCycle/mock_data/QSO_templates/

python3 rebin_and_get_SNR.py --input_cat ByCycle_Final_cat_with_qselfie_682_fobs_qso_templates.fits --cat-path /Volumes/T7_blu/4MOST_ByCycle/mock_data/QSO_templates/ --l1-spec-path /Volumes/T7_blu/4MOST_ByCycle/mock_data/QSOs_L1_output/ --rebinned-spec-path /Volumes/T7_blu/4MOST_ByCycle/mock_data/QSOs_rebinned_03ang/ --etc-grid-path /Volumes/T7_blu/4MOST_ByCycle/mock_data/wavelen_grid_03ang.npy
