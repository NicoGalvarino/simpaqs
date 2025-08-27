import pandas as pd
import numpy as np

from astropy.table import Table
from astropy.io import fits

from multiprocessing import Pool, cpu_count
from pathlib import Path
from argparse import ArgumentParser

from fits_utils import pandas_from_fits, save_to_fits

def recover_QSO_props(row):

    template = row['TEMPLATE']
    hdu = fits.open('/data2/home2/nguerrav/QSO_simpaqs/QSOs_full_cat/' + template)

    mag_abs = hdu[0].header['MAG']
    EBV = hdu[0].header['EBV']
    log_MBH = hdu[0].header['LOG_MBH']
    log_REdd = hdu[0].header['LOG_REDD']

    return mag_abs, EBV, log_MBH, log_REdd


def process_batch(batch_data):
    """Process a batch of rows for better efficiency"""
    batch_indices, batch_rows = batch_data
    results = []
    
    for idx, row in zip(batch_indices, batch_rows):
        try:
            qso_props = recover_QSO_props(row)
            results.append((idx, qso_props))
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            results.append((idx, (np.nan, np.nan, np.nan, np.nan)))
    
    return results


def process_catalog_parallel(cat, n_cores=None, batch_size=1000):
    """Process catalog using multiprocessing with batching"""
    if n_cores is None:
        n_cores = max(1, int(cpu_count() * 0.75))
    
    print(f"Processing {len(cat)} spectra using {n_cores} CPU cores with batch size {batch_size}")
    
    # Create batches
    indices = list(cat.index)
    rows = [cat.iloc[i] for i in indices]
    
    batches = []
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_rows = rows[i:i+batch_size]
        batches.append((batch_indices, batch_rows))
    
    print(f"Created {len(batches)} batches")
    
    # Process batches in parallel
    with Pool(processes=n_cores) as pool:
        batch_results = pool.map(process_batch, batches)
    
    # Collect results
    all_results = {}
    for batch_result in batch_results:
        for idx, qso_props in batch_result:
            all_results[idx] = qso_props
    
    total_rows = len(cat.index)
    progress_step = max(1, total_rows // 20)  # 5% increments
    processed_count = 0

    # Update catalog
    for idx in cat.index:
        if idx in all_results:
            mag_abs, EBV, log_MBH, log_REdd = all_results[idx]
            cat.loc[idx, 'QSO_abs_mag'] = mag_abs
            cat.loc[idx, 'QSO_EBV'] = EBV
            cat.loc[idx, 'QSO_log_MBH'] = log_MBH
            cat.loc[idx, 'QSO_log_REdd'] = log_REdd
        else:
            cat.loc[idx, ['QSO_abs_mag', 'QSO_EBV', 'QSO_log_MBH', 'QSO_log_REdd']] = np.nan

        processed_count += 1
        
        # Print progress every 5%
        if processed_count % progress_step == 0 or processed_count == total_rows:
            progress_percent = (processed_count / total_rows) * 100
            print(f"Progress: {processed_count}/{total_rows} rows processed ({progress_percent:.1f}%)")

    
    return cat

cat_path = Path('/data2/home2/nguerrav/Catalogues/')

# ------------------------------------------------------------------------------------------------------------------

def main():

    parser = ArgumentParser(description='Recover QSO properties and add them to the catalog')
    parser.add_argument('-n', '--number', type=int, default=None, help='Number of targets to process')
    parser.add_argument('--n-cores', type=int, default=None, help='Number of CPU cores (default: 75% of available)')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing (default: 100)')
    parser.add_argument('--input-cat', type=str, default='ByCycle_Final_Cat_fobs_qso_templates_with_SNR_golden_label.fits', 
                       help='Input catalog filename')
    parser.add_argument('--output-cat', type=str, default='ByCycle_Final_Cat_fobs_qso_templates_with_SNR_golden_label_QSO_props.fits',
                       help='Output catalog filename')
    args = parser.parse_args()

    cat = pandas_from_fits(cat_path / args.input_cat)

    for col in ['QSO_abs_mag', 'QSO_EBV', 'QSO_log_MBH', 'QSO_log_REdd']:
        if col not in cat.columns:
            cat[col] = np.nan

    cat_with_qso_props = process_catalog_parallel(cat, n_cores=args.n_cores, batch_size=args.batch_size)

    save_to_fits(cat_with_qso_props, cat_path / args.output_cat)

# ------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

