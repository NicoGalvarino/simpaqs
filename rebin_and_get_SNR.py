import pandas as pd
import numpy as np
from collections import Counter
from astropy.table import QTable, Table
import astropy.units as u
from pathlib import Path
import os
import datetime
from argparse import ArgumentParser
import spectres

from fits_utils import pandas_from_fits, save_to_fits

from multiprocessing import Pool, cpu_count
# from pandarallel import pandarallel
# pandarallel.initialize(progress_bar=True)

spec_units = {
    'WAVE': u.AA, 
    'FLUX': u.erg / (u.AA * u.s * u.cm**2), 
    'ERR_FLUX': u.erg / (u.AA * u.s * u.cm**2)
}

etc_grid = np.load('/data2/home2/nguerrav/QSO_simpaqs/npy_files/etc_wavelength_grid.npy')
cat_path =           Path('/data2/home2/nguerrav/Catalogues/')
L1_spec_path =       Path('/data2/home2/nguerrav/QSO_simpaqs/QSOs_full_cat_with_absorbers_in_blue_arm_ETC_L1_output_with_fobs/')
rebinned_spec_path = Path('/data2/home2/nguerrav/QSO_simpaqs/QSOs_full_cat_with_absorbers_in_blue_arm_ETC_L1_output_with_fobs_etc_grid/')
rebinned_spec_path.mkdir(exist_ok=True)

def process_batch(batch_data):
    """Process a batch of rows for better efficiency"""
    batch_indices, batch_rows = batch_data
    results = []
    
    for idx, row in zip(batch_indices, batch_rows):
        try:
            snr_results = rebin_and_SNR_single(row)
            results.append((idx, snr_results))
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            results.append((idx, (np.nan, np.nan, np.nan, np.nan)))
    
    return results

def rebin_and_SNR_single(row):
    """Process a single row - optimized version"""
    z_str = str(np.round(row['REDSHIFT_ESTIMATE'], 4))
    mag_str = str(np.round(row['MAG'], 2))
    target_name = row['NAME']
    model_id = f'QSO_sim_ETC_z{z_str}_mag{mag_str}_{target_name}'
    
    # spec_filename = model_id + '_LJ1.fits'
    spec_filename = model_id + '_LJ1_MgII.fits'
    L1_spec_file = L1_spec_path / spec_filename
    rebinned_spec_file = rebinned_spec_path / spec_filename
    
    # Quick check if L1 spectrum exists
    if not L1_spec_file.exists():
        return np.nan, np.nan, np.nan, np.nan
    
    # Check if rebinned version already exists and is newer than L1 version
    if rebinned_spec_file.exists():
        try:
            rebinned_spec = Table.read(rebinned_spec_file)
            SNR_mean, SNR_blue_mean, SNR_green_mean, SNR_red_mean = get_SNR_fast(rebinned_spec)
            return SNR_mean, SNR_blue_mean, SNR_green_mean, SNR_red_mean
        except:
            # If reading fails, recreate the file
            pass
    
    # Rebin the spectrum
    try:
        rebinned_spec = rebin_spec_optimized(spec_filename)
        SNR_mean, SNR_blue_mean, SNR_green_mean, SNR_red_mean = get_SNR_fast(rebinned_spec)
        return SNR_mean, SNR_blue_mean, SNR_green_mean, SNR_red_mean
    except Exception as e:
        print(f"Error processing {spec_filename}: {e}")
        return np.nan, np.nan, np.nan, np.nan

def rebin_spec_optimized(spec_filename):
    """Optimized version of rebin_spec"""
    L1_spec_file = L1_spec_path / spec_filename
    rebinned_spec_file = rebinned_spec_path / spec_filename
    
    # Read L1 spectrum
    L1_spec = Table.read(L1_spec_file)
    
    # Rebin using spectres
    rebin_flux, rebin_err = spectres.spectres(
        etc_grid, 
        L1_spec['WAVE'][0], 
        L1_spec['FLUX'][0], 
        spec_errs=L1_spec['ERR_FLUX'][0], 
        verbose=False
    )
    
    # Handle NaN values more efficiently
    nan_mask = np.isnan(rebin_flux)
    if np.any(nan_mask):
        # Forward fill NaN values
        for i in range(len(rebin_flux)):
            if nan_mask[i]:
                if i > 0:
                    rebin_flux[i] = rebin_flux[i-1]
                    rebin_err[i] = rebin_err[i-1]
                elif i < len(rebin_flux) - 1:
                    rebin_flux[i] = rebin_flux[i+1]
                    rebin_err[i] = rebin_err[i+1]
    
    # Create table
    rebin_spec_tab = Table()
    rebin_spec_tab['WAVE'] = etc_grid.astype(np.float64)
    rebin_spec_tab['FLUX'] = rebin_flux.astype(np.float64)
    rebin_spec_tab['ERR_FLUX'] = rebin_err.astype(np.float64)
    
    # Add units
    for col_name, unit in spec_units.items():
        if col_name in rebin_spec_tab.colnames:
            rebin_spec_tab[col_name].unit = unit
    
    # Calculate SNR
    with np.errstate(divide='ignore', invalid='ignore'):
        rebin_spec_tab['SNR'] = rebin_spec_tab['FLUX'].value / rebin_spec_tab['ERR_FLUX'].value
    
    # Set arms using vectorized operations
    wave_values = rebin_spec_tab['WAVE'].value
    arm_labels = np.full(len(wave_values), 'unknown', dtype='U7')
    arm_labels[wave_values <= 4355] = 'blue'
    arm_labels[(wave_values >= 5159.8) & (wave_values <= 5730)] = 'green'
    arm_labels[wave_values >= 6099.8] = 'red'
    rebin_spec_tab['arm'] = arm_labels
    
    # Save to file
    rebin_spec_tab.write(rebinned_spec_file, format='fits', overwrite=True)
    
    return rebin_spec_tab

def get_SNR_fast(rebinned_spec):
    """Optimized SNR calculation"""
    snr_values = rebinned_spec['SNR'].value
    arm_values = rebinned_spec['arm']
    
    # Use boolean indexing for faster filtering
    blue_mask = arm_values == 'blue'
    green_mask = arm_values == 'green'
    red_mask = arm_values == 'red'
    
    SNR_blue_mean = np.nanmean(snr_values[blue_mask]) if np.any(blue_mask) else np.nan
    SNR_green_mean = np.nanmean(snr_values[green_mask]) if np.any(green_mask) else np.nan
    SNR_red_mean = np.nanmean(snr_values[red_mask]) if np.any(red_mask) else np.nan
    SNR_mean = np.nanmean(snr_values)
    
    return SNR_mean, SNR_blue_mean, SNR_green_mean, SNR_red_mean

def process_catalog_parallel(cat, n_cores=None, batch_size=100):
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
        for idx, snr_values in batch_result:
            all_results[idx] = snr_values
    
    # Update catalog
    for idx in cat.index:
        if idx in all_results:
            snr_mean, snr_blue, snr_green, snr_red = all_results[idx]
            # cat.loc[idx, 'SNR_mean'] = snr_mean
            # cat.loc[idx, 'SNR_blue_mean'] = snr_blue
            # cat.loc[idx, 'SNR_green_mean'] = snr_green
            # cat.loc[idx, 'SNR_red_mean'] = snr_red
            cat.loc[idx, 'SNR_mean_mgii'] = snr_mean
            cat.loc[idx, 'SNR_blue_mean_mgii'] = snr_blue
            cat.loc[idx, 'SNR_green_mean_mgii'] = snr_green
            cat.loc[idx, 'SNR_red_mean_mgii'] = snr_red
        else:
            # cat.loc[idx, ['SNR_mean', 'SNR_blue_mean', 'SNR_green_mean', 'SNR_red_mean']] = np.nan
            cat.loc[idx, ['SNR_mean_mgii', 'SNR_blue_mean_mgii', 'SNR_green_mean_mgii', 'SNR_red_mean_mgii']] = np.nan
    
    return cat

def main():
    parser = ArgumentParser(description='Rebin spectra and calculate SNR')
    parser.add_argument('-n', '--number', type=int, default=None, help='Number of targets to process')
    parser.add_argument('--n-cores', type=int, default=None, help='Number of CPU cores (default: 75% of available)')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing (default: 100)')
    parser.add_argument('--input-cat', type=str, default='test_set_cat_not_in_golden_sample_SNR_3_with_MgII.fits', 
                       help='Input catalog filename')
    parser.add_argument('--output-cat', type=str, default='test_set_cat_not_in_golden_sample_SNR_3_with_MgII_with_SNR.fits',
                       help='Output catalog filename')
    
    args = parser.parse_args()
    
    t1 = datetime.datetime.now()
    
    # Load catalog
    print(f"Loading catalog: {args.input_cat}")
    cat = pandas_from_fits(cat_path / args.input_cat)
    
    # Filter if needed
    if args.number is not None:
        print(f"Processing only {args.number} targets")
        cat = cat.head(args.number)
    
    # Add SNR columns if they don't exist
    # for col in ['SNR_mean', 'SNR_blue_mean', 'SNR_green_mean', 'SNR_red_mean']:
    for col in ['SNR_mean_mgii', 'SNR_blue_mean_mgii', 'SNR_green_mean_mgii', 'SNR_red_mean_mgii']:
        if col not in cat.columns:
            cat[col] = np.nan
    
    # Filter to only process rows that don't have SNR calculated yet
    # mask = cat['SNR_mean'].isna() | (cat['SNR_mean'] < 0)
    mask = cat['SNR_mean_mgii'].isna() | (cat['SNR_mean_mgii'] < 0)
    if mask.any():
        print(f"Processing {mask.sum()} targets that need SNR calculation")
        cat_to_process = cat[mask].copy()
        
        # Process with multiprocessing
        cat_processed = process_catalog_parallel(cat_to_process, n_cores=args.n_cores, batch_size=args.batch_size)
        
        # Update original catalog
        cat.update(cat_processed)
    else:
        print("All targets already have SNR calculated")
    
    # Save results
    print(f"Saving results to: {args.output_cat}")
    save_to_fits(cat, cat_path / args.output_cat)
    
    t2 = datetime.datetime.now()
    dt = t2 - t1
    print(f"Finished processing in {dt.total_seconds():.1f} seconds")

if __name__ == '__main__':
    main()