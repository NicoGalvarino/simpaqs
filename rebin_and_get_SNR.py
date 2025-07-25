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

col_format_all_S17 = {
    'NAME':pd.StringDtype(),
    'RA':np.float64, 'DEC':np.float64,
    'PMRA':np.float32, 'PMDEC':np.float32,
    'EPOCH':np.float32, 'RESOLUTION':np.int16,
    'SUBSURVEY':pd.StringDtype(),
    'TEMPLATE':pd.StringDtype(), 
    'RULESET':pd.StringDtype(),
    'EXTENT_FLAG':np.int32,
    'EXTENT_PARAMETER':np.float32,'EXTENT_INDEX':np.float32,
    'MAG_TYPE':pd.StringDtype(),
    'MAG':np.float32, 'MAG_ERR':np.float32,
    'DATE_EARLIEST':np.float64, 'DATE_LATEST':np.float64,
    'CADENCE':np.int64,
    'REDDENING':np.float32,
    'REDSHIFT_ESTIMATE':np.float32,
    'REDSHIFT_ERROR':np.float32,
    'CAL_MAG_ID_BLUE':pd.StringDtype(),
    'CAL_MAG_ID_GREEN':pd.StringDtype(),
    'CAL_MAG_ID_RED':pd.StringDtype(),
    'CAL_MAG_ERR_BLUE':np.float32,
    'CAL_MAG_ERR_GREEN':np.float32,
    'CAL_MAG_ERR_RED':np.float32,
    'CAL_MAG_BLUE':np.float32,
    'CAL_MAG_GREEN':np.float32,
    'CAL_MAG_RED':np.float32,
    'CLASSIFICATION':pd.StringDtype(),
    'CLASS_SPEC':pd.StringDtype(),
    'COMPLETENESS':np.float32,
    'PARALLAX':np.float32,
    'SWEEP_NAME':pd.StringDtype(), 
    'BRICKNAME':pd.StringDtype(), 
    'TYPE':pd.StringDtype(), 
    'BAND_LEGACY':pd.StringDtype(), 
    'REFERENCE_BAND':pd.StringDtype(), 
    'COMBINATION_USE':pd.StringDtype(), 
    'REDSHIFT_REF':pd.StringDtype(), 
    'EBV':np.float64, 
    'PLXSIG': np.float64, 
    'PMSIG': np.float64, 
    'SN_MAX': np.float64, 
    'MAG_G': np.float32, 
    'MAGERR_G': np.float32, 
    'MAG_R': np.float32, 
    'MAGERR_R': np.float32, 
    'MAG_I': np.float32, 
    'MAGERR_I': np.float32, 
    'MAG_Z': np.float32, 
    'MAGERR_Z': np.float32, 
    'MAG_Y': np.float32, 
    'MAGERR_Y': np.float32, 
    'MAG_J': np.float32, 
    'MAGERR_J': np.float32, 
    'MAG_H': np.float32, 
    'MAGERR_H': np.float32, 
    'MAG_K': np.float32, 
    'MAGERR_K': np.float32, 
    'MAG_W1': np.float32, 
    'MAGERR_W1': np.float32, 
    'MAG_W2': np.float32, 
    'MAGERR_W2': np.float32, 
    'SPECTYPE_DESI': pd.StringDtype()
    }

col_units = {
    "RA": "deg", "DEC": "deg", "PMRA": "mas/yr", "PMDEC": "mas/yr",
    "EPOCH": "yr", "MAG": "mag", "MAG_ERR": "mag", "EXTENT_PARAMETER": "arcsec",
    "DATE_EARLIEST": "d", "DATE_LATEST": "d", "REDDENING": "mag",
    "CAL_MAG_BLUE": "mag", "CAL_MAG_GREEN": "mag", "CAL_MAG_RED": "mag",
    "CAL_MAG_ERR_BLUE": "mag", "CAL_MAG_ERR_GREEN": "mag", "CAL_MAG_ERR_RED": "mag",
    "PARALLAX": "mas",
}

def cols_format_dict(format_dict, dataframe):
    matching_columns = {}
    
    for col in dataframe.columns:
        if col in format_dict:
            matching_columns[col] = format_dict[col]
    
    return matching_columns

def format_pd_for_fits(df):
    
    df_copy = df.copy()
    
    for col_name in df_copy.columns:  # object to string

        col_values = df_copy[col_name].values

        if col_values.dtype == 'object':
            df_copy[col_name] = df_copy[col_name].astype(pd.StringDtype())

    format_cols = cols_format_dict(col_format_all_S17, df_copy)
    df_copy = df_copy.astype(format_cols)

    for col_name in df_copy.columns:  # fill empty cells

        col_series = df_copy[col_name].values

        if pd.api.types.is_string_dtype(df_copy[col_name]) or isinstance(col_series.dtype, pd.StringDtype):
            df_copy[col_name] = df_copy[col_name].fillna('-')
        else:
            if col_name in ['MAG_Z', 'MAG', 'MAGERR_Z', 'MAG_ERR', 'MAG_G', 'CAL_MAG_BLUE', 
                            'MAGERR_G', 'CAL_MAG_ERR_BLUE', 'MAG_R', 'CAL_MAG_GREEN', 'MAGERR_R', 'CAL_MAG_ERR_GREEN', 
                            'MAG_I', 'CAL_MAG_RED', 'MAGERR_I', 'CAL_MAG_ERR_RED']:
                df_copy[col_name] = df_copy[col_name].fillna(1.0)
            else:
                df_copy[col_name] = df_copy[col_name].fillna(-999)
    
    df_copy.reset_index(drop=True, inplace=True)
    return df_copy

def save_to_fits(df, filepath, meta=None):

    df_for_fits = format_pd_for_fits(df)
    
    t = Table()

    format_cols = cols_format_dict(col_format_all_S17, df_for_fits)
    for col_name in df_for_fits.columns:
        if col_name in format_cols.keys():
            col_data = df_for_fits[col_name].astype(col_format_all_S17[col_name])
            col_data = col_data.values
        else:
            col_data = df_for_fits[col_name].values

        if hasattr(col_data, 'values'):
            t[col_name] = col_data.values
        else:
            t[col_name] = [x for x in col_data]
            
    if meta:
        t.meta.update(meta)

    t.write(filepath, format='fits', overwrite=True)

def pandas_from_fits(filepath):
    t = Table.read(filepath, format='fits')
    
    t = t.to_pandas()

    format_cols = cols_format_dict(col_format_all_S17, t)
    t = t.astype(format_cols)

    return t

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