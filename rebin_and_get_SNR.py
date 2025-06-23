import pandas as pd
import numpy as np
from collections import Counter
from astropy.table import QTable, Table
import astropy.units as u
from pathlib import Path
import os

import spectres

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

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

cat_path =           Path('/data2/home2/nguerrav/Catalogues/')
L1_spec_path =       Path('/data2/home2/nguerrav/QSO_simpaqs/QSOs_full_cat_ETC_L1_output_with_fobs/')
rebinned_spec_path = Path('/data2/home2/nguerrav/QSO_simpaqs/QSOs_full_cat_ETC_L1_output_with_fobs_etc_grid/')

spec_units = {
    'WAVE': u.AA, 
    'FLUX': u.erg / (u.AA * u.s * u.cm**2), 
    'ERR_FLUX': u.erg / (u.AA * u.s * u.cm**2)
}

etc_grid = np.load('/data2/home2/nguerrav/QSO_simpaqs/npy_files/etc_wavelength_grid.npy')

cat = pandas_from_fits(cat_path / 'ByCycle_Final_Cat_fobs_qso_templates.fits')
cat[:100]

def rebin_spec(spec_filename):

    if os.path.exists(rebinned_spec_path / spec_filename):

        rebin_spec_tab = Table.read(rebinned_spec_path / spec_filename)

    else:
        L1_spec = Table.read(L1_spec_path / spec_filename)

        rebin_spec = spectres.spectres(etc_grid, L1_spec['WAVE'][0], L1_spec['FLUX'][0], spec_errs=L1_spec['ERR_FLUX'][0], 
                                       verbose=False)
        
        for i in range(rebin_spec[0].shape[0] - 1):
            if np.isnan(rebin_spec[0][i]):
                rebin_spec[0][i] = rebin_spec[0][i+1]
                rebin_spec[1][i] = rebin_spec[1][i+1]

        if np.isnan(rebin_spec[0][-1]):
            rebin_spec[0][-1] = rebin_spec[0][-2]
            rebin_spec[1][-1] = rebin_spec[1][-2]

        rebin_spec_tab = Table()

        rebin_spec_tab['WAVE'] = etc_grid.astype(np.float64)
        rebin_spec_tab['FLUX'] = rebin_spec[0].astype(np.float64)
        rebin_spec_tab['ERR_FLUX'] = rebin_spec[1].astype(np.float64)
        for col_name, unit in spec_units.items():
            rebin_spec_tab[col_name].unit = unit
        
        # get SNR
        rebin_spec_tab['SNR'] = rebin_spec_tab['FLUX'].value / rebin_spec_tab['ERR_FLUX'].value

        # set arms
        wav_ranges = [
            rebin_spec_tab['WAVE'] <= 4355,
            (rebin_spec_tab['WAVE'] >= 5159.8) & (rebin_spec_tab['WAVE'] <= 5730),
            rebin_spec_tab['WAVE'] >= 6099.8
        ]
        arms = ['blue', 'green', 'red']

        rebin_spec_tab['arm'] = np.select(wav_ranges, arms, default='unknown')

        rebin_spec_tab.write(rebinned_spec_path / spec_filename, format='fits', overwrite=True)

    return rebin_spec_tab


def get_SNR(rebinned_spec):

    SNR_blue = rebinned_spec['SNR'][rebinned_spec['arm']=='blue'].value
    SNR_green = rebinned_spec['SNR'][rebinned_spec['arm']=='green'].value
    SNR_red = rebinned_spec['SNR'][rebinned_spec['arm']=='red'].value

    SNR_blue_mean = np.mean(SNR_blue)
    SNR_green_mean = np.mean(SNR_green)
    SNR_red_mean = np.mean(SNR_red)

    SNR_mean = np.mean(rebinned_spec['SNR'].value)

    return (SNR_mean, SNR_blue_mean, SNR_green_mean, SNR_red_mean)

def rebin_and_SNR(row):

    z_str = str(np.round(row['REDSHIFT_ESTIMATE'], 4))
    mag_str = str(np.round(row['MAG'], 2))
    target_name = row['NAME']
    model_id = f'QSO_sim_ETC_z{z_str}_mag{mag_str}_{target_name}'

    spec_filename = model_id + '_LJ1.fits'

    if os.path.exists(L1_spec_path / spec_filename):
        rebinned_spec = rebin_spec(spec_filename)

        SNR_mean, SNR_blue_mean, SNR_green_mean, SNR_red_mean = get_SNR(rebinned_spec)

        return SNR_mean, SNR_blue_mean, SNR_green_mean, SNR_red_mean

    else:
        # print('Spectrum ', spec_filename, 'not defined \n')
        return np.nan, np.nan, np.nan, np.nan

def main():
    cat[['SNR_mean', 'SNR_blue_mean', 'SNR_green_mean', 'SNR_red_mean']] = cat.parallel_apply(
        lambda x: pd.Series(rebin_and_SNR(x)), axis=1
    )

    save_to_fits(cat, cat_path / 'ByCycle_Final_Cat_fobs_qso_templates_with_SNR.fits')

if __name__ == '__main__':
    main()
