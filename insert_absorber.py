from astropy.io import fits
from astropy.table import Table, vstack, Column
from astropy.time import Time
import astropy.units as u
from argparse import ArgumentParser

from collections import Counter

import os
import sys
import datetime
import pandas as pd
import h5py
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

from lmfit.models import VoigtModel

from os import listdir
from os.path import isfile, join

import warnings
warnings.filterwarnings('ignore', message='invalid value encountered in divide', category=RuntimeWarning)
# warnings.filterwarnings('default')

from qmostetc import SEDTemplate, QMostObservatory, Ruleset, Rule, Filter, L1DXU

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

def load_MgII(folder):
    ''' Function to load hdf5 files with MgII absorbers from TNG50
    Input:
        folder: folder of hdf5 files
    Returns:
        MgII_dir: array with hdf5 files at different redshifts

    '''
    MgII_files = [f for f in listdir(folder) if isfile(join(folder, f))]
    MgII_dir = []
    
    for file in MgII_files:
        if '.hdf5' in file:
            MgII_dir.append(h5py.File(folder + file, 'r'))
        
    return MgII_dir

def estimate_metal_redshift(wave_mgii, flux_mgii, metal_cent):

    ''' Function to estimate the redshift of an MgII absorber using a Voigt profile fit 
    based on the MgII 2796 line
    
    Input:
        wave_mgii: wavelength grid of MgII lines
        flux_mgii: flux grid of one MgII absorber
    Returns:
        z_abs: estimated MgII absorber redshift
        mgii_2796_center_pos: central position of the MgII 2796 line in the wavelength grid
        
    '''
    
    # Estimate position of MgII 2796 based on the lowest value in flux grid
    line_pos = np.where(flux_mgii == min(flux_mgii))
    
    # Only consider the surrounding 20 Angstroms of the estimated MgII position
    min_pos = line_pos[0][0]
    if(min_pos+1 == len(wave_mgii)):
        min_pos = min_pos-1
    wl_dist = wave_mgii[min_pos+1] - wave_mgii[min_pos]
    max_pos = min_pos + int(10/wl_dist)
    min_pos = min_pos - int(10/wl_dist)
    if(min_pos < 0):
        min_pos = 0
    if(max_pos > len(wave_mgii)):
        max_pos = len(wave_mgii)

    # Fit a Voigt profile and estimate center of MgII 2796 absorption line  
    vModel = VoigtModel()
    params = vModel.guess(1-flux_mgii[min_pos:max_pos], x=wave_mgii[min_pos:max_pos])    
    fitted_Voigt = vModel.fit(1-flux_mgii[min_pos:max_pos], params, x=wave_mgii[min_pos:max_pos])
    
    # Find position on wavelength grid that fits the center of the MgII 2796 line the best
    mgii_2796_center = fitted_Voigt.best_values['center']
    mgii_2796_center_pos = np.where(wave_mgii <= mgii_2796_center)[0]  # stops at the maximum wave i.e. peak of the profile
    
    if(len(mgii_2796_center_pos) == 0):
        mgii_2796_center_pos = 0
    else:
        mgii_2796_center_pos = mgii_2796_center_pos[-1]
    
    # Calculate estimated redshift
    z_abs = (mgii_2796_center - metal_cent) / metal_cent
    
    return z_abs, mgii_2796_center_pos, mgii_2796_center



def shift_metal_abs(wave_mgii, flux_mgii, z_shifted, metal_cent, verbose=False):
    ''' Function to shift MgII spectrum to a specific redshift.
    Input:
        wave_mgii: wavelength grid of MgII lines
        flux_mgii: flux grid of one MgII absorber
        z_shift: delta z of shift
    Returns:
        flux_mgii_shifted: shifted flux grid of MgII absorber
        z_shifted: redshift of shifted MgII line
    '''
    # Estimate redshift and position in grid of MgII absoprtion line using a Voigt 
    # profile fit of the MgII 2796 line
    z_mgii, wl_z_mgii_pos, mgii_2796_center_wl = estimate_metal_redshift(wave_mgii, flux_mgii, metal_cent)
    # z_shifted = z_mgii + z_shift
    
    # Calculate central wavelength of shifted and original MgII 2796 line
    wl_z_shifted = metal_cent * (1 + z_shifted)
    wl_z_mgii = metal_cent * (1 + z_mgii)
    
    # Find position in grid to which the absorber should be shifted to and shift spectrum
    wl_z_shifted_pos = np.where(wave_mgii <= wl_z_shifted)[0]
    
    if(len(wl_z_shifted_pos) == 0):
        flux_mgii_shifted = flux_mgii
    else:
        wl_z_shifted_pos = wl_z_shifted_pos[-1]
        shift = wl_z_shifted_pos - wl_z_mgii_pos
        flux_mgii_shifted = np.roll(flux_mgii, shift)
    
    z_shifted, mgii_2796_center_pos, mgii_2796_center_wl = estimate_metal_redshift(wave_mgii,flux_mgii_shifted, metal_cent)
    
    # Sanity check
    if(verbose == True):
        print('Estimated original z:', z_mgii)
        print('Estimated shifted z:', z_shifted)
    
        # if z_mgii <= z_qso:

        #     good_z = True
        #     if(verbose==True):
        #         print('good z found')
        #         print('MgII z: ', z_mgii)
        #         print('QSO z: ', z_qso)
        # else:
        #     if(verbose==True):
        #         print('no good z - redo')
  
    return flux_mgii_shifted, z_shifted, mgii_2796_center_pos, mgii_2796_center_wl



def insert_metal_abs(spectrum, MgIIflux):
    ''' Function to insert MgII absorption line into spectrum. Simple multiplication. 
        Input:
            spectrum: QSO spectrum
            MgII_flux: MgII absorption line flux
        Returns:
            spectrum * MgIIflux: spectrum with inserted MgII absorption line
    '''

    spec_with_MgII = spectrum.copy()
    spec_with_MgII[:len(MgIIflux)] = spectrum[:len(MgIIflux)] * MgIIflux

    return spec_with_MgII


def insert_random_MgII(qso_template, MgII_dir, z_shifted_range, # z_shift_max_arr, 
                       z_qso, saveto, EW_min=0.002, verbose=False):
    
    ''' Function to insert a random MgII absorption line into the spectrum
    Input:
        qso_spectrum: QSO spectrum
        MgII_dir: MgII array with MgII lines at different redshifts (use load_MgII to get the right format)
        z_shift_max: maximum range of shifting the MgII line. This is done to get a continuous 
                     function of MgII absorber in z-space
        z_shifted_range: range of final z of the absorber
    Returns:
        spectrum: QSO spectrum with added MgII absorption line
        MgII_flux_shifted: MgII absorption spectrum shifted to a specific redshift
        MgII_prop: Properties of the MgII line. EW are rescaled to rest-frame 
    '''

    qso_template = Table.read(qso_template)
    qso_flux = qso_template['FLUX_DENSITY'][:]
    qso_spectrum = np.asarray(qso_flux)
    
    # good_z = False
    # while (good_z == False):
    # Randomly select file of MgII absorbers, shift in z and MgII line in that file
    # print('\n len(MgII_dir) =',  len(MgII_dir))
    # print(MgII_dir[0])
    # print(MgII_dir[1], '\n')
    if len(MgII_dir) == 1:
        z_file_num = 0
    # if(z_qso < 0.8):
    #     z_file_num = np.random.randint(1, 3)
    else:
        z_file_num = np.random.randint(0, len(MgII_dir))
    # z_file_num = np.random.randint(0, len(MgII_dir))

    # z_shift_max = z_shift_max_arr[z_file_num]
    z_shifted_min, z_shifted_max = z_shifted_range
    # z_shifted = np.random.uniform(-z_shift_max, z_shift_max)
    z_shifted = np.random.uniform(z_shifted_min, z_shifted_max)
    # while z_qso <= z_shifted:
    #     z_shifted = np.random.uniform(z_shifted_min, z_shifted_max)

    # print(z_shifted)
    # print('z_file_num:', z_file_num)
    # print(MgII_dir[z_file_num].keys())
    mask = MgII_dir[z_file_num]['EW_MgII_2796'][:] >= EW_min
    idx_with_absorber = np.where(mask)[0]
    # print('np.where(mask) =', np.where(mask))
    # print('idx_with_absorber', idx_with_absorber)
    # print('len(idx_with_absorber)', len(idx_with_absorber))
    # print(Counter(mask))

    # if len(idx_with_absorber) == 0:
    #     continue
        
    MgII_num = np.random.choice(idx_with_absorber)
    # MgII_num = np.random.randint(0, len(MgII_dir[z_file_num]['flux']))

    MgII_flux = MgII_dir[z_file_num]['flux'][MgII_num]
    wave = MgII_dir[z_file_num]['wave'][:]

    # Shift MgII line based on the random value
    MgII_flux_shifted, z_MgII, mgii_2796_center_pos, mgii_2796_center_wl = shift_metal_abs(wave, MgII_flux, z_shifted, 2796., verbose=verbose)
    
    # if(z_MgII <= z_qso and MgII_dir[z_file_num]['EW_total'][MgII_num] >= EW_min):
    # if z_MgII <= z_qso:

    #     good_z = True
    #     if(verbose==True):
    #         print('good z found')
    #         print('MgII z: ', z_MgII)
    #         print('QSO z: ', z_qso)
    #         print(z_file_num)
    # else:
    #     if(verbose==True):
    #         print('no good z - redo')
    
    # Insert shifted MgII line
    spectrum = insert_metal_abs(qso_spectrum, MgII_flux_shifted)  # 1D flux array
    
    # Create a MgII absorption line property array with EW rescaled to rest frame
    MgII_prop = [MgII_dir[z_file_num]['EW_MgII_2796'][MgII_num]/(1+z_MgII),
                 MgII_dir[z_file_num]['EW_MgII_2803'][MgII_num]/(1+z_MgII),
                #  MgII_dir[z_file_num]['EW_total'][MgII_num]/(1+z_MgII),
                 z_MgII, 
                 mgii_2796_center_pos, 
                 mgii_2796_center_wl]
    
    if saveto is not None:
        qso_template_with_abs = qso_template.copy()
        qso_template_with_abs['FLUX_DENSITY'][:] = spectrum
        qso_template_with_abs.write(saveto, overwrite=True)
    
    return spectrum, MgII_flux_shifted, MgII_prop


def add_MgII_absorber(catalog, MgII_abs, z_shifted_range, #z_shift_max_arr, 
                    *, # ruleset_fname, rules_fname,
                    output_dir='with_MgIIs', template_path='',
                    # airmass=1.2,  # 1.0 - 1.5
                    # seeing=0.8,  # 0.4 - 1.5
                    # moon='gray',
                    # CR_rate=1.67e-7, #l1_type='joined', 
                    N_targets=None):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    warnings.simplefilter('ignore', u.UnitsWarning)
    warnings.simplefilter('ignore', fits.card.VerifyWarning)
    # print("Adding MgII to QSO templates:")


    for num, row in enumerate(catalog, 1):

        template = row['TEMPLATE']
        template_name_no_ext = template[:len(template)-5]
        template_MgII = f'{template_name_no_ext}_with_MgII.fits'
        output = os.path.join(output_dir, f"{template_name_no_ext}_with_MgII.fits")

        if os.path.exists(output):
            pass
        # if False:
        #     pass

        else:

            template_fname = os.path.join(template_path, row['TEMPLATE'])
            # qso_template = Table.read(template_fname)
            # qso_flux = qso_template['FLUX_DENSITY'][:]
            # np.array(qso_flux)
            # hdul = fits.open(template_fname)  # open a FITS file
            # header_ = hdul[0].header

            z_qso = row['REDSHIFT_ESTIMATE']

            if z_qso <= z_shifted_range[0]:
                pass

            else:  # z_qso > z_shifted_range[0]: z_qso at least larger than min of the range
                z_shifted_range = (z_shifted_range[0], z_qso)


            spectrum_t, MgII_flux_shifted_t, MgII_prop = insert_random_MgII(template_fname, #np.array(qso_flux), 
                                                                            MgII_abs, 
                                                                            # z_shift_max_arr, 
                                                                            z_shifted_range, 
                                                                            row['REDSHIFT_ESTIMATE'], 
                                                                            saveto=output, 
                                                                            verbose=False)

            catalog['has_MgII'][num-1] = True

            catalog['EW_MgII_2796'][num-1] = MgII_prop[0]
            # print('EW_MgII_2796 =', MgII_prop[0])
            catalog['EW_MgII_2803'][num-1] = MgII_prop[1]
            catalog['z_MgII'][num-1] = MgII_prop[2]

            catalog['MgII_2796_center_pos'] = MgII_prop[3]
            # print('MgII_2796_center_wl =', MgII_prop[4])
            catalog['MgII_2796_center_wl'] = MgII_prop[4]
            
            catalog['TEMPLATE_with_MgII'][num-1] = template_MgII

                # MgII_prop = [MgII_dir[z_file_num]['EW_MgII_2796'][MgII_num]/(1+z_MgII),
                #  MgII_dir[z_file_num]['EW_MgII_2803'][MgII_num]/(1+z_MgII),
                #  z_MgII, 
                #  mgii_2796_center_pos, 
                #  mgii_2796_center_wl]

        # progress
        # if (num / len(catalog)) * 100 % 5.0 == 0.0:  # every 5%
        #     sys.stdout.write(f"\r{100*num/len(catalog)}% done \n")
        sys.stdout.flush()
    return catalog


def process_chunk(chunk_data):
    """Helper function to process a single chunk of catalog data"""
    chunk_cat, MgII_abs, z_shifted_range, output_dir, template_path, chunk_idx, num_chunks, start_idx, end_idx = chunk_data
    
    print(f"\nProcessing chunk {chunk_idx+1}/{num_chunks} (QSOs {start_idx+1}-{end_idx})")
    
    chunk_cat_mgii = add_MgII_absorber(chunk_cat,
                      MgII_abs, 
                      z_shifted_range, 
                      output_dir=output_dir,
                      template_path=template_path,
                      N_targets=len(chunk_cat))
    
    return chunk_cat_mgii.to_pandas()


def main():
    parser = ArgumentParser(description='Insert MgII absorbers in QSO templates')
    parser.add_argument('-n', '--number', type=int, default=None)
    parser.add_argument('--temp-dir', type=str, default='/data2/home2/nguerrav/QSO_simpaqs/QSOs_full_cat/', help='Directory of spectral templates')
    parser.add_argument("-o", "--output", type=str, default='/data2/home2/nguerrav/QSO_simpaqs/QSOs_full_cat_with_absorbers_in_blue_arm/', help="output directory")
    parser.add_argument('--n-cores', type=int, default=None, help='Number of CPU cores to use for parallel processing (default: 75% of available cores)')

    args = parser.parse_args()

    t1 = datetime.datetime.now()
    # catalog = Table.read('/data2/home2/nguerrav/Catalogues/ByCycle_Final_Cat_fobs_qso_templates_with_SNR_golden_label.fits')
    catalog = Table.read('/data2/home2/nguerrav/Catalogues/test_set_cat_not_in_golden_sample.fits')  # not in training set
# print(Counter(catalog['golden']))
    # catalog = catalog[:10]
    # print(Counter(catalog['golden']))
    # catalog = catalog[catalog['golden']==False][:]

    catalog['has_MgII'] = False
    catalog['z_MgII'] = -999
    catalog['EW_MgII_2796'] = -999
    catalog['EW_MgII_2803'] = -999
    catalog['TEMPLATE_with_MgII'] = ' ' * 57
    catalog['EW_MgII_total'] = -999
    catalog['MgII_2796_center_pos'] = -999
    catalog['MgII_2796_center_wl'] = -999

    MgII_abs = load_MgII('/data2/home2/nguerrav/TNG50_spec/')
    arm = 'blue'
    # arm = 'green'
    # arm = 'red'
    # absorbers_z_05 = h5py.File('/data2/home2/nguerrav/TNG50_spec/spectra_TNG50-1_z0.5_n2000d2-rndfullbox_4MOST-HRS_MgII_combined.hdf5', 'r')
    # z_shift_max_arr = [0.395, 0.558]

    if arm == 'blue':
        z_min = (3926 - 2796) / 2796
        z_max = (4355 - 2803) / 2803
        z_shifted_range = [z_min, z_max]
    elif arm == 'green':
        z_min = (5160 - 2796) / 2796
        z_max = (5730 - 2803) / 2803
        z_shifted_range = [z_min, z_max]
    if arm == 'red':
        z_min = (6100 - 2796) / 2796
        z_max = (6790 - 2803) / 2803
        z_shifted_range = [z_min, z_max]

    if args.number is not None:
        N_targets = args.number
        idx = np.random.choice(np.arange(len(catalog)), N_targets, replace=False)
        catalog = catalog[idx]
    else:
        N_targets = len(catalog)

    if N_targets > 50000:

        chunk_size = 10000
        num_chunks = (N_targets + chunk_size - 1) // chunk_size
        
        chunk_data_list = []
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, N_targets)
            chunk_cat = catalog[start_idx:end_idx]
            
            chunk_data = (chunk_cat, MgII_abs, z_shifted_range, args.output, 
                         args.temp_dir, chunk_idx, num_chunks, start_idx, end_idx)
            chunk_data_list.append(chunk_data)
        
        # Use multiprocessing to process chunks in parallel
        # Use 75% of available CPU cores by default to avoid overwhelming the system
        if args.n_cores is not None:
            n_cores = max(1, args.n_cores)
        else:
            n_cores = max(1, int(cpu_count() * 0.75))
        print(f"Processing {num_chunks} chunks using {n_cores} CPU cores")
        
        with Pool(processes=n_cores) as pool:
            chunks_cat_mgii = pool.map(process_chunk, chunk_data_list)

        catalog_mgii = pd.concat(chunks_cat_mgii, ignore_index=True)
        save_to_fits(catalog_mgii, 
                 '/data2/home2/nguerrav/Catalogues/test_set_cat_not_in_golden_sample_with_MgII.fits')

    else:

        catalog_mgii = add_MgII_absorber(catalog,
                          MgII_abs, 
                        #   z_shift_max_arr, 
                        z_shifted_range, 
                output_dir=args.output,
                template_path=args.temp_dir,
                N_targets=N_targets)
        
        # save_to_fits(catalog_mgii, 
        #              '/data2/home2/nguerrav/Catalogues/ByCycle_Cat_test_set_with_MgII.fits')
        catalog_mgii.write('/data2/home2/nguerrav/Catalogues/test_set_cat_not_in_golden_sample_with_MgII.fits', format='fits', overwrite=True)

    t2 = datetime.datetime.now()
    dt = t2 - t1
    
    print(f"Finished inserting MgII absorbers in {N_targets} QSO templates in {dt.total_seconds():.1f} seconds")

if __name__ == '__main__':
    main()
