from astropy.io import fits
from astropy.table import Table, vstack
from astropy.time import Time
import astropy.units as u
from argparse import ArgumentParser

import os
import sys
import datetime
import pandas as pd
import h5py
import numpy as np

from lmfit.models import VoigtModel

from os import listdir
from os.path import isfile, join

import warnings
warnings.filterwarnings('ignore', message='invalid value encountered in divide', category=RuntimeWarning)
# warnings.filterwarnings('default')

from qmostetc import SEDTemplate, QMostObservatory, Ruleset, Rule, Filter, L1DXU

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



def shift_metal_abs(wave_mgii, flux_mgii, z_shift, metal_cent, verbose=False):
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
    z_shifted = z_mgii + z_shift
    
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


def insert_random_MgII(qso_template, MgII_dir, z_shift_max_arr, z_qso, saveto, EW_min=0.002, verbose=False):
    
    ''' Function to insert a random MgII absorption line into the spectrum
    Input:
        qso_spectrum: QSO spectrum
        MgII_dir: MgII array with MgII lines at different redshifts (use load_MgII to get the right format)
        z_shift_max: maximum range of shifting the MgII line. This is done to get a continuous 
                     function of MgII absorber in z-space
    Returns:
        spectrum: QSO spectrum with added MgII absorption line
        MgII_flux_shifted: MgII absorption spectrum shifted to a specific redshift
        MgII_prop: Properties of the MgII line. EW are rescaled to rest-frame 
    '''

    qso_template = Table.read(qso_template)
    qso_flux = qso_template['FLUX_DENSITY'][:]
    qso_spectrum = np.asarray(qso_flux)
    
    good_z = False
    while(good_z == False):
        # Randomly select file of MgII absorbers, shift in z and MgII line in that file
        if(z_qso < 0.6):
            z_file_num = 1
        if(z_qso < 0.8):
            z_file_num = np.random.randint(1, 3)
        else:
            z_file_num = np.random.randint(0, len(MgII_dir))
            
        z_shift_max = z_shift_max_arr[z_file_num]
        z_shift = np.random.uniform(-z_shift_max, z_shift_max)
        print(z_shift)
        MgII_num = np.random.randint(0, len(MgII_dir[z_file_num]['flux']))

        MgII_flux = MgII_dir[z_file_num]['flux'][MgII_num]

        # Shift MgII line based on the random value
        MgII_flux_shifted, z_MgII, mgii_2796_center_pos, mgii_2796_center_wl = shift_metal_abs(MgII_dir[z_file_num]['wave'][:], MgII_flux, z_shift, 2796., verbose=verbose)
        
        # if(z_MgII <= z_qso and MgII_dir[z_file_num]['EW_total'][MgII_num] >= EW_min):
        if z_MgII <= z_qso:

            good_z = True
            if(verbose==True):
                print('good z found')
                print('MgII z: ', z_MgII)
                print('QSO z: ', z_qso)
                print(z_file_num)
        else:
            if(verbose==True):
                print('no good z - redo')
    
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


def add_MgII_absorber(catalog, MgII_abs, z_shift_max_arr, *, # ruleset_fname, rules_fname,
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
    print("Adding MgII to QSO templates:")


    for num, row in enumerate(catalog, 1):

        template = row['TEMPLATE']
        template_name_no_ext = template[:len(template)-5]
        template_MgII = f'{template_name_no_ext}_with_MgII.fits'
        output = os.path.join(output_dir, f"{template_name_no_ext}_with_MgII.fits")

        if os.path.exists(output):
            pass

        else:

            template_fname = os.path.join(template_path, row['TEMPLATE'])
            # qso_template = Table.read(template_fname)
            # qso_flux = qso_template['FLUX_DENSITY'][:]
            # np.array(qso_flux)
            # hdul = fits.open(template_fname)  # open a FITS file
            # header_ = hdul[0].header

            spectrum_t, MgII_flux_shifted_t, MgII_prop = insert_random_MgII(template_fname, #np.array(qso_flux), 
                                                                            MgII_abs, 
                                                                            z_shift_max_arr, 
                                                                            row['REDSHIFT_ESTIMATE'], 
                                                                            saveto=output, 
                                                                            verbose=False)

            catalog['has_MgII'][num-1] = True

            catalog['EW_MgII_2796'][num-1] = MgII_prop[0]
            catalog['EW_MgII_2803'][num-1] = MgII_prop[1]
            catalog['z_MgII'][num-1] = MgII_prop[2]

            catalog['MgII_2796_center_pos'] = MgII_prop[3]
            catalog['MgII_2796_center_wl'] = MgII_prop[4]
            
            catalog['TEMPLATE_with_MgII'][num-1] = template_MgII

                # MgII_prop = [MgII_dir[z_file_num]['EW_MgII_2796'][MgII_num]/(1+z_MgII),
                #  MgII_dir[z_file_num]['EW_MgII_2803'][MgII_num]/(1+z_MgII),
                #  z_MgII, 
                #  mgii_2796_center_pos, 
                #  mgii_2796_center_wl]

        # progress
        if (num / len(catalog)) * 100 % 5.0 == 0.0:  # every 5%
            sys.stdout.write(f"\r{100*num/len(catalog)}% done \n")
        sys.stdout.flush()


def main():
    parser = ArgumentParser(description='Insert MgII absorbers in QSO templates')
    parser.add_argument('-n', '--number', type=int, default=None)
    parser.add_argument('--temp-dir', type=str, default='/data2/home2/nguerrav/QSO_simpaqs/QSOs_full_cat/', help='Directory of spectral templates')
    parser.add_argument("-o", "--output", type=str, default='/data2/home2/nguerrav/QSO_simpaqs/QSOs_full_cat_with_absorbers/', help="output directory")

    args = parser.parse_args()

    t1 = datetime.datetime.now()
    catalog = Table.read('/data2/home2/nguerrav/Catalogues/ByCycle_Final_Cat_fobs_qso_templates_with_SNR_golden_label.fits')
    catalog = catalog[:10]

    catalog['has_MgII'] = False
    catalog['z_MgII'] = -999
    catalog['EW_MgII_2796'] = -999
    catalog['EW_MgII_2803'] = -999
    catalog['TEMPLATE_with_MgII'] = ''
    catalog['EW_MgII_total'] = -999
    catalog['MgII_2796_center_pos'] = -999
    catalog['MgII_2796_center_wl'] = -999

    MgII_abs = load_MgII('/data2/home2/nguerrav/TNG50_spec/')
    # absorbers_z_05 = h5py.File('/data2/home2/nguerrav/TNG50_spec/spectra_TNG50-1_z0.5_n2000d2-rndfullbox_4MOST-HRS_MgII_combined.hdf5', 'r')
    z_shift_max_arr = [0.395, 0.558]

    if args.number is not None:
        N_targets = args.number
        idx = np.random.choice(np.arange(len(catalog)), N_targets, replace=False)
        catalog = catalog[idx]
    else:
        N_targets = len(catalog)

    if N_targets > 50000:
        # Process in chunks to manage memory
        chunk_size = 10000
        num_chunks = (N_targets + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, N_targets)

            print(f"\nProcessing chunk {chunk_idx+1}/{num_chunks} (QSOs {start_idx+1}-{end_idx})")
            
            chunk_cat = catalog[start_idx:end_idx]

            add_MgII_absorber(chunk_cat,
                              MgII_abs, 
                              z_shift_max_arr, 
                            output_dir=args.output,
                            template_path=args.temp_dir,
                            N_targets=N_targets,
                            )

    else:

        add_MgII_absorber(catalog,
                          MgII_abs, 
                          z_shift_max_arr, 
                output_dir=args.output,
                template_path=args.temp_dir,
                N_targets=N_targets)

    t2 = datetime.datetime.now()
    dt = t2 - t1
    
    print(f"Finished inserting MgII absorbers in {N_targets} QSO templates in {dt.total_seconds():.1f} seconds")

if __name__ == '__main__':
    main()
