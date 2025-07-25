"""
Script to simulate the exposure times and mock reduced 4MOST spectra
for a given input target catalog and a set of spectral rules and rulesets.
"""

__author__ = 'JK Krogager'

from astropy.io import fits
from astropy.table import Table, vstack
from astropy.time import Time
import astropy.units as u
from argparse import ArgumentParser
import numpy as np
import warnings
import os
import sys
import datetime
import pandas as pd
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore', message='invalid value encountered in divide', category=RuntimeWarning)
# warnings.filterwarnings('default')

from qmostetc import SEDTemplate, QMostObservatory, Ruleset, Rule, Filter, L1DXU


def load_rulesets(qmost, ruleset_fname, rules_fname):
    rules = Rule.read(qmost, rules_fname)
    ruleset_list = Ruleset.read(ruleset_fname)
    rulesets = {}
    for rs in ruleset_list:
        rs.set_rules(rules)
        rulesets[rs.name] = rs
    return rulesets


def update_header(hdu_list, target, prog_id='4MOST-ETC'):
    specuid = np.random.randint(10000000)
    hdu_list[0].header['OBID'] = 101
    hdu_list[0].header['OBID1'] = 101
    hdu_list[0].header['ESO TEL AIRM END'] = target['AIRMASS']
    hdu_list[0].header['ESO TEL AIRM START'] = target['AIRMASS']
    hdu_list[0].header['ESO TEL AMBI FWHM END'] = target['SEEING']
    hdu_list[0].header['ESO TEL AMBI FWHM START'] = target['SEEING']
    hdu_list[0].header['ESO TEL AMBI MOON'] = target['MOON']
    hdu_list[0].header['PROG_ID'] = prog_id
    hdu_list[0].header['MJD-OBS'] = Time.now().mjd
    hdu_list[0].header['MJD-END'] = Time.now().mjd
    hdu_list[0].header['OBJ_UID'] = hash(target['NAME'])
    hdu_list[0].header['OBJ_NME'] = target['NAME']
    hdu_list[1].header['OBJ_UID'] = hash(target['NAME'])
    hdu_list[1].header['OBJ_NME'] = target['NAME']
    hdu_list[0].header['TRG_UID'] = hash(target['NAME'])
    hdu_list[0].header['TRG_NME'] = target['NAME']
    hdu_list[1].header['TRG_UID'] = hash(target['NAME'])
    hdu_list[1].header['TRG_NME'] = target['NAME']
    hdu_list[1].header['TRG_MAG'] = target['MAG']
    hdu_list[0].header['SPECUID'] = specuid
    hdu_list[1].header['SPECUID'] = specuid
    hdu_list[1].header['TRG_Z'] = target['REDSHIFT_ESTIMATE']
    hdu_list[1].header['TRG_TMP'] = os.path.basename(target['TEMPLATE_with_MgII'])
    # hdu_list[1].header['TRG_TMP'] = os.path.basename(target['TEMPLATE'])
    return hdu_list


def process_catalog(catalog, *, ruleset_fname, rules_fname,
                    output_dir='l1_data', template_path='',
                    # airmass=1.2,  # 1.0 - 1.5
                    # seeing=0.8,  # 0.4 - 1.5
                    moon='gray',
                    CR_rate=1.67e-7, #l1_type='joined', 
                    N_targets=None,
                    prog_id='4MOST-ETC', t_min=20*u.min, t_max=1e9*u.min):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    spectrograph = 'hrs'
    qmost = QMostObservatory(spectrograph)
    rulesets = load_rulesets(qmost, ruleset_fname, rules_fname)

    catalog['MOON'] = moon
    catalog['SEEING'] = np.random.normal(0.8, 0.3, len(catalog))
    catalog['AIRMASS'] = np.random.uniform(1.0, 2.5, len(catalog))

    # alt = np.arccos(1. / airmass) * 180 / np.pi * u.deg
    # obs = qmost(alt, seeing*u.arcsec, moon)

    warnings.simplefilter('ignore', u.UnitsWarning)
    warnings.simplefilter('ignore', fits.card.VerifyWarning)
    print("Applying 4MOST ETC to the catalog:")

    exptime_log = []

    for num, row in enumerate(catalog, 1):

        z_str = str(np.round(row['REDSHIFT_ESTIMATE'], 4))
        mag_str = str(np.round(row['MAG'], 2))
        ruleset_name = row['RULESET']
        target_name = row['NAME']
        model_id = f'QSO_sim_ETC_z{z_str}_mag{mag_str}_{target_name}'
        # print(model_id, '\n')
        # output = os.path.join(output_dir, f"{model_id}_LJ1.fits")
        output = os.path.join(output_dir, f"{model_id}_LJ1_MgII.fits")

        if os.path.exists(output) or len(row['TEMPLATE_with_MgII']) < 10:
            pass  # spectrum already simulated

        else:
            row['MOON'] = moon
            seeing = row['SEEING']
            airmass = row['AIRMASS']

            alt = np.arccos(1. / airmass) * 180 / np.pi * u.deg
            obs = qmost(alt, seeing*u.arcsec, moon)

            ruleset = rulesets[ruleset_name]
            etc = ruleset.etc(alt, seeing*u.arcsec, moon)
            template_fname = os.path.join(template_path, row['TEMPLATE_with_MgII'])
            # template_fname = os.path.join(template_path, row['TEMPLATE'])
            
            # try:
            SED = SEDTemplate(template_fname)
            # except:
            #     # warning_msg = f"Skipping QSO {template_fname}"
            #     # print(warning_msg)
            #     # print(template_wave_min, template_wave_max)
            #     # warning_file = os.path.join('./', 'simulate_catalog_warnings.log')
            #     # with open(warning_file, 'a') as f:
            #     #     timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            #     #     f.write(f"[{timestamp}] Target: {target_name}, z={row['REDSHIFT_ESTIMATE']}, MAG={row['MAG']}, fobs={row['fobs']}, {warning_msg}\n")
            #     # warning_target_list_file = os.path.join('./', 'simulate_catalog_warnings_skipped_target_list.log')
            #     # with open(warning_target_list_file, 'a') as f:
            #     #     f.write(f"{target_name}\n")
            #     pass

            template_wave_min = SED.wavelength.min().value
            template_wave_max = SED.wavelength.max().value
            if template_wave_max < 678.0 or template_wave_min > 392.7:
                warning_msg = f"May not fully cover 4MOST range (3700-9500 Å)"
                print(warning_msg)
                print(template_wave_min, template_wave_max)
                warning_file = os.path.join('./', 'simulate_catalog_warnings.log')
                with open(warning_file, 'a') as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] Target: {target_name}, z={row['REDSHIFT_ESTIMATE']}, MAG={row['MAG']}, fobs={row['fobs']}, {warning_msg}\n")
                warning_target_list_file = os.path.join('./', 'simulate_catalog_warnings_target_list.log')
                with open(warning_target_list_file, 'a') as f:
                    f.write(f"{target_name}\n")


            # Add the target spectrum from the template with a magnitude
            mag_type_str = 'DECam_z_AB'  # row['MAG_TYPE']
            survey, band, ab_vega = mag_type_str.split('_')
            mag_type = [filt_id for filt_id in Filter.list()
                        if survey.upper() in filt_id.upper() and '.'+band in filt_id][0]
            mag_unit = u.ABmag
            # if ab_vega != 'AB':
            #     print("Warning not AB magnitude in catalog... may be incorrect")
            mag = row['MAG'] * mag_unit
            # SED = SED.redshift(row['REDSHIFT_ESTIMATE'])
            # etc.set_target(SED(mag, mag_type), 'point')
            obs.set_target(SED(mag, mag_type), 'point')

            # Get and print the exposure time
            texp_col = 'texp_' + moon[0]
            # print(texp_col)
            if texp_col in catalog.colnames:
                texp = row[texp_col] * u.min
            else:
                etc.set_target(SED(mag, mag_type), 'point')
                texp = etc.get_exptime()
            if texp < t_min:
                texp = t_min
            if texp > t_max:
                texp = t_max

            if 'fobs' in catalog.colnames:

                texp_fobs = row['fobs'] * texp
                exptime_log.append({'NAME': target_name, 'MAG': row['MAG'],
                                'TEXP': texp, 'fobs':row['fobs'], 'TEXP_fobs':texp_fobs, 
                                'REDSHIFT': row['REDSHIFT_ESTIMATE'], 
                                'SUBSURVEY': row['SUBSURVEY'], 'SEEING': seeing, 'AIRMASS': airmass})
            else:
                exptime_log.append({'NAME': target_name, 'MAG': row['MAG'],
                                'TEXP': texp, 
                                'REDSHIFT': row['REDSHIFT_ESTIMATE'], 
                                'SUBSURVEY': row['SUBSURVEY'], 'SEEING': seeing, 'AIRMASS': airmass})

            res = obs.expose(texp)  # 'wavelength', 'binwidth', 'efficiency', 'gain', , 'target', 'sky', 'dark', 'ron', 'noise'

            # if np.isnan(res['target']).any():
            #     res['target'][np.isnan(res['target'])] = 0.
            # if np.isnan(res['sky']).any():
            #     res['sky'][np.isnan(res['sky'])] = 0.
            res = obs.expose(texp)

            flux_floor = 1e-25  # Very small positive value
            if np.isnan(res['target']).any():
                nan_mask = np.isnan(res['target'])
                res['target'][nan_mask] = flux_floor * u.electron
                warning_msg = f"Warning: {np.sum(nan_mask)} NaN values in target flux replaced with floor value"
                print(warning_msg)
                warning_file = os.path.join('./', 'simulate_catalog_warnings.log')
                with open(warning_file, 'a') as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] Target: {target_name}, z={row['REDSHIFT_ESTIMATE']}, MAG={row['MAG']}, fobs={row['fobs']}, {warning_msg}\n")

                warning_target_list_file = os.path.join('./', 'simulate_catalog_warnings_target_list.log')
                with open(warning_target_list_file, 'a') as f:
                    f.write(f"{target_name}\n")

            if np.isnan(res['sky']).any():
                nan_mask = np.isnan(res['sky'])
                res['sky'][nan_mask] = flux_floor * u.electron
                warning_msg = f"Warning: {np.sum(nan_mask)} NaN values in sky flux replaced with floor value"
                print(warning_msg)
                warning_file = os.path.join('./', 'simulate_catalog_warnings.log')
                with open(warning_file, 'a') as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] Target: {target_name}, z={row['REDSHIFT_ESTIMATE']}, MAG={row['MAG']}, fobs={row['fobs']}, {warning_msg}\n")

                warning_target_list_file = os.path.join('./', 'simulate_catalog_warnings_target_list.log')
                with open(warning_target_list_file, 'a') as f:
                    f.write(f"{target_name}\n")

            # Also handle the noise component
            if np.isnan(res['noise']).any():
                nan_mask = np.isnan(res['noise'])
                res['noise'][nan_mask] = np.sqrt(flux_floor) * u.electron  # Reasonable error for floor flux
                warning_msg = f"Warning: {np.sum(nan_mask)} NaN values in noise flux replaced with floor value"
                print(warning_msg)
                warning_file = os.path.join('./', 'simulate_catalog_warnings.log')
                with open(warning_file, 'a') as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] Target: {target_name}, z={row['REDSHIFT_ESTIMATE']}, MAG={row['MAG']}, fobs={row['fobs']}, {warning_msg}\n")
            
                warning_target_list_file = os.path.join('./', 'simulate_catalog_warnings_target_list.log')
                with open(warning_target_list_file, 'a') as f:
                    f.write(f"{target_name}\n")

            etc_wave_min = np.min(res['wavelength'])
            etc_wave_max = np.max(res['wavelength'])
            if etc_wave_max.value < 678 or etc_wave_min.value > 393:  # in nm
                warning_msg = f"May not fully cover 4MOST range"
                print(warning_msg)
                print(etc_wave_min, etc_wave_max, '\n')
                warning_file = os.path.join('./', 'simulate_catalog_warnings.log')
                with open(warning_file, 'a') as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] Target: {target_name}, z={row['REDSHIFT_ESTIMATE']}, MAG={row['MAG']}, fobs={row['fobs']}, {warning_msg}\n")

                warning_target_list_file = os.path.join('./', 'simulate_catalog_warnings_target_list.log')
                with open(warning_target_list_file, 'a') as f:
                    f.write(f"{target_name}\n")

            # Add cosmic rays:
            N_pix = len(res)
            N_cosmic = np.random.poisson(CR_rate * texp.value * N_pix * 0.8)
            idx = np.random.choice(np.arange(N_pix), N_cosmic, replace=False)
            CR_boost = 10**np.random.normal(2.0, 0.15, N_cosmic) * u.electron
            res['target'][idx] += CR_boost
            res['noise'][idx] = np.sqrt(res['noise'][idx]**2 + CR_boost * u.electron)

            dxu = L1DXU(qmost, res, texp)

            # Write individual L1 files
            # if l1_type[0].upper() == 'A':
            #     for arm_name in qmost.keys():
            #         INST = 'L' if spectrograph == 'lrs' else 'H'
            #         INST += arm_name.upper()[0]
            #         # INST += '1'

            #         z_str = str(np.round(row['REDSHIFT_ESTIMATE'], 4))
            #         model_id = f'QSO_sim_ETC_z{z_str}_{target_name}'
            #         output_arm = os.path.join(output_dir, f'{model_id}_{INST}.fits')  # saves fluxin ADU
            #         try:
            #             hdu_list = dxu.per_arm(arm_name)
            #             hdu_list = update_header(hdu_list, row)
            #             hdu_list.writeto(output_arm, overwrite=True)
            #         except ValueError as e:
            #             print(f"Failed to save the spectrum: {row['TEMPLATE']}")
            #             print(f"for arm: {arm_name}")

            # if spectrograph.lower() == 'lrs':
            # Create JOINED L1 SPECTRUM:

            # try:
            hdu_list = dxu.joined()
            
            # Set flux floor and clean up data
            flux_floor = 1e-25
            flux_data = hdu_list[1].data['FLUX']
            err_data = hdu_list[1].data['ERR_FLUX']
            
            # Count issues before fixing
            n_negative = np.sum(flux_data < 0)
            n_zero = np.sum(flux_data == 0)
            n_nan_flux = np.sum(np.isnan(flux_data))
            n_nan_err = np.sum(~np.isfinite(err_data))
            
            # if n_negative + n_zero + n_nan_flux + n_nan_err > 0:
            #     print(f"Cleaning spectrum {model_id}: {n_negative} negative, {n_zero} zero, {n_nan_flux} NaN flux, {n_nan_err} bad errors")
            
            # Fix flux issues
            bad_flux_mask = (flux_data <= 0) | np.isnan(flux_data)
            flux_data[bad_flux_mask] = flux_floor
            
            # Fix error issues
            bad_err_mask = ~np.isfinite(err_data) | (err_data <= 0)
            err_data[bad_err_mask] = np.sqrt(flux_floor) * 10  # Conservative error
            
            # Update HDU data
            hdu_list[1].data['FLUX'] = flux_data
            hdu_list[1].data['ERR_FLUX'] = err_data
            
            hdu_list = update_header(hdu_list, row, prog_id)
            hdu_list.writeto(output, overwrite=True)
                
            # except (IndexError, ValueError) as e:
            #     error_file = os.path.join(output_dir, 'failed_spectra.txt')
            #     error_message = f"Failed to save spectrum for {model_id}: {str(e)}"
            #     with open(error_file, 'a') as f:
            #         timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            #         f.write(f"[{timestamp}] {error_message}\n")
            #     print(error_message)
            #     continue  # Skip to next target

            # hdu_list = dxu.joined()
            # hdu_list = update_header(hdu_list, row, prog_id)
            # hdu_list.writeto(output, overwrite=True)

        if (num / len(catalog)) * 100 % 5.0 == 0.0:  # every 5%
            # sys.stdout.write(f"\r{num}/{len(catalog)} \n")
            sys.stdout.write(f"\r{100*num/len(catalog)}% done \n")
        sys.stdout.flush()
    # exptimes = Table(exptime_log)
    exptimes = pd.DataFrame(exptime_log)
    # exptimes.meta['comments'] = ['Exposure times in seconds']
    log_fname = os.path.join(output_dir, 'exposure_times.csv')
    # print('type(exptime_log):', type(exptime_log))
    # print('exptime_log:', exptime_log)
    # print(exptimes)
    
    if os.path.exists(log_fname):
        exptimes_prev = pd.read_csv(log_fname)  # Table.read(log_fname)
        # exptimes = vstack([exptimes_prev, exptimes])
        exptimes = pd.concat([exptimes_prev, exptimes], axis=0, ignore_index=True)
    else:
        # exptimes = exptimes.data
        pass

    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     exptimes.write(log_fname,
    #                formats={'TEXP': '%.1f', 'TEXP_fobs': '%.1f', 
    #                         'MAG': '%.2f', 'REDSHIFT': '%.4f'},
    #                overwrite=True, comment='# ', format='csv')
    exptimes['TEXP'] = exptimes['TEXP'].round(1)
    exptimes['MAG'] = exptimes['MAG'].round(2)
    exptimes['REDSHIFT'] = exptimes['REDSHIFT'].round(4)
    if 'TEXP_fobs' in exptimes.columns:
        exptimes['TEXP_fobs'] = exptimes['TEXP_fobs'].round(1)
    exptimes.to_csv(log_fname, index=False)
    print(' ')

def process_chunk_wrapper(chunk_data):
    """Helper function to process a single chunk of catalog data"""
    chunk_cat, ruleset_fname, rules_fname, output_dir, template_path, moon, prog_id, chunk_idx, num_chunks, start_idx, end_idx = chunk_data
    
    print(f"\nProcessing chunk {chunk_idx+1}/{num_chunks} (QSOs {start_idx+1}-{end_idx})")
    
    process_catalog(chunk_cat,
                    ruleset_fname=ruleset_fname,
                    rules_fname=rules_fname,
                    output_dir=output_dir,
                    template_path=template_path,
                    moon=moon,
                    N_targets=len(chunk_cat),
                    prog_id=prog_id)
    
    return f"Chunk {chunk_idx+1}/{num_chunks} completed"

def main():
    parser = ArgumentParser(description="Generate simulated spectra from 4MOST Target Catalog")
    # parser.add_argument("input", type=str, help="input target FITS catalog", 
    #                     # default='/data2/home2/nguerrav/QSO_simpaqs/ByCycle_Final_Cat_qso_templates_fobs_notna.fits'
    #                     )
    parser.add_argument('--airmass', type=float, default=1.2)
    parser.add_argument('--moon', type=str, default='gray', choices=['dark', 'gray', 'bright'])
    parser.add_argument('--seeing', type=float, default=0.8)
    parser.add_argument('-n', '--number', type=int, default=None)
    parser.add_argument('--rules', type=str, default='./../S17_20250122T1441Z_rules.csv', help='Rules definition (FITS or CSV)')
    parser.add_argument('--ruleset', type=str, default='./../S17_20250122T1443Z_rulesets.csv', help='Ruleset definition (FITS or CSV)')
    parser.add_argument('--temp-dir', type=str, default='/data2/home2/nguerrav/QSO_simpaqs/QSOs_full_cat_with_absorbers_in_blue_arm/', help='Directory of spectral templates')
    parser.add_argument("-o", "--output", type=str, default='/data2/home2/nguerrav/QSO_simpaqs/QSOs_full_cat_with_absorbers_in_blue_arm_ETC_L1_output_with_fobs/', help="output directory")
    parser.add_argument('--n-cores', type=int, default=None, help='Number of CPU cores to use for parallel processing (default: 75% of available cores)')
    # parser.add_argument('--arm', type=str, default='ALL', choices=['J', 'joined', 'ALL', 'a'])
    parser.add_argument('--prog', type=str, default='4MOST-ETC',
                        help="Determines the PROG_ID header keyword")

    args = parser.parse_args()

    t1 = datetime.datetime.now()
    catalog = Table.read('/data2/home2/nguerrav/Catalogues/test_set_cat_not_in_golden_sample_SNR_3_with_MgII.fits')

    if args.number is not None:
        N_targets = args.number
        idx = np.random.choice(np.arange(len(catalog)), N_targets, replace=False)
        catalog = catalog[idx]
    else:
        N_targets = len(catalog)

    if N_targets > 50000:
        # Process in chunks using parallel processing
        chunk_size = 10000
        num_chunks = (N_targets + chunk_size - 1) // chunk_size
        
        chunk_data_list = []
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, N_targets)
            chunk_cat = catalog[start_idx:end_idx]
            
            chunk_data = (chunk_cat, args.ruleset, args.rules, args.output, 
                         args.temp_dir, args.moon, args.prog, 
                         chunk_idx, num_chunks, start_idx, end_idx)
            chunk_data_list.append(chunk_data)
        
        # Use multiprocessing to process chunks in parallel
        # Use 75% of available CPU cores by default to avoid overwhelming the system
        if args.n_cores is not None:
            n_cores = max(1, args.n_cores)
        else:
            n_cores = max(1, int(cpu_count() * 0.75))
        print(f"Processing {num_chunks} chunks using {n_cores} CPU cores")
        
        with Pool(processes=n_cores) as pool:
            results = pool.map(process_chunk_wrapper, chunk_data_list)
        
        print("All chunks completed successfully!")
        for result in results:
            print(result)

    else:
        process_catalog(catalog,
                ruleset_fname=args.ruleset,
                rules_fname=args.rules,
                output_dir=args.output,
                template_path=args.temp_dir,
                # airmass=args.airmass,
                # seeing=args.seeing,
                moon=args.moon,
                # l1_type=args.arm,
                N_targets=N_targets,
                prog_id=args.prog,
                )

    t2 = datetime.datetime.now()
    dt = t2 - t1
    
    print(f"Finished simulation of {N_targets} targets in {dt.total_seconds():.1f} seconds")

if __name__ == '__main__':
    main()
