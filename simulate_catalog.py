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
    hdu_list[1].header['TRG_TMP'] = os.path.basename(target['TEMPLATE'])
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
    if N_targets:
        idx = np.random.choice(np.arange(len(catalog)), N_targets, replace=False)
        catalog = catalog[idx]

    for num, row in enumerate(catalog, 1):

        z_str = str(np.round(row['REDSHIFT_ESTIMATE'], 4))
        mag_str = str(np.round(row['MAG'], 2))
        ruleset_name = row['RULESET']
        target_name = row['NAME']
        model_id = f'QSO_sim_ETC_z{z_str}_mag{mag_str}_{target_name}'
        print(model_id, '\n')
        output = os.path.join(output_dir, f"{model_id}_LJ1.fits")

        if os.path.exists(output):
            pass  # spectrum already simulated

        else:
            row['MOON'] = moon
            seeing = row['SEEING']
            airmass = row['AIRMASS']

            alt = np.arccos(1. / airmass) * 180 / np.pi * u.deg
            obs = qmost(alt, seeing*u.arcsec, moon)

            ruleset = rulesets[ruleset_name]
            etc = ruleset.etc(alt, seeing*u.arcsec, moon)
            template_fname = os.path.join(template_path, row['TEMPLATE'])
            SED = SEDTemplate(template_fname)

            # Add the target spectrum from the template with a magnitude
            mag_type_str = row['MAG_TYPE']
            survey, band, ab_vega = mag_type_str.split('_')
            mag_type = [filt_id for filt_id in Filter.list()
                        if survey.upper() in filt_id.upper() and '.'+band in filt_id][0]
            mag_unit = u.ABmag
            # if ab_vega != 'AB':
            #     print("Warning not AB magnitude in catalog... may be incorrect")
            mag = row['MAG'] * mag_unit
            # SED = SED.redshift(row['REDSHIFT_ESTIMATE'])
            etc.set_target(SED(mag, mag_type), 'point')
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
                exptime_log.append({'NAME': target_name, 'MAG': row['MAG'],
                                'TEXP': texp, 'fobs':row['fobs'], 'REDSHIFT': row['REDSHIFT_ESTIMATE'], 
                                'SUBSURVEY': row['SUBSURVEY'], 'SEEING': seeing, 'AIRMASS': airmass})
                # print(row['fobs'])
                # print(texp)
                texp = row['fobs'] * texp
                # print(texp)
            else:
                exptime_log.append({'NAME': target_name, 'MAG': row['MAG'],
                                'TEXP': texp, 'REDSHIFT': row['REDSHIFT_ESTIMATE'], 
                                'SUBSURVEY': row['SUBSURVEY'], 'SEEING': seeing, 'AIRMASS': airmass})

            res = obs.expose(texp)  # 'wavelength', 'binwidth', 'efficiency', 'gain', , 'target', 'sky', 'dark', 'ron', 'noise'
            if np.isnan(res['target']).any():
                res['target'][np.isnan(res['target'])] = 0.
            if np.isnan(res['sky']).any():
                res['sky'][np.isnan(res['sky'])] = 0.

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

            try:
                hdu_list = dxu.joined()
                hdu_list = update_header(hdu_list, row, prog_id)
                hdu_list.writeto(output, overwrite=True)
            # except ValueError as e:
            #     print(f"Failed to save the joined spectrum: {row['TEMPLATE']}")
            except (ValueError, IndexError) as e:
                print(f"\nFailed to save the joined spectrum for {target_name} (z={row['REDSHIFT_ESTIMATE']}): {str(e)}")

        sys.stdout.write(f"\r{num}/{len(catalog)} \n")
        sys.stdout.flush()
    exptimes = Table(exptime_log)
    exptimes.meta['comments'] = ['Exposure times in seconds']
    log_fname = os.path.join(output_dir, 'exposure_times.csv')
    
    if os.path.exists(f'{log_fname}'):
        exptimes = Table.read(f'{output_dir}/model_parameters.fits')
        exptimes = vstack([exptimes, exptimes.data])
    else:
        exptimes = exptimes.data

    exptimes.write(log_fname,
                   formats={'TEXP': '%.1f', 'MAG': '%.2f', 'REDSHIFT': '%.4f'},
                   overwrite=True, comment='# ', format='csv')
    print("")



def main():
    parser = ArgumentParser(description="Generate simulated spectra from 4MOST Target Catalog")
    parser.add_argument("input", type=str, help="input target FITS catalog", 
                        # default='/data2/home2/nguerrav/QSO_simpaqs/ByCycle_Final_Cat_qso_templates_fobs_notna.fits'
                        )
    parser.add_argument('--airmass', type=float, default=1.2)
    parser.add_argument('--moon', type=str, default='gray', choices=['dark', 'gray', 'bright'])
    parser.add_argument('--seeing', type=float, default=0.8)
    parser.add_argument('-n', '--number', type=int, default=None)
    parser.add_argument('--rules', type=str, default='./../S17_20250122T1441Z_rules.csv', help='Rules definition (FITS or CSV)')
    parser.add_argument('--ruleset', type=str, default='./../S17_20250122T1443Z_rulesets.csv', help='Ruleset definition (FITS or CSV)')
    parser.add_argument('--temp-dir', type=str, default='/data2/home2/nguerrav/QSO_simpaqs/QSOs_full_cat/', help='Directory of spectral templates')
    parser.add_argument("-o", "--output", type=str, default='/data2/home2/nguerrav/QSO_simpaqs/QSOs_full_cat_ETC_L1_output_with_fobs/', help="output directory")
    # parser.add_argument('--arm', type=str, default='ALL', choices=['J', 'joined', 'ALL', 'a'])
    parser.add_argument('--prog', type=str, default='4MOST-ETC',
                        help="Determines the PROG_ID header keyword")

    args = parser.parse_args()

    t1 = datetime.datetime.now()
    catalog = Table.read(args.input)

    process_catalog(catalog,
                    ruleset_fname=args.ruleset,
                    rules_fname=args.rules,
                    output_dir=args.output,
                    template_path=args.temp_dir,
                    # airmass=args.airmass,
                    # seeing=args.seeing,
                    moon=args.moon,
                    # l1_type=args.arm,
                    N_targets=args.number,
                    prog_id=args.prog,
                    )
    t2 = datetime.datetime.now()
    dt = t2 - t1
    if args.number:
        N_targets = args.number
    else:
        N_targets = len(catalog)
    print(f"Finished simulation of {N_targets} targets in {dt.total_seconds():.1f} seconds")


if __name__ == '__main__':
    main()
