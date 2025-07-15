# functions written by Roland Szakacs: GitHub @astroland93

import numpy as np
from lmfit.models import VoigtModel

import h5py
from os import listdir
from os.path import isfile, join

def create_hdf5(QSO_sample, filename):
    wave_hdf5 = QSO_sample['wave'][0][0]
    spectrum_hdf5 = []
    EW_MgII_2796_hdf5 = []
    EW_MgII_2803_hdf5 = []
    EW_MgII_total_hdf5 = []
    z_MgII_hdf5 = []
    # EW_CIV_1548_hdf5 = []
    # EW_CIV_1551_hdf5 = []
    # EW_CIV_total_hdf5 = []
    # z_CIV_hdf5 = []
    
    for i in range(0,len(QSO_sample['wave'])):
        spectrum_hdf5.append(QSO_sample['spectrum'][i][0])
        EW_MgII_2796_hdf5.append(QSO_sample['EW_MgII_2796'][i][0])
        EW_MgII_2803_hdf5.append(QSO_sample['EW_MgII_2803'][i][0])
        EW_MgII_total_hdf5.append(QSO_sample['EW_MgII_total'][i][0])
        z_MgII_hdf5.append(QSO_sample['z_MgII'][i][0])
        # EW_CIV_1548_hdf5.append(QSO_sample['EW_CIV_1548'][i][0])
        # EW_CIV_1551_hdf5.append(QSO_sample['EW_CIV_1551'][i][0])
        # EW_CIV_total_hdf5.append(QSO_sample['EW_CIV_total'][i][0])
        # z_CIV_hdf5.append(QSO_sample['z_CIV'][i][0])
        
        
    QSO_sample_hdf5 = h5py.File(filename, 'w')
    QSO_sample_hdf5.create_dataset('wave_master', data=wave_hdf5)
    QSO_sample_hdf5.create_dataset('spectrum', data=spectrum_hdf5)
    QSO_sample_hdf5.create_dataset('EW_MgII_2796', data=EW_MgII_2796_hdf5)
    QSO_sample_hdf5.create_dataset('EW_MgII_2803', data=EW_MgII_2803_hdf5)
    QSO_sample_hdf5.create_dataset('EW_MgII_total', data=EW_MgII_total_hdf5)
    QSO_sample_hdf5.create_dataset('z_MgII', data=z_MgII_hdf5)
    # QSO_sample_hdf5.create_dataset('EW_CIV_1548', data=EW_MgII_2796_hdf5)
    # QSO_sample_hdf5.create_dataset('EW_CIV_1551', data=EW_MgII_2803_hdf5)
    # QSO_sample_hdf5.create_dataset('EW_CIV_total', data=EW_MgII_total_hdf5)
    # QSO_sample_hdf5.create_dataset('z_CIV', data=z_MgII_hdf5)

    QSO_sample_hdf5.close()

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
    return spectrum * MgIIflux



def insert_random_MgII(spectrum, MgII_dir, z_shift_max_arr, z_qso, EW_min = 0.002, verbose=False):
    
    ''' Function to insert a random MgII absorption line into the spectrum
    Input:
        spectrum: QSO spectrum
        MgII_dir: MgII array with MgII lines at different redshifts (use load_MgII to get the right format)
        z_shift_max: maximum range of shifting the MgII line. This is done to get a continuous 
                     function of MgII absorber in z-space
    Returns:
        spectrum: QSO spectrum with added MgII absorption line
        MgII_flux_shifted: MgII absorption spectrum shifted to a specific redshift
        MgII_prop: Properties of the MgII line. EW are rescaled to rest-frame 
    '''
    
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
        MgII_flux_shifted, z_MgII, mgii_2796_center_pos, mgii_2796_center_wl = shift_metal_abs(MgII_dir[z_file_num]['master_wave'][:], MgII_flux, z_shift, 2796., verbose=verbose)
        
        if(z_MgII <= z_qso and MgII_dir[z_file_num]['EW_total'][MgII_num] >= EW_min):
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
    spectrum = insert_metal_abs(spectrum, MgII_flux_shifted)
    
    # Create a MgII absorption line property array with EW rescaled to rest frame
    MgII_prop = [MgII_dir[z_file_num]['EW_MgII_2796'][MgII_num]/(1+z_MgII),
                 MgII_dir[z_file_num]['EW_MgII_2803'][MgII_num]/(1+z_MgII),
                 MgII_dir[z_file_num]['EW_total'][MgII_num]/(1+z_MgII),
                 z_MgII, mgii_2796_center_pos, mgii_2796_center_wl]    
    return spectrum, MgII_flux_shifted, MgII_prop

def add_MgII_to_loaded(wave, loaded_spectra, loaded_qso_objects, #plot=False, SNR=5, v
                       verbose=False):
    
    QSO_sample = pd.DataFrame(data={'wave': [], #'spectrum': [], 
                                    'continuum': [], #'z_QSO': [], 
                                    # 'SNR': [], 
                                    'EW_MgII_2796': [], 
                                    'EW_MgII_2803': [], 'EW_MgII_total': [], 'z_MgII': [], 'MgII_2796_center_pos': [], 'MgII_2796_center_wl': [], 
                                    # 'EW_CIV_1548': [], 'EW_CIV_1551': [], 'EW_CIV_total': [], 
                                    # 'z_CIV': [], 'CIV_1548_center_pos': [], 'CIV_1548_center_wl': []
                                    })


    MgII_abs = load_MgII('/Users/rszakacs/Desktop/simqso-master/examples/mgii_data/')
    z_shift_max = [0.1, 0.1, 0.2]
    j = 0
    for spectra in loaded_spectra:
        i = 0
        for spectrum in spectra:
            spectrum_t, MgII_flux_shifted_t, MgII_prop = insert_random_MgII(np.array(spectrum), MgII_abs, z_shift_max, loaded_qso_objects[j].z[i], verbose=verbose)
            
            # noised_spectrum = noise_spectrum(wave, spectrum_t, SNR, loaded_qso_objects[j].z[i], verbose=verbose)
            
            QSO_sample = QSO_sample.append({'wave': [wave], # 'spectrum': [noised_spectrum], 
                                            'continuum': [spectrum_t], # 'z_QSO': [loaded_qso_objects[j].z[i]], #'SNR': [SNR],
                                    'EW_MgII_2796': [MgII_prop[0]], 'EW_MgII_2803': [MgII_prop[1]], 
                                    'EW_MgII_total': [MgII_prop[2]], 'z_MgII': [MgII_prop[3]], 'MgII_2796_center_pos': [MgII_prop[4]], 'MgII_2796_center_wl': [MgII_prop[5]],
                                    # 'EW_CIV_1548': [0], 'EW_CIV_1551': [0], 
                                    # 'EW_CIV_total': [0], 'z_CIV': [0], 'CIV_1548_center_pos': [0], 'CIV_1548_center_wl': [0]
                                    }, 
                                    ignore_index=True)
            i = i+1
        j = j+1
        
    create_hdf5(QSO_sample, '/data2/home2/nguerrav/TNG50_spec/MgII/')

def main():

    # etc_grid = np.load

    MgII_dir = load_MgII('/data2/home2/nguerrav/TNG50_spec/')
    # z_shift_max = [0.1, 0.1, 0.2]
    # test_spectrum, MgII_flux, MgII_prop = insert_random_MgII(spectrum[0], MgII_dir, z_shift_max, qso_object.z[0], verbose=True)
    add_MgII_to_loaded(waves, loaded_qso_objects, 
                       verbose=False)
    MgII_abs = load_MgII('/Users/rszakacs/Desktop/simqso-master/examples/mgii_data/')

if __name__ == '__main__':
    main()
