
'''

'''

import pandas as pd
import numpy as np
from astropy.table import Table
import astropy.units as u

from qmostetc import SEDTemplate, QMostObservatory, Spectrum
from qmostetc.catalog import _split_magtype as split_magtype

template_4fs = Table.read('./../4M_templ_z1_00_extended.fits')
template_4fs

spec = Spectrum(np.asarray(template_4fs['LAMBDA']) * u.Angstrom, 
                np.asarray(template_4fs['FLUX_DENSITY']) * u.erg / (u.cm**2 * u.s * u.Angstrom)
                )

template = SEDTemplate(spec)#.to('erg / (nm m² s)'))
flux = template(20*u.ABmag, 'DECam.z')

qmost = QMostObservatory('hrs')  # high-resolution
obs = qmost(45*u.deg,  # airmass
            1.3*u.arcsec,  # seeing
            'gray')  # moon conditions

obs.set_target(flux, 'point')
tbl = obs.expose((10000*60)*u.s)

np.save('/data2/home2/nguerrav/QSO_simpaqs/npy_files/etc_wavelength_grid.npy', 
        np.asarray(tbl['wavelength']) * 10,  # in angstroms
        allow_pickle=True)
