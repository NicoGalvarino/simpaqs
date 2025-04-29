
import sys
import numpy as np
from astropy import table

from simulate_absorbers import make_absorber_templates
from simulate_quasars import add_quasar_continuum
from simulate_spectra import process_catalog

__author__ = 'Jens-Kristian Krogager'
__email__ = 'jens-kristian.krogager@univ-lyon1.fr'


# -- Input parameters
Z_MIN = 0.55
Z_MAX = 1.0

texp_hours = 2
EXPTIME = texp_hours * 60 * 60  # seconds
MOON = 'dark'

# EXPTIME = 53.29 * 60  # seconds
# MOON = 'gray'

# EXPTIME = 97.446 * 60  # seconds
# MOON = 'bright'

MAG_MIN = 19.0
MAG_MAX = 21.0

# MAG_MIN = 15.0
# MAG_MAX = 17.0

OUTPUT_DIR = '../simpaqs_outputs_test_dim/etc_output'
QSO_OUTPUT_DIR = '../simpaqs_outputs_test_dim/quasar_models'

BAL = False

##########################

N_TOTAL = int(sys.argv[1])
# np.random.seed(20230521)

abs_template_list, abslog, DLAlog = make_absorber_templates(N_TOTAL, z_min=Z_MIN, z_max=Z_MAX, verbose=True)

# abs_template_list = table.Table.read("output/list_templates.csv") # For midway inspection

model_input = add_quasar_continuum(abs_template_list, BAL=BAL, output_dir=QSO_OUTPUT_DIR,
                                   # dust_mode='flat',
                                   )

# model_input = table.Table.read('output/quasar_models/model_input.csv') # For midway inspection

process_catalog(model_input, mag_min=MAG_MIN, mag_max=MAG_MAX, template_path=QSO_OUTPUT_DIR,
                exptime=EXPTIME, moon=MOON, output=OUTPUT_DIR)#, spectro=spectro)

