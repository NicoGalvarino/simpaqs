# PAQS Spectral Simulator


Code to generate a sample of random quasars with realistic absorption systems:
Lyman-alpha forest, Lyman-limit systems and DLAs, including their metal lines
and molecular and CI fine-structure lines in 20% of the DLAs above z > 2.4.
The metal lines only include singly ionized lines plus CIV and SiIV for now.

The code also allows a simulation and exposure time estimation of a 4MOST target catalog.
For this see the section about *ETC simulator* below.


## Installation

Clone the code:
    
    git clone 
    cd simpaqs
    python -m pip install -r requirements.txt


Install 4MOST ETC:

    QMOST_PYPI=https://gitlab.4most.eu/api/v4/projects/212/packages/pypi/simple
    python -m pip install --extra-index-url $QMOST_PYPI qmostetc

Clone and install `fits_utils` in the same environment: [https://github.com/NicoGalvarino/fits_utils/tree/main](https://github.com/NicoGalvarino/fits_utils/tree/main)

Install `simqso` (not needed if only running `simulate_catalog.py`):

    git clone git@github.com:imcgreer/simqso.git
    cd simqso
    python setup.py install

## Pipeline Overview

The full simulation pipeline runs in three steps:

1. **`simulate_quasars_no_abs.py`** — Generate synthetic QSO spectral templates without absorbers
2. **`simulate_catalog.py`** — Run the 4MOST ETC on the templates to produce mock L1 spectra
3. **`rebin_and_get_SNR.py`** — Rebin L1 spectra onto the ETC wavelength grid and compute per-arm SNR

---

## Step 1 — Simulate quasar templates: `simulate_quasars_no_abs.py`

Generates synthetic QSO continuum + emission line templates using `simqso`. The continuum is a broken power law (break at 1215 Å) with Gaussian-sampled slopes, luminosities derived from log-normal BH mass and Eddington ratio distributions, Fe emission from the VW01 template, and SMC dust extinction.

**From a redshift list (recommended — uses the real survey catalogue):**

```bash
python simulate_quasars_no_abs.py \
    --input_cat_path cat.fits
    --dir ./output_directory/
```

The script reads redshifts, magnitudes, and subsurvey labels directly from input catalogue provided in `--input_cat`. This input cat should be in the 4FS format with optionally an added `fobs` column.

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dir` | `./QSO_templates` | Output directory for `.fits` template files |
| `--number` | all golden-sample QSOs | Number of QSOs to simulate |
| `--wavelen_grid` | `TNG50_wavelength_grid_extended.npy` | Wavelength grid `.npy` file |
| `--wmin` / `--wmax` | 3000 / 11000 Å | Wavelength range fallback if no grid file |
| `--dust` | `exponential` | Dust sampling: `exponential` (Krawczyk+2015) or `uniform` |

Each QSO is saved as a `.fits` file: `QSO_sim_z{z}_{name}.fits` with header keywords `REDSHIFT_ESTIMATE`, `MAG`, `EBV`, `LOG_MBH`, `LOG_REDD`, `LOG_LBOL`.

By the end, the scripts saves a catalogue table to `--dir` where the `TEMPLATES` column was modified to the newly created template.

---

## Step 2 — Run the ETC and L1 simulation: `simulate_catalog.py`

Takes a 4MOST target catalogue with associated spectral templates plus rules/rulesets, and produces mock L1 spectra (joined HRS) with realistic noise, cosmic rays, and per-target exposure times.

```bash
python simulate_catalog.py \
    --input_cat /path/to/catalog.fits \
    --temp-dir  /path/to/QSO_templates/ \
    --rules     S17_20250122T1441Z_rules.csv \
    --ruleset   S17_20250122T1443Z_rulesets.csv \
    --output    /path/to/l1_output/
```

The catalogue must follow the 4FS format and contain at minimum: `TEMPLATE`, `REDSHIFT_ESTIMATE`, `MAG`, `SUBSURVEY`, `RULESET`, and optionally `fobs`. If no `fobs` column is available in the input cat, the script assumes `fobs = 1.0`.

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_cat` | (see script) | Input target FITS catalogue |
| `--temp-dir` | `QSOs_full_cat/` | Directory of spectral template `.fits` files |
| `--rules` | `S17_…_rules.csv` | 4FS rules CSV |
| `--ruleset` | `S17_…_rulesets.csv` | 4FS rulesets CSV |
| `-o` / `--output` | `QSOs_L1_output_…/` | Output directory for L1 `.fits` spectra |
| `--moon` | `gray` | Sky background: `dark`, `gray`, `bright` |
| `-n` / `--number` | all | Number of targets to process |
| `--n-cores` | 75% of CPUs | Cores for parallel processing (chunks of 10 000) |

Output spectra are named `{model_id}_ETC_LJ1.fits`. Exposure times are logged to `exposure_times.csv` in the output directory.

---

## Step 3 — Rebin and compute SNR: `rebin_and_get_SNR.py`

Rebins each L1 spectrum onto the ETC wavelength grid using `spectres`, splits into blue/green/red arms, and writes per-arm mean SNR back into the catalogue.

```bash
python rebin_and_get_SNR.py \
    --input-cat       catalog.fits \
    --output-cat      catalog_with_SNR.fits \
    --cat-path        /path/to/catalogues/ \
    --l1-spec-path    /path/to/l1_output/ \
    --rebinned-spec-path /path/to/rebinned_output/ \
    --etc-grid-path   /path/to/etc_wavelength_grid.npy
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-cat` | `ByCycle_Final_cat_with_qselfie_682.fits` | Input catalogue |
| `--output-cat` | `…_with_SNR.fits` | Output catalogue with SNR columns added |
| `--cat-path` | `Catalogues/cat_april15/` | Catalogue directory |
| `--l1-spec-path` | `QSOs_L1_output_…/` | Directory of L1 spectra |
| `--rebinned-spec-path` | `QSOs_L1_output_…_rebinned/` | Output directory for rebinned spectra |
| `--etc-grid-path` | `npy_files/etc_wavelength_grid.npy` | ETC wavelength grid |
| `-n` / `--number` | all | Number of targets to process |
| `--n-cores` | 75% of CPUs | Parallel cores |
| `--batch-size` | 100 | Spectra per batch |

Adds columns `SNR_mean_mgii`, `SNR_blue_mean_mgii`, `SNR_green_mean_mgii`, `SNR_red_mean_mgii` to the output catalogue. Arm boundaries: blue ≤ 4355 Å, green 5159.8–5730 Å, red ≥ 6099.8 Å.

