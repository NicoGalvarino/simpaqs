# 4MOST S17 Spectral Simulation Pipeline

Produces a catalogue of synthetic 4MOST HRS quasar spectra with per-arm SNR values,
starting from a 4FS target catalogue with `fobs` allocations.

**Output:** `data/catalog_with_SNR.fits` — the input catalogue augmented with
`SNR_mean_mgii`, `SNR_blue_mean_mgii`, `SNR_green_mean_mgii`, `SNR_red_mean_mgii`.

---

## Environment

### 1. Create the conda environment

```bash
conda create -n 4most_sims python=3.13
conda activate 4most_sims
```

### 2. Install standard dependencies via pip

```bash
pip install numpy==1.26.4 astropy scipy pandas matplotlib tqdm extinction spectres
```

### 3. Install qmostetc (4MOST private PyPI)

```bash
QMOST_PYPI=https://gitlab.4most.eu/api/v4/projects/212/packages/pypi/simple
pip install --extra-index-url $QMOST_PYPI qmostetc
```

You need a 4MOST GitLab account with access to the ETC project. This installs `qmostetc==2.6.1`.

### 4. Install fits_utils

```bash
cd /path/to/ESO
git clone https://github.com/NicoGalvarino/fits_utils.git
cd fits_utils
pip install -e .
```

### 5. Install extinction (conda-forge)

```bash
conda install -c conda-forge extinction
```

Note: `pip install extinction` may fail on some platforms; conda-forge is more reliable.

### 6. Install simqso (requires patches for Python 3.13)

```bash
cd /path/to/ESO
git clone git@github.com:imcgreer/simqso.git
cd simqso
```

Apply these patches before installing:

**`simqso/sqbase.py`** — `pkg_resources` removed in 3.13:
```python
# replace:
from pkg_resources import resource_filename
datadir = resource_filename('simqso', 'data')
# with:
from importlib.resources import files
datadir = str(files('simqso') / 'data')
```

**`simqso/sqgrids.py`** and **`simqso/sqphoto.py`** — `simps` renamed:
```python
# replace:
from scipy.integrate import simps
# with:
from scipy.integrate import simpson as simps
```

**`simqso/lumfun.py`** — `romberg` removed, `scipy.ndimage.filters` moved:
```python
# replace:
from scipy.integrate import quad,dblquad,romberg,simps
from scipy.ndimage.filters import convolve
# with:
from scipy.integrate import quad, dblquad, simpson as simps
from scipy.ndimage import convolve

def romberg(func, a, b, args=(), **kwargs):
    from scipy.integrate import quad as _quad
    return _quad(func, a, b, args=args)[0]
```

Then install:
```bash
python setup.py install
```

### 7. Fix qmostetc numpy 2.x incompatibility

```bash
sed -i 's/np\.trapz/np.trapezoid/g' $(python -c "import qmostetc; import os; print(os.path.dirname(qmostetc.__file__))")/spectrum.py
```

---

## Pre-step — ETC wavelength grid

`data/etc_wavelength_grid.npy` is already included in this repository (18 018 pixels,
3926–6790 Å). It was generated once using `save_etc_wavelength_grid.py`, which runs the 4MOST ETC
to extract the native wavelength grid of the joined HRS arms. You do not need to
regenerate it.

---

## Directory layout

```
ESO/
├── S17_20250122T1441Z_rules.csv
├── S17_20250122T1443Z_rulesets.csv
└── simpaqs/
    ├── data/
    │   ├── ByCycle_Final_Cat_with_all_S17_cols_with_qselfie_848.fits   ← input (see below)
    │   ├── etc_wavelength_grid.npy
    │   ├── catalog_with_templates.fits    ← created after Step 1
    │   └── catalog_with_SNR.fits         ← created after Step 3
    ├── QSO_templates/    ← created by Step 1
    ├── L1_products/      ← created by Step 2
    └── L1_rebinned/      ← created by Step 3
```

### Input catalogue

`ByCycle_Final_Cat_with_all_S17_cols_with_qselfie_848.fits` is not tracked in git
(980 MB). It is a merged catalogue combining the S17 4FS target list with the output
of a specific 4FS qselfie run (run 848), which provides columns such as `fobs`
(fraction of survey time allocated per target), `texp_g`, `texp_d`, `texp_b`,
`texp_s` (pre-computed gray/dark/bright/seeing-limited exposure times), and the full
set of S17 columns. Obtain it from the project team before running the pipeline.

---

## Step 1 — Generate synthetic QSO templates

Simulates quasar continua + emission lines using `simqso`. Outputs one `.fits` template
per target and a `golden_sample_expanded.fits` summary table.

```bash
python simulate_quasars_no_abs.py \
    --input_cat_path data/ByCycle_Final_Cat_with_all_S17_cols_with_qselfie_848.fits \
    --dir            QSO_templates/
```

The script processes the full catalogue (~1.35 M targets) in chunks of 10 000.
Template IDs are 7-digit, e.g. `QSO_z2.5_0000001`.

### Step 1b — Merge template IDs back into the catalogue

`simulate_quasars_no_abs.py` writes template IDs into `golden_sample_expanded.fits`
as it goes, then attempts to merge them back into the input catalogue in a
post-processing block at the end. That block crashes because it references a
`model_parameters.fits` file that is no longer produced by the script.
`fix_template_column.py` is a standalone replacement for that step: it does
a positional merge of the `TEMPLATE` column from `golden_sample_expanded.fits`
into the original catalogue.

```bash
python fix_template_column.py \
    --catalog data/ByCycle_Final_Cat_with_all_S17_cols_with_qselfie_848.fits \
    --golden  QSO_templates/golden_sample_expanded.fits \
    --output  data/catalog_with_templates.fits
```

Check the output for a redshift-alignment warning. If `max |dz| < 1e-4` the merge is correct.

---

## Step 2 — Run the 4MOST ETC and produce mock L1 spectra

Takes `catalog_with_templates.fits` and produces one `_ETC_LJ1.fits` spectrum per target.
Exposure times are logged to `L1_products/exposure_times.csv`.

```bash
python simulate_catalog.py \
    --input_cat data/catalog_with_templates.fits \
    --temp-dir  QSO_templates/ \
    --rules     ../S17_20250122T1441Z_rules.csv \
    --ruleset   ../S17_20250122T1443Z_rulesets.csv \
    --output    L1_products/ \
    --n-cores   30
```

**Notes:**
- ~270 000 targets have `fobs = 0` (not allocated any survey time) and are skipped;
  they are logged to `L1_products/failed_spectra.txt`.
- The script is resumable: if a spectrum file already exists it is skipped and its
  exposure-time entry is reconstructed from the catalogue's `texp_g` column.
- On SLURM, request `--cpus-per-task=40` (nodes have 40 physical cores;
  `--n-cores 30` uses 75 % of them).

---

## Step 3 — Rebin spectra and compute per-arm SNR

Rebins each L1 spectrum onto the ETC wavelength grid using `spectres`, splits into
blue / green / red arms, and writes mean SNR columns back into the catalogue.

```bash
python rebin_and_get_SNR.py \
    --input-cat          catalog_with_templates.fits \
    --output-cat         catalog_with_SNR.fits \
    --cat-path           data/ \
    --l1-spec-path       L1_products/ \
    --rebinned-spec-path L1_rebinned/ \
    --etc-grid-path      data/etc_wavelength_grid.npy
```

Output columns added to `catalog_with_SNR.fits`:

| Column | Description |
|---|---|
| `SNR_mean_mgii` | Mean SNR across full wavelength range |
| `SNR_blue_mean_mgii` | Mean SNR, blue arm (≤ 4355 Å) |
| `SNR_green_mean_mgii` | Mean SNR, green arm (5159.8–5730 Å) |
| `SNR_red_mean_mgii` | Mean SNR, red arm (≥ 6099.8 Å) |

The script is resumable: targets with an existing rebinned file are read directly
without re-running `spectres`.

---

## Step 4 — Build SNR lookup grid

Produces a 3-D (mag, log fobs, redshift) grid of median log SNR values for survey
planning. Run from the `S17_4MOST_Cataloguing/Code/` directory, which expects
`../Catalogues/catalog_with_SNR.fits`.

```bash
cd /path/to/S17_4MOST_Cataloguing/Code/
python SNR_grid.py
# Outputs: SNR_grid.npz, SNR_grid.csv
```

The script filters out `fobs = 0` and `SNR < 0` rows before binning to avoid
`log10(0)` producing non-monotonic bin edges.
