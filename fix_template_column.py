"""
Merge the TEMPLATE column from golden_sample_expanded.fits into the input catalog.

simulate_quasars_no_abs.py writes per-QSO template IDs into golden_sample_expanded.fits
but a crash in the post-processing block prevents those IDs from being written back to
the input catalog. This script does that merge.

The TEMPLATE values are IDs without the .fits extension (e.g. QSO_z1.2345_000001).
simulate_catalog.py appends .fits itself when loading the spectrum file.

Usage:
    python fix_template_column.py \
        --catalog  data/ByCycle_Final_Cat_with_all_S17_cols_with_qselfie_848.fits \
        --golden   QSO_templates/golden_sample_expanded.fits \
        --output   data/catalog_with_templates.fits
"""

import argparse
import numpy as np
from astropy.table import Table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', required=True, help='Input catalog FITS file')
    parser.add_argument('--golden',  required=True,
                        help='golden_sample_expanded.fits from Step 1')
    parser.add_argument('--output',  required=True, help='Output FITS file')
    args = parser.parse_args()

    cat    = Table.read(args.catalog)
    golden = Table.read(args.golden)

    print(f"Catalog rows:        {len(cat)}")
    print(f"golden_sample rows:  {len(golden)}")

    if len(golden) > len(cat):
        # golden_sample was built up over multiple runs via vstack — the most
        # recent run's data is at the end. Take only the last len(cat) rows.
        n_extra = len(golden) - len(cat)
        print(f"golden_sample has {len(golden)//len(cat)}x catalog rows "
              f"(likely {len(golden)//len(cat)} stacked runs). "
              f"Using the last {len(cat)} rows (most recent run).")
        golden = golden[n_extra:]

    if len(cat) != len(golden):
        raise ValueError(
            f"Row count mismatch: catalog={len(cat)}, golden={len(golden)}. "
            "Cannot do a positional merge — counts must be equal."
        )

    # Sanity check: redshifts should match in order
    if 'REDSHIFT_ESTIMATE' in cat.colnames and 'redshift' in golden.colnames:
        dz = np.abs(np.array(cat['REDSHIFT_ESTIMATE'], dtype=float)
                    - np.array(golden['redshift'],       dtype=float))
        max_dz = float(np.max(dz))
        n_bad  = int(np.sum(dz > 1e-4))
        if max_dz > 1e-4:
            print(f"WARNING: {n_bad} rows have |dz| > 1e-4 (max={max_dz:.6f})")
            print("         Rows may not be aligned — check before using the output.")
        else:
            print(f"Redshift check OK: max |dz| = {max_dz:.2e}")

    cat['TEMPLATE'] = np.array(golden['TEMPLATE'], dtype='<U100')

    print(f"Sample TEMPLATE values:")
    for v in cat['TEMPLATE'][:3]:
        print(f"  {v}")

    cat.write(args.output, overwrite=True)
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()
