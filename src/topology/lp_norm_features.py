# -*- coding: utf-8 -*-

"""
L^p-norm Landscape Summary — Gidea & Katz (2018) style feature reduction

Collapses the raw persistence landscape grid samples produced by
tda_pipeline.py (180 columns: 3 levels x 30 grid points x 2 degrees)
down to one scalar L^p-norm per landscape level, for a single chosen
homology degree.

This is literature-matched to Gidea & Katz, "Topological Data Analysis
of Financial Time Series: Landscapes of Crashes" (2018), who used only
H1 (cycle/loop structure) and summarised each landscape level with its
L^2 norm as a scalar early-warning feature. H0 is deliberately excluded
here it tracks point-cloud density/clustering, not the cyclic
co-movement structure the paper's claim rests on, so including it would
be borrowing more than the citation actually supports.

This function is pure post-processing on the DataFrame returned by
run_tda_pipeline(). 
"""

import numpy as np
import pandas as pd


def extract_lp_norm_features(tda_df, grid, n_landscapes=3, p=2, degrees=('h1',)):
    """
    Collapse raw persistence landscape grid samples into L^p-norm summary
    features, one scalar per landscape level.

    For landscape level k, the discretised L^p norm over the filtration
    grid is:

        ||lambda_k||_p = ( dt * sum_i |lambda_k(t_i)|^p ) ** (1/p)

    which approximates the continuous norm used in Gidea & Katz. p=2
    is their choice; expose it as a parameter so other p values can be
    checked without rewriting anything.

    Args:
        tda_df       : DataFrame from run_tda_pipeline(), containing
                        columns 'l{degree}_k{k}_g{g}' plus the 6 scalar
                        feature columns (betti_1, entropy_h0, ...).
        grid         : 1D array, the global filtration grid returned
                        alongside tda_df (needed for grid spacing dt).
        n_landscapes : number of landscape levels to reduce (matches
                        N_LANDSCAPES used when tda_df was built).
        p            : norm order. 2 = Gidea & Katz's L^2 norm.
        degrees      : which homology degrees to reduce. Default is
                        ('h1',) -- the literature-matched choice. Pass
                        ('h0', 'h1') only if you want the exploratory,
                        non-literature-matched variant.

    Returns:
        DataFrame, same DatetimeIndex as tda_df, containing:
            - the original 6 scalar feature columns, unchanged
            - one 'l{degree}_k{k}_norm' column per (degree, level)
        For the default H1-only case this is 6 + 3 = 9 total columns,
        down from 186.
    """
    grid = np.asarray(grid)
    dt = float(grid[1] - grid[0]) if len(grid) > 1 else 1.0

    scalar_cols = ['betti_1', 'entropy_h0', 'entropy_h1',
                   'max_persistence', 'total_persistence', 'wasserstein']
    scalar_cols = [c for c in scalar_cols if c in tda_df.columns]

    out = tda_df[scalar_cols].copy()

    for deg in degrees:
        for k in range(n_landscapes):
            prefix = f'l{deg}_k{k}_g'
            cols = [c for c in tda_df.columns if c.startswith(prefix)]
            if not cols:
                continue
            # sort by grid index -- column order in tda_df isn't guaranteed
            cols_sorted = sorted(cols, key=lambda c: int(c[len(prefix):]))
            values = tda_df[cols_sorted].to_numpy()          # (n_days, G)
            norm = (dt * np.sum(np.abs(values) ** p, axis=1)) ** (1.0 / p)
            out[f'l{deg}_k{k}_norm'] = norm

    return out


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.data.loader import load_config

    config = load_config()
    paths = config['paths']

    tda_df = pd.read_parquet(paths['tda_features_landscape'])
    grid = np.load(paths['landscape_grid'])

    lp_df = extract_lp_norm_features(tda_df, grid, n_landscapes=3, p=2, degrees=('h1',))

    print(f"Reduced {tda_df.shape[1]} columns -> {lp_df.shape[1]} columns")
    print(lp_df.describe().round(4))

    out_path = paths.get('tda_features_lpnorm', 'data/processed/tda_features_lpnorm.parquet')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lp_df.to_parquet(out_path)
    print(f"Saved to {out_path}")