# -*- coding: utf-8 -*-

"""
multitrack.py — checks several candidate explanations for the L^p-norm
landscape series' cyclical pattern against each other in one pass.

Already ruled out:
  - Crash-specific timing (diagnostic_lpnorm_vs_crashes.py) -- the series
    is cyclical across the whole sample, not spiking specifically before
    known crashes.
  - Realized pairwise correlation (check_lpnorm_vs_realized_corr.py) --
    Pearson r ~ 0.03-0.08 against 60-day realized correlation, i.e. not
    a correlation proxy.

Still open -- this script checks three more candidates in one run, all
model-free (no TDA recomputation, no DCC, no GARCH refitting):

  1. Realized volatility -- is the landscape mostly tracking how big
     returns are, independent of co-movement? (average of each asset's
     own rolling realized vol, equal-weighted across the 5 assets)

  2. Within-window autocorrelation -- since embed_dim=80 inside a
     252-day window means each point cloud is built from a fairly
     short, overlapping span, a trending/autocorrelated period could
     change point-cloud shape even with no crash and no vol spike.
     Proxied here by rolling lag-1 autocorrelation of each asset's
     returns, averaged across assets.

  3. VIX level (if available locally) -- the most direct, literature-
     standard stress proxy, included as a sanity check alongside the
     two return-derived ones above. Skipped automatically if no VIX
     file/column is found -- this is opportunistic, not required.

Each candidate is compared against all three lh1_k*_norm columns via
Pearson correlation, with a combined plot for visual inspection.

Usage:
  python multitrack.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

ROLLING_WINDOW = 60  # trading days -- same window used in the earlier
                      # realized-correlation check, kept consistent here


# CANDIDATE 1: REALIZED VOLATILITY

def compute_realized_volatility(log_returns_df, window=ROLLING_WINDOW):
    """
    Equal-weighted average of each asset's own rolling realized vol
    (rolling std of daily log returns, annualisation doesn't matter
    here since we only care about correlation with the norm series,
    not the absolute scale).
    """
    per_asset_vol = log_returns_df.rolling(window).std()
    return per_asset_vol.mean(axis=1)


# CANDIDATE 2: WITHIN-WINDOW AUTOCORRELATION / TRENDINESS

def compute_rolling_autocorr(log_returns_df, window=ROLLING_WINDOW, lag=1):
    """
    Equal-weighted average lag-1 autocorrelation of each asset's
    returns, computed on a rolling basis. Proxies how "trendy" vs.
    "choppy" the recent return series has been -- a delay-embedded
    point cloud from a trending window has a different shape than one
    from a mean-reverting/choppy window, independent of volatility
    level or correlation.
    """
    def roll_autocorr_1d(series, window, lag):
        return series.rolling(window).apply(
            lambda x: pd.Series(x).autocorr(lag=lag), raw=False
        )

    per_asset_autocorr = log_returns_df.apply(
        lambda col: roll_autocorr_1d(col, window, lag), axis=0
    )
    return per_asset_autocorr.mean(axis=1)


# CANDIDATE 3: VIX (OPTIONAL, OPPORTUNISTIC)

def try_load_vix(paths):
    """
    Looks for a VIX series under a couple of plausible config keys /
    file names. Returns a Series or None -- this candidate is skipped
    cleanly if nothing is found, since it's not part of the existing
    pipeline and may not be on disk.
    """
    candidate_keys = ['vix', 'vix_series', 'vix_data']
    for key in candidate_keys:
        path = paths.get(key)
        if path and os.path.exists(path):
            df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path, index_col=0, parse_dates=True)
            col = 'VIX' if 'VIX' in df.columns else df.columns[0]
            print(f"  Found VIX data at {path} (column '{col}')")
            return df[col]

    fallback_paths = ['data/processed/vix.parquet', 'data/raw/vix.csv', 'data/processed/vix.csv']
    for fp in fallback_paths:
        if os.path.exists(fp):
            df = pd.read_parquet(fp) if fp.endswith('.parquet') else pd.read_csv(fp, index_col=0, parse_dates=True)
            col = 'VIX' if 'VIX' in df.columns else df.columns[0]
            print(f"  Found VIX data at {fp} (column '{col}')")
            return df[col]

    print("  No VIX file found -- skipping this candidate (not required).")
    return None


# COMBINED CHECK

def run_multitrack(tda_features_df, log_returns_df, paths,
                   window=ROLLING_WINDOW, out_path="multitrack_results.png"):

    norm_cols = sorted(
        [c for c in tda_features_df.columns if c.startswith('lh1_') and c.endswith('_norm')],
        key=lambda c: int(c.split('_k')[1].split('_')[0])
    )
    if not norm_cols:
        raise ValueError("No lh1_k*_norm columns found in tda_features_df.")

    print("Computing candidate series...")
    print("  [1/3] realized volatility...")
    realized_vol = compute_realized_volatility(log_returns_df, window=window)

    print("  [2/3] rolling autocorrelation...")
    rolling_autocorr = compute_rolling_autocorr(log_returns_df, window=window)

    print("  [3/3] VIX (optional)...")
    vix = try_load_vix(paths)

    candidates = {
        'realized_volatility': realized_vol,
        'rolling_autocorr':    rolling_autocorr,
    }
    if vix is not None:
        candidates['vix'] = vix

    # Align everything on shared, non-NaN dates
    common_idx = tda_features_df.index
    for series in candidates.values():
        common_idx = common_idx.intersection(series.dropna().index)
    print(f"\nOverlapping dates across all series: {len(common_idx)}")

    aligned_norms = tda_features_df.loc[common_idx, norm_cols]
    aligned_candidates = {name: s.loc[common_idx] for name, s in candidates.items()}

    print("\nPearson correlation: lh1_k*_norm  vs.  each candidate")
    print("_" * 60)
    results = {}
    for cand_name, cand_series in aligned_candidates.items():
        results[cand_name] = {}
        print(f"\n  vs. {cand_name}:")
        for col in norm_cols:
            r = aligned_norms[col].corr(cand_series)
            results[cand_name][col] = r
            print(f"    {col:18s}  r = {r:+.4f}")

    # Plot: top panel per candidate, one row per norm column below it
    n_candidates = len(aligned_candidates)
    n_rows = n_candidates + len(norm_cols)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.6 * n_rows), sharex=True)

    row = 0
    for cand_name, cand_series in aligned_candidates.items():
        axes[row].plot(common_idx, cand_series, color='black', linewidth=0.9)
        axes[row].set_ylabel(cand_name)
        axes[row].grid(alpha=0.3)
        row += 1

    for col in norm_cols:
        r_summary = ", ".join(f"{name} r={results[name][col]:+.2f}" for name in aligned_candidates)
        axes[row].plot(common_idx, aligned_norms[col], color='steelblue', linewidth=0.8,
                       label=r_summary)
        axes[row].set_ylabel(col)
        axes[row].legend(loc='upper left', fontsize=8)
        axes[row].grid(alpha=0.3)
        row += 1

    axes[0].set_title(
        "multitrack: L^p-norm landscape series vs. candidate explanations\n"
        "(top rows = candidate series; bottom rows = topology features, "
        "Pearson r vs. each candidate in legend)"
    )
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")

    print("\nReading the result:")
    print("  Any |r| > ~0.5-0.6 against a candidate -> that candidate")
    print("    likely explains most of the cyclical pattern; the L^p-norm")
    print("    feature is largely redundant with something simpler to")
    print("    compute directly.")
    print("  All |r| small across all three candidates -> the cyclical")
    print("    pattern isn't explained by volatility, within-window")
    print("    trendiness, or (if available) VIX either -- genuinely")
    print("    distinct signal, worth treating as a real finding and")
    print("    investigating what it tracks more directly (e.g. PCA")
    print("    explained-variance ratio over time, or point-cloud")
    print("    diameter/spread before the topology step).")

    return results


if __name__ == "__main__":
    from src.data.loader import load_config

    config = load_config()
    paths = config['paths']

    log_returns = pd.read_parquet(paths['log_returns'])
    tda_features = pd.read_parquet(
        paths.get('tda_features_lpnorm', 'data/processed/tda_features_lpnorm.parquet')
    )

    print(f"Log returns shape: {log_returns.shape}")
    print(f"TDA features shape: {tda_features.shape}")

    results = run_multitrack(tda_features, log_returns, paths, window=ROLLING_WINDOW)