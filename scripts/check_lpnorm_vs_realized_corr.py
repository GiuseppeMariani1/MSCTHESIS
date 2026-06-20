# -*- coding: utf-8 -*-

"""
Check: is the L^p-norm landscape series just a noisy proxy for realized
pairwise correlation, rather than carrying distinct topological information?

Motivation: the diagnostic plot (lpnorm_vs_crashes.png) showed lh1_k*_norm
oscillating in a smooth, multi-year cycle across the full 2000-2025 sample,
with no clear pre-crash rise specific to known stress events (2008, 2011,
2018, 2020, 2022). That's NOT the Gidea & Katz pattern. But correlation
regimes are themselves known to run in slow multi-year cycles independent
of crash timing -- so if lh1_k*_norm is mostly tracking realized
correlation directly, that would explain:
  (a) why the series looks cyclical rather than crash-spiking, and
  (b) why it sometimes nudged DCC log-likelihood upward (it's a noisy
      version of the very quantity DCC is trying to model) without
      reliably beating a permutation shuffle (a slow, autocorrelated
      proxy partially survives shuffling too, since DCC only needs SOME
      day-to-day flexibility, not crash-specific timing).

This does NOT use TopoDCC or GARCH residuals at all -- it's a direct,
model-free comparison: rolling realized correlation from raw returns vs.
the L^p-norm series, on the same dates.

Usage:
  python check_lpnorm_vs_realized_corr.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

ROLLING_WINDOW = 60  # trading days, for realized correlation -- matches the
                      # 60-day rolling mean already used in the diagnostic plot


def compute_realized_avg_correlation(log_returns_df, window=ROLLING_WINDOW):
    """
    Rolling average pairwise correlation across all assets, one value per
    day. This is model-free -- straight from raw log returns, nothing to
    do with TDA, GARCH, or DCC.

    Returns a Series aligned to log_returns_df.index (first `window` days
    will be NaN, same as any rolling computation).
    """
    n_assets = log_returns_df.shape[1]
    pairs = [(i, j) for i in range(n_assets) for j in range(i + 1, n_assets)]

    avg_corr = pd.Series(index=log_returns_df.index, dtype=float)

    roll_corr = log_returns_df.rolling(window).corr()
    # roll_corr is a (n_days * n_assets) x n_assets MultiIndex frame;
    # pull out one date at a time and average the off-diagonal pairs.
    for date in log_returns_df.index[window - 1:]:
        try:
            corr_matrix = roll_corr.loc[date]
        except KeyError:
            continue
        pair_vals = [corr_matrix.iloc[i, j] for i, j in pairs]
        avg_corr.loc[date] = np.nanmean(pair_vals)

    return avg_corr


def compare_lpnorm_to_realized_corr(tda_features_df, log_returns_df,
                                    window=ROLLING_WINDOW,
                                    out_path="lpnorm_vs_realized_corr.png"):
    """
    Aligns the L^p-norm features with rolling realized correlation on
    shared dates, reports Pearson correlation between each lh1_k*_norm
    series and realized correlation, and plots them together.
    """
    print(f"Computing {window}-day rolling realized correlation from raw returns...")
    realized_corr = compute_realized_avg_correlation(log_returns_df, window=window)

    norm_cols = sorted(
        [c for c in tda_features_df.columns if c.startswith('lh1_') and c.endswith('_norm')],
        key=lambda c: int(c.split('_k')[1].split('_')[0])
    )
    if not norm_cols:
        raise ValueError("No lh1_k*_norm columns found in tda_features_df.")

    common_idx = tda_features_df.index.intersection(realized_corr.dropna().index)
    print(f"Overlapping dates: {len(common_idx)}")

    aligned_norms = tda_features_df.loc[common_idx, norm_cols]
    aligned_corr = realized_corr.loc[common_idx]

    print("\nPearson correlation: lh1_k*_norm  vs.  realized avg pairwise correlation")
    print("_" * 60)
    results = {}
    for col in norm_cols:
        r = aligned_norms[col].corr(aligned_corr)
        results[col] = r
        print(f"  {col:18s}  r = {r:+.4f}")

    fig, axes = plt.subplots(len(norm_cols) + 1, 1, figsize=(14, 3 * (len(norm_cols) + 1)),
                             sharex=True)

    axes[0].plot(common_idx, aligned_corr, color='black', linewidth=0.9)
    axes[0].set_ylabel(f"realized avg corr\n({window}d rolling)")
    axes[0].grid(alpha=0.3)
    axes[0].set_title(
        "L^p-norm landscape series vs. realized pairwise correlation\n"
        "(top panel = ground truth; lower panels = topology features, "
        "r given in legend)"
    )

    for ax, col in zip(axes[1:], norm_cols):
        ax.plot(common_idx, aligned_norms[col], color='steelblue', linewidth=0.8,
                label=f"r = {results[col]:+.3f}")
        ax.set_ylabel(col)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")

    return results, aligned_norms, aligned_corr


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

    results, aligned_norms, aligned_corr = compare_lpnorm_to_realized_corr(
        tda_features, log_returns, window=ROLLING_WINDOW
    )

    print("\nReading the result:")
    print("  |r| > ~0.5-0.6  -> lh1_k*_norm is largely a noisy proxy for")
    print("    realized correlation -- explains the cyclical (non-crash-")
    print("    specific) pattern in the earlier diagnostic plot, and why")
    print("    it nudges DCC log-likelihood without passing the")
    print("    permutation test (DCC already has access to better, more")
    print("    direct information about realized correlation than this).")
    print("  |r| small/near 0 -> lh1_k*_norm is capturing something NOT")
    print("    explained by simple realized correlation -- worth digging")
    print("    into what that something is, since it isn't crash-specific")
    print("    either based on the earlier plot.")