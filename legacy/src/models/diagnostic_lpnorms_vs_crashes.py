# -*- coding: utf-8 -*-

"""
Diagnostic: does the L^p-norm landscape series rise before known crashes?

This is independent of DCC entirely -- it's a direct check on whether the
Gidea & Katz pattern (L^p-norm of H1 persistence landscape rises in the
months before a crash) is actually present in *this* pipeline's output,
given your specific embed_dim/PCA/grid choices.

Why this matters now: both permutation tests (raw 186-feature landscape,
and the 9-feature L^p-norm summary) failed when topology was asked to
explain DAILY DCC correlation-parameter variation averaged across the
whole 2000-2025 sample -- mostly calm markets. That's a different, much
harder claim than what Gidea & Katz actually tested. Before redesigning
the modelling target, it's worth checking the more basic question this
plot answers: is the topological signal even visible in your data at all,
near the events it's supposed to be sensitive to?

Two outcomes, two different next steps:
  - Norms visibly rise before crash windows -> signal is present upstream,
    problem is in how it's being fed to DCC / the target it's predicting.
    Proceed to redefining the target (crash-window classification /
    near-term correlation jump prediction) as planned.
  - Norms don't rise, or rise/fall with no relationship to crash timing ->
    signal may be getting destroyed before it ever reaches the landscape
    step (embed_dim=80, PCA to 15 dims, grid construction). Worth
    revisiting those choices before testing any new DCC target, since a
    new target won't help if the upstream signal isn't there.

Usage:
  python diagnostic_lpnorm_vs_crashes.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Known US equity market stress / crash periods within a 2000-2025 window.
# Dates are approximate windows of acute stress, not single-day events --
# Gidea & Katz look at the ~250 trading days *before* the peak/crash date,
# so each entry here marks the crash/peak itself; the run-up period is
# whatever precedes it on the plot.
CRASH_PERIODS = [
    ("Dotcom peak/crash",    "2000-03-10", "2002-10-09"),
    ("2008 GFC (Lehman)",    "2008-09-15", "2009-03-09"),
    ("2011 US downgrade / EU debt", "2011-07-01", "2011-10-03"),
    ("2018 Q4 selloff",      "2018-10-01", "2018-12-24"),
    ("COVID crash",          "2020-02-19", "2020-03-23"),
    ("2022 bear market",     "2022-01-03", "2022-10-12"),
]


def plot_lpnorm_diagnostic(tda_features_df, out_path="lpnorm_vs_crashes.png"):
    """
    Plots lh1_k0_norm, lh1_k1_norm, lh1_k2_norm over time, shading known
    crash/stress windows so the pre-crash-rise pattern (or its absence)
    is visible directly.
    """
    norm_cols = [c for c in tda_features_df.columns if c.startswith('lh1_') and c.endswith('_norm')]
    if not norm_cols:
        raise ValueError(
            "No lh1_k*_norm columns found -- did you pass the L^p-norm "
            "reduced dataframe (from lp_norm_features.py), not the raw "
            "186-column landscape one?"
        )
    norm_cols = sorted(norm_cols, key=lambda c: int(c.split('_k')[1].split('_')[0]))

    fig, axes = plt.subplots(len(norm_cols), 1, figsize=(14, 3 * len(norm_cols)), sharex=True)
    if len(norm_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, norm_cols):
        ax.plot(tda_features_df.index, tda_features_df[col], linewidth=0.8, color='steelblue')

        # 60-day rolling mean to make the trend legible against day-to-day noise
        rolling = tda_features_df[col].rolling(60, min_periods=20).mean()
        ax.plot(tda_features_df.index, rolling, linewidth=1.6, color='darkred',
                 label='60-day rolling mean')

        for name, start, end in CRASH_PERIODS:
            start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
            if end_ts < tda_features_df.index[0] or start_ts > tda_features_df.index[-1]:
                continue  # outside data range
            ax.axvspan(start_ts, end_ts, color='orange', alpha=0.25)

        ax.set_ylabel(col)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)

    axes[0].set_title(
        "L^p-norm of H1 persistence landscape vs. known market stress periods\n"
        "(orange shading = crash/stress window; check whether the blue/red line "
        "rises in the months BEFORE each shaded region)"
    )
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    return fig


if __name__ == "__main__":
    from src.data.loader import load_config

    config = load_config()
    paths = config['paths']

    tda_features = pd.read_parquet(
        paths.get('tda_features_lpnorm', 'data/processed/tda_features_lpnorm.parquet')
    )

    print(f"Features shape: {tda_features.shape}")
    print(f"Date range: {tda_features.index[0].date()} -> {tda_features.index[-1].date()}")

    plot_lpnorm_diagnostic(tda_features, out_path="lpnorm_vs_crashes.png")