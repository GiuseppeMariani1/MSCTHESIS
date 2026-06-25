# -*- coding: utf-8 -*-
"""
diagnostic_lpnorm_vs_crashes.py

Plots the L^p-norm landscape series (lh1_k*_norm) over time with
vertical lines marking known crisis periods, to check whether the
topology signal rises prior to financial stress events as per
Gidea & Katz (2018).

Crisis windows marked:
  - Dotcom crash:     2000-03-10
  - GFC peak:         2008-09-15 (Lehman)
  - European debt:    2011-08-01
  - Vol spike:        2018-12-24
  - COVID crash:      2020-03-20
  - Rate hike shock:  2022-01-01

Usage:
  python scripts/diagnostic_lpnorm_vs_crashes.py
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.loader import load_config

CRISIS_DATES = {
    'GFC\n(Sep 2008)':      '2008-09-15',
    'EU Debt\n(Aug 2011)':  '2011-08-01',
    'Vol Spike\n(Dec 2018)':'2018-12-24',
    'COVID\n(Mar 2020)':    '2020-03-20',
    'Rates\n(Jan 2022)':    '2022-01-01',
}

if __name__ == "__main__":
    config = load_config()
    paths  = config['paths']

    tda_features = pd.read_parquet(
        paths.get('tda_features_lpnorm', 'data/processed/tda_features_lpnorm.parquet')
    )

    norm_cols = sorted(
        [c for c in tda_features.columns if c.startswith('lh1_') and c.endswith('_norm')],
        key=lambda c: int(c.split('_k')[1].split('_')[0])
    )

    print(f"Features shape: {tda_features.shape}")
    print(f"Date range: {tda_features.index[0].date()} -> {tda_features.index[-1].date()}")
    print(f"Norm columns: {norm_cols}")

    fig, axes = plt.subplots(len(norm_cols), 1, figsize=(16, 3.5 * len(norm_cols)), sharex=True)
    if len(norm_cols) == 1:
        axes = [axes]

    colors = ['steelblue', 'darkorange', 'seagreen']

    for ax, col, color in zip(axes, norm_cols, colors):
        # 60-day rolling mean to smooth noise
        smoothed = tda_features[col].rolling(60).mean()
        ax.plot(tda_features.index, tda_features[col],
                color=color, linewidth=0.5, alpha=0.4, label='daily')
        ax.plot(tda_features.index, smoothed,
                color=color, linewidth=1.5, label='60d mean')

        for label, date_str in CRISIS_DATES.items():
            crisis_date = pd.Timestamp(date_str)
            if tda_features.index[0] <= crisis_date <= tda_features.index[-1]:
                ax.axvline(crisis_date, color='red', linewidth=1.2,
                           linestyle='--', alpha=0.8)
                ax.text(crisis_date, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else smoothed.max(),
                        label, fontsize=7, color='red',
                        ha='center', va='bottom', rotation=0)

        ax.set_ylabel(col, fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))

    axes[0].set_title(
        "L^p-norm Persistence Landscape Series vs. Crisis Periods\n"
        "SPY, EEM, GLD, TLT, DBC  |  embed_dim=5, pca_dim=10, window=250\n"
        "Red dashed lines = crisis dates; Gidea & Katz (2018) predict rising norms before crashes",
        fontsize=11
    )
    axes[-1].set_xlabel("Date", fontsize=10)

    plt.tight_layout()
    out_path = "data/processed/diagnostic_lpnorm_vs_crashes.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved to {out_path}")
    plt.show()