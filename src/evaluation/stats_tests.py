# -*- coding: utf-8 -*-
"""
stats_tests.py — diagnostic battery for thesis defence.

Tests stacked here:
  1. ADF stationarity on lh1_k*_norm series
  2. a_t / b_t summary stats and crisis period behaviour
  (add more below as needed)

Usage:
  python src/evaluation/stats_tests.py
"""

import os
import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.loader import load_config

config = load_config()
paths  = config['paths']

tda   = pd.read_parquet(paths.get('tda_features_lpnorm', 'data/processed/tda_features_lpnorm.parquet'))
garch = pd.read_parquet(paths['garch_residuals'])

# ── 1. ADF STATIONARITY ───────────────────────────────────────────────────────
print("-" * 60)
print("TEST 1: ADF Stationarity — lh1_k*_norm series")
print("-" * 60)
norm_cols = [c for c in tda.columns if c.startswith('lh1_') and c.endswith('_norm')]
for col in sorted(norm_cols):
    stat, pval, _, nobs, _, _ = adfuller(tda[col].dropna(), autolag='AIC')
    result = 'STATIONARY' if pval < 0.05 else 'NON-STATIONARY'
    print(f"  {col:18s}  ADF={stat:7.3f}  p={pval:.4f}  [{result}]")

# ── 2. a_t / b_t SUMMARY AND CRISIS BEHAVIOUR ────────────────────────────────
print()
print("-" * 60)
print("TEST 2: TopoDCC a_t / b_t parameter diagnostics")
print("-" * 60)

results_path = paths.get('dcc_topo_lpnorm', 'data/processed/dcc_topo_lpnorm_results.npy')
results = np.load(results_path, allow_pickle=True).item()
a = results['a_seq']
b = results['b_seq']

print(f"\n  a_t:  mean={a.mean():.4f}  std={a.std():.4f}  "
      f"min={a.min():.4f}  max={a.max():.4f}")
print(f"  b_t:  mean={b.mean():.4f}  std={b.std():.4f}  "
      f"min={b.min():.4f}  max={b.max():.4f}")
print(f"  a+b:  mean={(a+b).mean():.4f}  std={(a+b).std():.4f}")

crisis_windows = {
    'GFC       (2008-07 to 2009-03)': ('2008-07-01', '2009-03-31'),
    'EU Debt   (2011-07 to 2012-01)': ('2011-07-01', '2012-01-31'),
    'COVID     (2020-02 to 2020-06)': ('2020-02-01', '2020-06-30'),
    'Rates     (2022-01 to 2022-12)': ('2022-01-01', '2022-12-31'),
}

common = garch.index.intersection(tda.index)
dates  = common

print(f"\n  Crisis window averages vs full-sample:")
print(f"  {'Period':<35} {'a_t mean':>10} {'b_t mean':>10} {'a+b mean':>10}")
print(f"  {'-'*65}")
print(f"  {'Full sample':<35} {a.mean():>10.4f} {b.mean():>10.4f} {(a+b).mean():>10.4f}")

a_series = pd.Series(a, index=dates)
b_series = pd.Series(b, index=dates)

for label, (start, end) in crisis_windows.items():
    mask = (dates >= start) & (dates <= end)
    if mask.sum() == 0:
        print(f"  {label:<35} {'no data':>10}")
        continue
    a_crisis = a_series[mask].mean()
    b_crisis = b_series[mask].mean()
    print(f"  {label:<35} {a_crisis:>10.4f} {b_crisis:>10.4f} {a_crisis+b_crisis:>10.4f}")

# ── ADD MORE TESTS BELOW ──────────────────────────────────────────────────────

print()
print("Done.")