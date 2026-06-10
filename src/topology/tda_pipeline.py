# -*- coding: utf-8 -*-

"""
TDA Pipeline — Persistence Landscape Version

Replaces persistence images with persistence landscapes, which are:
  - Evaluated on a fixed global filtration grid (comparable across time)
  - Mathematically stable (Lipschitz in bottleneck/Wasserstein metrics)
  - Free of per-window normalisation artefacts

Two-phase pipeline:
  Phase 1 (parallel): compute persistence diagrams + scalar features
  Phase 2 (sequential): build global filtration grid, compute landscapes

Feature groups produced:
  - lh0_k{k}_g{g}     : H0 landscape functions (k=0..N_LANDSCAPES-1)
  - lh1_k{k}_g{g}     : H1 landscape functions
  - betti_1            : number of significant H1 bars
  - entropy_h0         : H0 persistence entropy
  - entropy_h1         : H1 persistence entropy
  - max_persistence    : longest H1 bar
  - total_persistence  : sum of H1 bar lengths
  - wasserstein        : W_2 distance to previous window's H1 diagram
"""

import numpy as np
import pandas as pd
from ripser import ripser
from persim import wasserstein
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────

WINDOW         = 252    # 1-year rolling window (trading days)
STEP           = 1      # daily rolling
EMBED_DIM      = 80     # delay-embedding dimension
PCA_DIM        = 15     # PCA reduction before Ripser
MIN_PERSIST    = 1e-6   # minimum bar length to count as significant

# Persistence landscape settings
N_LANDSCAPES   = 3      # number of landscape functions λ_1, λ_2, λ_3
GRID_POINTS    = 30     # filtration grid resolution
GRID_QUANTILE  = 0.95   # quantile of max-death values used to set grid upper bound

# ── POINT CLOUD HELPERS ───────────────────────────────────────────────────────

def embed_multivariate(returns_window, embed_dim):
    W, n_assets = returns_window.shape
    if W < embed_dim:
        return np.empty((0, n_assets * embed_dim))
    n_points = W - embed_dim + 1
    X = np.zeros((n_points, n_assets * embed_dim))
    for i in range(n_points):
        for j in range(embed_dim):
            X[i, j * n_assets:(j + 1) * n_assets] = (
                returns_window[i + embed_dim - 1 - j, :]
            )
    return X


def normalize_pointcloud(X):
    X = X - X.mean(axis=0)
    X = X / (X.std(axis=0) + 1e-10)
    return X


def reduce_to_pca(X, n_components):
    k = min(n_components, X.shape[0], X.shape[1])
    if k < 2:
        return X
    pca = PCA(n_components=k, svd_solver='randomized', random_state=42)
    return pca.fit_transform(X)

# ── PERSISTENCE DIAGRAM HELPERS ───────────────────────────────────────────────

def compute_betti(dgm_h0, dgm_h1):
    betti_0 = int(np.sum(np.isfinite(dgm_h0[:, 1])))
    betti_1 = int(np.sum(np.isfinite(dgm_h1[:, 1])))
    return betti_0, betti_1


def compute_persistence_stats(dgm):
    finite = np.isfinite(dgm[:, 1])
    bars   = dgm[finite]
    if len(bars) == 0:
        return 0.0, 0.0, 0.0
    lengths = bars[:, 1] - bars[:, 0]
    lengths = lengths[lengths > MIN_PERSIST]
    if len(lengths) == 0:
        return 0.0, 0.0, 0.0
    max_pers   = float(np.max(lengths))
    total_pers = float(np.sum(lengths))
    p          = lengths / (lengths.sum() + 1e-10)
    entropy    = float(-(p * np.log(p + 1e-10)).sum())
    return max_pers, total_pers, entropy

# ── PERSISTENCE LANDSCAPE ─────────────────────────────────────────────────────

def compute_persistence_landscape(dgm, grid, n_landscapes=N_LANDSCAPES):
    """
    Persistence landscape evaluated on a fixed filtration grid.

    For each bar (b, d) the tent function is:
        f(s) = max(0, min(s - b, d - s))

    λ_k(s) = k-th largest tent value at filtration value s.

    Returns: (n_landscapes, len(grid)) array — zero-padded if diagram is small.
    """
    finite = np.isfinite(dgm[:, 1])
    bars   = dgm[finite]
    result = np.zeros((n_landscapes, len(grid)))

    if len(bars) == 0:
        return result

    s    = grid[np.newaxis, :]       # (1, G)
    b    = bars[:, 0:1]              # (N, 1)
    d    = bars[:, 1:2]              # (N, 1)
    tent = np.maximum(0.0, np.minimum(s - b, d - s))   # (N, G)

    # k-th landscape = k-th largest tent value at each grid point
    sorted_tent = np.sort(tent, axis=0)[::-1]           # (N, G) descending
    k = min(n_landscapes, sorted_tent.shape[0])
    result[:k] = sorted_tent[:k]

    return result


def build_filtration_grid(all_dgms, n_points=GRID_POINTS, quantile=GRID_QUANTILE):
    """
    Build a global fixed filtration grid from all persistence diagrams.
    Uses the `quantile`-th percentile of finite death values as the upper bound.
    This makes landscape features comparable across all time windows.
    """
    max_deaths = []
    for dgm in all_dgms:
        if dgm is None or len(dgm) == 0:
            continue
        finite_deaths = dgm[np.isfinite(dgm[:, 1]), 1]
        if len(finite_deaths) > 0:
            max_deaths.append(float(finite_deaths.max()))

    grid_max = float(np.quantile(max_deaths, quantile)) if max_deaths else 1.0
    return np.linspace(0.0, grid_max, n_points)

# ── PER-WINDOW COMPUTATION (Phase 1) ─────────────────────────────────────────

def compute_window(s, e, returns_data, dates):
    """
    Phase 1: compute persistence diagrams and scalar features for window [s, e).
    Landscapes are NOT computed here — they require the global grid built in Phase 2.

    Returns: (scalar_features, dgm_h0, dgm_h1, end_date)
    """
    ret_win = returns_data[s:e]
    if ret_win.shape[0] < EMBED_DIM or np.isnan(ret_win).any():
        return None, None, None, dates[e - 1]

    X = embed_multivariate(ret_win, EMBED_DIM)
    if X.shape[0] < 3:
        return None, None, None, dates[e - 1]

    X = normalize_pointcloud(X)
    Y = reduce_to_pca(X, PCA_DIM)
    if np.isnan(Y).any():
        return None, None, None, dates[e - 1]

    try:
        result = ripser(Y, maxdim=1)
        dgm_h0 = result['dgms'][0]
        dgm_h1 = result['dgms'][1]
    except Exception:
        return None, None, None, dates[e - 1]

    _, betti_1 = compute_betti(dgm_h0, dgm_h1)
    _, _, entropy_h0 = compute_persistence_stats(dgm_h0)
    max_pers, total_pers, entropy_h1 = compute_persistence_stats(dgm_h1)

    scalar_features = {
        'betti_1':          betti_1,
        'entropy_h0':       entropy_h0,
        'entropy_h1':       entropy_h1,
        'max_persistence':  max_pers,
        'total_persistence': total_pers,
        'wasserstein':      0.0,    # filled in sequentially after parallel loop
    }

    return scalar_features, dgm_h0, dgm_h1, dates[e - 1]

# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def run_tda_pipeline(log_returns_df,
                     window=WINDOW,
                     step=STEP,
                     n_jobs=-1,
                     verbose=True):
    """
    Main TDA pipeline — persistence landscape version.

    Args:
        log_returns_df : DataFrame (n_days x n_assets), date-indexed
        window         : rolling window size in trading days
        step           : step between windows
        n_jobs         : parallel workers (-1 = all cores)
        verbose        : print progress

    Returns:
        tda_df         : DataFrame of topological features, date-indexed
        grid           : the global filtration grid (save for out-of-sample use)
    """
    returns_data = log_returns_df.values
    dates        = log_returns_df.index
    n_days       = len(returns_data)
    slices       = [(s, s + window) for s in range(0, n_days - window + 1, step)]

    if verbose:
        print("=" * 60)
        print("TDA PIPELINE — PERSISTENCE LANDSCAPE VERSION")
        print("=" * 60)
        print(f"  Window:         {window} days")
        print(f"  Step:           {step} days")
        print(f"  Embed dim:      {EMBED_DIM}")
        print(f"  PCA dim (Rips): {PCA_DIM}")
        print(f"  Landscapes:     {N_LANDSCAPES} functions x {GRID_POINTS} grid points")
        print(f"  Data shape:     {returns_data.shape}")
        print(f"  Total windows:  {len(slices)}\n")
        print("Phase 1: computing persistence diagrams (parallel)...")

    # ── Phase 1: parallel diagram computation ─────────────────────────────────
    raw_results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(compute_window)(s, e, returns_data, dates)
        for s, e in slices
    )

    # Filter valid windows
    valid = [(sf, dh0, dh1, dt)
             for sf, dh0, dh1, dt in raw_results
             if sf is not None]

    if verbose:
        print(f"\n  Valid windows: {len(valid)} / {len(slices)}")

    scalar_list = [v[0] for v in valid]
    dgms_h0     = [v[1] for v in valid]
    dgms_h1     = [v[2] for v in valid]
    date_list   = [v[3] for v in valid]

    # Sequential Wasserstein (requires consecutive diagrams)
    prev_dgm = None
    for i, dgm_h1 in enumerate(dgms_h1):
        if prev_dgm is not None and dgm_h1 is not None:
            if len(dgm_h1) > 0 and len(prev_dgm) > 0:
                try:
                    scalar_list[i]['wasserstein'] = float(
                        wasserstein(dgm_h1, prev_dgm)
                    )
                except Exception:
                    pass
        if dgm_h1 is not None:
            prev_dgm = dgm_h1

    # ── Phase 2: build global grid, compute landscapes ─────────────────────────
    if verbose:
        print("\nPhase 2: building global filtration grid...")

    grid = build_filtration_grid(dgms_h0 + dgms_h1,
                                 n_points=GRID_POINTS,
                                 quantile=GRID_QUANTILE)

    if verbose:
        print(f"  Grid: 0 -> {grid[-1]:.4f}  ({GRID_POINTS} points, "
              f"{GRID_QUANTILE:.0%} quantile of max-death values)")
        print(f"\nPhase 2: computing landscapes on fixed grid...")

    feature_list = []
    for i, (sf, dh0, dh1) in enumerate(zip(scalar_list, dgms_h0, dgms_h1)):
        row = dict(sf)  # copy scalar features

        # H0 landscapes
        lh0 = compute_persistence_landscape(dh0, grid, N_LANDSCAPES)
        for k in range(N_LANDSCAPES):
            for g in range(GRID_POINTS):
                row[f'lh0_k{k}_g{g}'] = float(lh0[k, g])

        # H1 landscapes
        lh1 = compute_persistence_landscape(dh1, grid, N_LANDSCAPES)
        for k in range(N_LANDSCAPES):
            for g in range(GRID_POINTS):
                row[f'lh1_k{k}_g{g}'] = float(lh1[k, g])

        feature_list.append(row)

    tda_df = pd.DataFrame(feature_list, index=pd.DatetimeIndex(date_list))

    n_scalar    = 6
    n_landscape = 2 * N_LANDSCAPES * GRID_POINTS
    if verbose:
        print(f"\nTDA pipeline complete")
        print(f"  Output shape:   {tda_df.shape}")
        print(f"  Date range:     {tda_df.index[0].date()} -> {tda_df.index[-1].date()}")
        print(f"  Scalar features:    {n_scalar}")
        print(f"  Landscape features: {n_landscape}  "
              f"({N_LANDSCAPES} functions x {GRID_POINTS} pts x 2 degrees)")
        print(f"  Total features:     {tda_df.shape[1]}")
        print(f"\n  Summary statistics:")
        print(tda_df[['betti_1', 'entropy_h1', 'max_persistence',
                       'total_persistence', 'wasserstein']].describe().round(4))

    return tda_df, grid

# ── STANDALONE EXECUTION ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import joblib

    log_returns = pd.read_parquet('data/processed/log_returns.parquet')

    tda_features, grid = run_tda_pipeline(log_returns, n_jobs=-1, verbose=True)

    tda_features.to_parquet('data/processed/tda_features_landscape.parquet')
    print("\nSaved to data/processed/tda_features_landscape.parquet")

    np.save('data/processed/landscape_grid.npy', grid)
    print("Saved global filtration grid to data/processed/landscape_grid.npy")
