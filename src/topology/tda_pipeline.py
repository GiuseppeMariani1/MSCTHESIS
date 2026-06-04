# -*- coding: utf-8 -*-

"""
TDA Pipeline for DCC State Variables
Computes topological features from multivariate return embeddings.
Output: DataFrame aligned by date with topological state features.

KEY CHANGE vs v1
  - Replaced scalar persistence summaries (4 numbers) with
    persistence images (PI_RESOLUTION^2 numbers per diagram).
  - Increased PCA_DIM from 5 → 15 to retain more geometry before Ripser.
  - Kept entropy + wasserstein as lightweight diagnostics.
  - Added PCA compression of persistence images across time so that
    X_t fed into the DCC remains a manageable vector (PI_N_COMPONENTS).

Feature groups produced:
  - pi_h0_pc_{0..PI_N_COMPONENTS-1}   : persistence image PCs for H0
  - pi_h1_pc_{0..PI_N_COMPONENTS-1}   : persistence image PCs for H1
  - betti_0, betti_1                  : Betti numbers
  - entropy_h0, entropy_h1            : persistence entropy
  - wasserstein                       : W_2 distance to prev diagram
  - max_persistence, total_persistence: H1 scalar stats (kept for interp.)
"""

import numpy as np
import pandas as pd
from ripser import ripser
from persim import wasserstein
from sklearn.decomposition import PCA, IncrementalPCA
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

#CONFIG

WINDOW         = 252    # 1-year rolling window (trading days)
STEP           = 1      # daily rolling
EMBED_DIM      = 80     # delay-embedding dimension
PCA_DIM        = 15     # ↑ from 5 → 15: keep more geometry before Ripser
MIN_PERSIST    = 1e-6   # minimum bar length to count

# Persistence image settings
PI_RESOLUTION  = 20     # 20×20 grid → 400-d vector per diagram
PI_SIGMA       = 0.15   # Gaussian bandwidth (fraction of range)
PI_WEIGHT      = 1      # weight bars by persistence^PI_WEIGHT

# Dimensionality of persistence image PCs fed into DCC
# 400-d images → compress to PI_N_COMPONENTS across time
PI_N_COMPONENTS = 10

#  POINT CLOUD HELPERS

def embed_multivariate(returns_window, embed_dim):
    """
    Delay-embedding of multivariate returns.

    returns_window : (W, n_assets)
    Returns X      : (n_points, n_assets * embed_dim)
    """
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
    """Centre and unit-variance scale."""
    X = X - X.mean(axis=0)
    X = X / (X.std(axis=0) + 1e-10)
    return X


def reduce_to_pca(X, n_components):
    """PCA compression before Ripser."""
    k = min(n_components, X.shape[0], X.shape[1])
    if k < 2:
        return X
    pca = PCA(n_components=k, svd_solver='randomized', random_state=42)
    return pca.fit_transform(X)

#  PERSISTENCE DIAGRAM HELPERS

def compute_betti(dgm_h0, dgm_h1):
    """β₀ = finite H0 bars, β₁ = finite H1 bars."""
    betti_0 = int(np.sum(np.isfinite(dgm_h0[:, 1])))
    betti_1 = int(np.sum(np.isfinite(dgm_h1[:, 1])))
    return betti_0, betti_1


def compute_persistence_stats(dgm):
    """
    Lightweight scalar stats (kept for interpretability / diagnostics).
    Returns: max_pers, total_pers, entropy, n_significant
    """
    finite = np.isfinite(dgm[:, 1])
    bars   = dgm[finite]

    if len(bars) == 0:
        return 0.0, 0.0, 0.0, 0

    lengths = bars[:, 1] - bars[:, 0]
    lengths = lengths[lengths > MIN_PERSIST]

    if len(lengths) == 0:
        return 0.0, 0.0, 0.0, 0

    max_pers   = float(np.max(lengths))
    total_pers = float(np.sum(lengths))

    total = lengths.sum()
    p       = lengths / (total + 1e-10)
    entropy = float(-(p * np.log(p + 1e-10)).sum())

    threshold = np.percentile(lengths, 75) if len(lengths) > 1 else 0
    n_sig     = int(np.sum(lengths > threshold))

    return max_pers, total_pers, entropy, n_sig


def compute_persistence_image(dgm, resolution=PI_RESOLUTION,
                               sigma=PI_SIGMA, weight_power=PI_WEIGHT):
    """
    Vectorise a persistence diagram as a persistence image.

    Works in birth–persistence coordinates (standard for PIs):
      x-axis = birth value
      y-axis = persistence (= death - birth)

    Each bar (b, d) contributes a weighted Gaussian bump at (b, d-b).
    Weight = (d - b)^weight_power  (longer bars weighted more).

    Returns: flat np.array of length resolution^2
    """
    finite = np.isfinite(dgm[:, 1])
    bars   = dgm[finite]

    if len(bars) == 0:
        return np.zeros(resolution * resolution)

    births       = bars[:, 0]
    persistences = np.clip(bars[:, 1] - bars[:, 0], 0, None)
    weights      = persistences ** weight_power

    b_min = births.min();       b_max = births.max() + 1e-10
    p_min = 0.0;                p_max = persistences.max() + 1e-10

    b_grid = np.linspace(b_min, b_max, resolution)
    p_grid = np.linspace(p_min, p_max, resolution)

    bw_b = sigma * (b_max - b_min) + 1e-10
    bw_p = sigma * (p_max - p_min) + 1e-10

    image = np.zeros((resolution, resolution))
    for b, p, w in zip(births, persistences, weights):
        db     = (b_grid - b) / bw_b
        dp     = (p_grid - p) / bw_p
        image += w * np.outer(
            np.exp(-0.5 * db ** 2),
            np.exp(-0.5 * dp ** 2)
        )

    return image.flatten()

#  PER-WINDOW COMPUTATION

def compute_window(s, e, returns_data, dates):
    """
    Compute all TDA features for a single rolling window [s, e).

    Returns:
        (raw_features_dict, dgm_h1, end_date)
        raw_features_dict contains scalar features + raw PI vectors
        (PI PCA is applied *after* all windows are done, across time).
    """
    features = {}

    ret_win = returns_data[s:e]
    if ret_win.shape[0] < EMBED_DIM or np.isnan(ret_win).any():
        return None, None, dates[e - 1]

    # Point cloud
    X = embed_multivariate(ret_win, EMBED_DIM)
    if X.shape[0] < 3:
        return None, None, dates[e - 1]

    X = normalize_pointcloud(X)

    # PCA reduction (higher dim than before: PCA_DIM=15)
    Y = reduce_to_pca(X, PCA_DIM)
    if np.isnan(Y).any():
        return None, None, dates[e - 1]

    # Ripser
    try:
        result  = ripser(Y, maxdim=1)
        dgm_h0  = result['dgms'][0]
        dgm_h1  = result['dgms'][1]
    except Exception:
        return None, None, dates[e - 1]

    # Betti numbers
    betti_0, betti_1 = compute_betti(dgm_h0, dgm_h1)
    features['betti_0'] = betti_0
    features['betti_1'] = betti_1

    #Scalar stats (H1) — kept for interpretability
    max_pers, total_pers, entropy_h1, _ = compute_persistence_stats(dgm_h1)
    features['max_persistence']   = max_pers
    features['total_persistence'] = total_pers
    features['entropy_h1']        = entropy_h1

    #Entropy H0
    _, _, entropy_h0, _ = compute_persistence_stats(dgm_h0)
    features['entropy_h0'] = entropy_h0

    #Persistence images (raw vectors — PCA applied later)
    # H0
    pi_h0 = compute_persistence_image(dgm_h0)
    features['_pi_h0'] = pi_h0          # stored as array, extracted below

    # H1
    pi_h1 = compute_persistence_image(dgm_h1)
    features['_pi_h1'] = pi_h1

    # Wasserstein placeholder (recomputed with prev diagram after parallel loop)
    features['wasserstein'] = 0.0

    return features, dgm_h1, dates[e - 1]


#  POST-PROCESSING: PI → PCA ACROSS TIME

def compress_persistence_images(feature_list, n_components=PI_N_COMPONENTS):
    """
    Stack all per-window persistence images into a matrix and apply PCA
    across time.  This is the right place to do the compression because:

      - We need the full time series to fit the PCA (can't do per-window).
      - Compressing to n_components principal components across time gives
        a low-d representation that still captures most of the variance
        in the image sequence — far more than 4 hand-crafted scalars.

    Replaces _pi_h0 / _pi_h1 keys with pi_h0_pc_0 ... pi_h0_pc_{k-1}.
    """
    # Stack raw images
    pi_h0_matrix = np.stack([f['_pi_h0'] for f in feature_list])  # (T, 400)
    pi_h1_matrix = np.stack([f['_pi_h1'] for f in feature_list])  # (T, 400)

    k = min(n_components, pi_h0_matrix.shape[0], pi_h0_matrix.shape[1])

    pca_h0 = PCA(n_components=k, svd_solver='randomized', random_state=42)
    pca_h1 = PCA(n_components=k, svd_solver='randomized', random_state=42)

    scores_h0 = pca_h0.fit_transform(pi_h0_matrix)  # (T, k)
    scores_h1 = pca_h1.fit_transform(pi_h1_matrix)  # (T, k)

    var_h0 = pca_h0.explained_variance_ratio_.cumsum()[-1]
    var_h1 = pca_h1.explained_variance_ratio_.cumsum()[-1]
    print(f"  PI-PCA H0: {k} components explain {var_h0:.1%} of image variance")
    print(f"  PI-PCA H1: {k} components explain {var_h1:.1%} of image variance")

    # Write back into feature dicts
    for t, f in enumerate(feature_list):
        del f['_pi_h0']
        del f['_pi_h1']
        for i in range(k):
            f[f'pi_h0_pc_{i}'] = float(scores_h0[t, i])
            f[f'pi_h1_pc_{i}'] = float(scores_h1[t, i])

    return feature_list, pca_h0, pca_h1

#  MAIN PIPELINE

def run_tda_pipeline(log_returns_df,
                     window=WINDOW,
                     step=STEP,
                     n_jobs=-1,
                     verbose=True):
    """
    Main TDA pipeline.

    Args:
        log_returns_df  : DataFrame (n_days × n_assets), date-indexed
        window          : rolling window size (trading days)
        step            : step between windows
        n_jobs          : parallel workers (-1 = all cores)
        verbose         : print progress

    Returns:
        tda_df          : DataFrame of topological features, date-indexed
        pca_h0, pca_h1  : fitted PCA objects (for out-of-sample use)
    """
    returns_data = log_returns_df.values
    dates        = log_returns_df.index
    n_days       = len(returns_data)

    if verbose:
        print("=" * 60)
        print("TDA PIPELINE — PERSISTENCE IMAGE VERSION")
        print("=" * 60)
        print(f"  Window:         {window} days")
        print(f"  Step:           {step} days")
        print(f"  Embed dim:      {EMBED_DIM}")
        print(f"  PCA dim (Rips): {PCA_DIM}  (was 5)")
        print(f"  PI resolution:  {PI_RESOLUTION}×{PI_RESOLUTION} = "
              f"{PI_RESOLUTION**2} features per diagram")
        print(f"  PI PCA comps:   {PI_N_COMPONENTS}")
        print(f"  Data shape:     {returns_data.shape}")

    slices = [(s, s + window) for s in range(0, n_days - window + 1, step)]

    if verbose:
        print(f"  Total windows:  {len(slices)}\n")
        print("Computing TDA features (parallel)...")

    # Parallel window computation
    raw_results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(compute_window)(s, e, returns_data, dates)
        for s, e in slices
    )

    #Recompute Wasserstein sequentially (needs previous diagram)
    feature_list = []
    date_list    = []
    prev_dgm     = None

    for features, dgm_h1, end_date in raw_results:
        if features is None:
            continue

        if prev_dgm is not None and dgm_h1 is not None:
            if len(dgm_h1) > 0 and len(prev_dgm) > 0:
                try:
                    features['wasserstein'] = float(
                        wasserstein(dgm_h1, prev_dgm)
                    )
                except Exception:
                    features['wasserstein'] = 0.0

        feature_list.append(features)
        date_list.append(end_date)
        if dgm_h1 is not None:
            prev_dgm = dgm_h1

    # Compress persistence images via PCA across time
    if verbose:
        print(f"\nApplying PCA to persistence images across {len(feature_list)} windows...")

    feature_list, pca_h0, pca_h1 = compress_persistence_images(
        feature_list, n_components=PI_N_COMPONENTS
    )

    # Assemble DataFrame 
    tda_df = pd.DataFrame(feature_list, index=pd.DatetimeIndex(date_list))

    if verbose:
        print(f"\n DA pipeline complete")
        print(f"   Output shape:  {tda_df.shape}")
        print(f"   Date range:    {tda_df.index[0].date()} → "
              f"{tda_df.index[-1].date()}")
        print(f"\n   Feature columns ({len(tda_df.columns)}):")
        for col in tda_df.columns:
            print(f"     {col}")
        print(f"\n   Summary statistics:")
        print(tda_df.describe().round(4))

    return tda_df, pca_h0, pca_h1

#  STANDALONE EXECUTION

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/content/MSCTHESIS')

    log_returns = pd.read_parquet(
        '/content/MSCTHESIS/data/processed/log_returns.parquet'
    )

    tda_features, pca_h0, pca_h1 = run_tda_pipeline(
        log_returns, n_jobs=-1, verbose=True
    )

    # Save features
    tda_features.to_parquet(
        '/content/MSCTHESIS/data/processed/tda_features_pi.parquet'
    )
    print("\n✅ Saved to data/processed/tda_features_pi.parquet")

    # Optionally save PCA models for out-of-sample use
    import joblib
    joblib.dump(pca_h0, '/content/MSCTHESIS/data/processed/pca_h0.pkl')
    joblib.dump(pca_h1, '/content/MSCTHESIS/data/processed/pca_h1.pkl')
    print("Saved PCA models to data/processed/")
