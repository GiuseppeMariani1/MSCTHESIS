"""
Regularised Topology-DCC with Ridge (L2) penalty.

Why Ridge here:
  Landscape features are highly correlated by construction — adjacent grid
  points and adjacent landscape levels move together. Ridge handles collinear
  predictors well: it shrinks all weights proportionally rather than picking
  one arbitrarily (Lasso) or zeroing groups (Group-Lasso). Elastic Net would
  also work and is tried here as a comparison.

Workflow:
  1. Standardise features.
  2. 80/20 chronological train/test split.
  3. Grid-search over lambda_reg values.
  4. For each lambda: fit on train, evaluate log-likelihood on test.
  5. Save best model and results.

Usage:
  python src/models/dcc_topo_reg.py
"""

import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure the repository root is on PYTHONPATH when running the script directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.dcc_topo import TopoDCC, dcc_topo_recursion, compute_Q_bar


#REGULARISED TRAINING

def fit_dcc_topo_reg(z_train, X_train, Q_bar,
                     lambda_l2=1e-2,
                     lambda_l1=0.0,
                     n_iter=500,
                     lr=0.01,
                     verbose=False):
    """
    Fit TopoDCC with elastic-net regularisation on training data.

    Loss = -ll(train) + lambda_l2 * ||w||_2^2 + lambda_l1 * ||w||_1

    Setting lambda_l1=0 gives pure Ridge.
    Setting lambda_l2=0 gives pure Lasso.

    Returns: fitted model, ll_history (train)
    """
    n_features = X_train.shape[1]
    model      = TopoDCC(n_features)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    ll_history = []

    for i in range(n_iter):
        optimizer.zero_grad()
        a_seq, b_seq = model(X_train)
        _, ll        = dcc_topo_recursion(z_train, a_seq, b_seq, Q_bar)

        l2_pen = model.w_a.pow(2).sum() + model.w_b.pow(2).sum()
        l1_pen = model.w_a.abs().sum()  + model.w_b.abs().sum()
        loss   = -ll + lambda_l2 * l2_pen + lambda_l1 * l1_pen

        loss.backward()
        optimizer.step()
        ll_history.append(ll.item())

        if verbose and (i % 100 == 0 or i == n_iter - 1):
            a_seq_d, b_seq_d = a_seq.detach(), b_seq.detach()
            print(f"  iter {i:4d} | ll={ll.item():.2f} | "
                  f"a={a_seq_d.mean():.4f}±{a_seq_d.std():.4f} | "
                  f"b={b_seq_d.mean():.4f}±{b_seq_d.std():.4f}")

    return model, ll_history


def eval_oos_ll(model, z_t, X_t, Q_bar, train_size):
    """
    Run the full DCC recursion over all data (so test period is warm-started
    from training history), then return the log-likelihood on the test portion.
    """
    with torch.no_grad():
        a_seq, b_seq = model(X_t)
        R_seq, _     = dcc_topo_recursion(z_t, a_seq, b_seq, Q_bar)

        z_test = z_t[train_size:]
        R_test = R_seq[train_size:]

        sign, log_det = torch.linalg.slogdet(R_test)
        R_inv  = torch.linalg.inv(R_test)
        mahal  = torch.sum(
            z_test * (R_inv @ z_test.unsqueeze(-1)).squeeze(-1), dim=1
        )
        ll_test = -0.5 * (log_det.sum() + mahal.sum())

    return ll_test.item()


# LAMBDA GRID SEARCH

LAMBDA_GRID = [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1]

def lambda_search(z_t, X_t, Q_bar, train_size,
                  lambda_grid=LAMBDA_GRID,
                  n_iter=500, lr=0.01):
    """
    Fit one model per lambda value, report train and test log-likelihoods.
    Returns a DataFrame of results sorted by test ll descending.
    """
    z_train = z_t[:train_size]
    X_train = X_t[:train_size]

    rows = []
    for lam in lambda_grid:
        print(f"\n  lambda_l2={lam:.0e}  fitting...")
        model, ll_hist = fit_dcc_topo_reg(
            z_train, X_train, Q_bar,
            lambda_l2=lam, n_iter=n_iter, lr=lr, verbose=False
        )
        ll_train = ll_hist[-1]
        ll_test  = eval_oos_ll(model, z_t, X_t, Q_bar, train_size)

        with torch.no_grad():
            w_norm = (model.w_a.pow(2).sum() + model.w_b.pow(2).sum()).sqrt().item()
            a_seq, b_seq = model(X_t[:train_size])
            a_std = a_seq.std().item()
            b_std = b_seq.std().item()

        rows.append({
            'lambda_l2':  lam,
            'll_train':   round(ll_train, 2),
            'll_test':    round(ll_test,  2),
            '|w|_2':      round(w_norm,   4),
            'a_std':      round(a_std,    4),
            'b_std':      round(b_std,    4),
        })
        print(f"    ll_train={ll_train:.2f}  ll_test={ll_test:.2f}  |w|={w_norm:.4f}")

    results_df = pd.DataFrame(rows).sort_values('ll_test', ascending=False)
    return results_df


# MAIN

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.data.loader import load_config

    config = load_config()
    paths = config['paths']

    # Load data
    garch_residuals = pd.read_parquet(paths['garch_residuals'])
    tda_features    = pd.read_parquet(paths['tda_features_landscape'])

    # Drop betti_0 if present (constant)
    if 'betti_0' in tda_features.columns:
        tda_features = tda_features.drop(columns=['betti_0'])

    assert (garch_residuals.index == tda_features.index).all(), \
        "Index mismatch — re-run tda_pipeline.py"

    print(f"Residuals: {garch_residuals.shape}")
    print(f"Features:  {tda_features.shape}")

    # Tensors
    z_t = torch.tensor(garch_residuals.values, dtype=torch.float32)

    X_raw  = tda_features.values
    X_mean = X_raw.mean(axis=0)
    X_std  = X_raw.std(axis=0) + 1e-8
    X_t    = torch.tensor((X_raw - X_mean) / X_std, dtype=torch.float32)

    config = load_config()
    paths = config['paths']
    topo_reg_cfg = config['models']['topo_reg']
    eval_cfg = config['evaluation']

    T = z_t.shape[0]
    train_size = int(T * (1.0 - eval_cfg['test_size']))
    print(f"\nTrain: {train_size} days  |  Test: {T - train_size} days")

    Q_bar = compute_Q_bar(z_t[:train_size])

    # Lambda grid search
    print("\n" + "=" * 60)
    print("LAMBDA GRID SEARCH (Ridge regularisation)")
    print("=" * 60)

    results = lambda_search(
        z_t,
        X_t,
        Q_bar,
        train_size,
        lambda_grid=topo_reg_cfg['lambda_grid'],
        n_iter=topo_reg_cfg['n_iter'],
        lr=topo_reg_cfg['lr']
    )

    print("\n" + "=" * 60)
    print("RESULTS (sorted by out-of-sample ll)")
    print("=" * 60)
    print(results.to_string(index=False))

    #  Fit best model 
    best_lambda = results.iloc[0]['lambda_l2']
    print(f"\nBest lambda_l2 = {best_lambda:.0e}")
    print("Fitting final model on full data...")

    model, ll_hist = fit_dcc_topo_reg(
        z_t,
        X_t,
        compute_Q_bar(z_t),
        lambda_l2=best_lambda,
        n_iter=topo_reg_cfg['n_iter'],
        lr=topo_reg_cfg['lr'],
        verbose=True
    )

    with torch.no_grad():
        a_seq, b_seq = model(X_t)
        R_seq, _     = dcc_topo_recursion(z_t, a_seq, b_seq, compute_Q_bar(z_t))

    ll_final = ll_hist[-1]
    print(f"\nFinal ll (full data): {ll_final:.2f}")
    print(f"Improvement over baseline (-4790.84): {ll_final - (-4790.84):.2f}")

    #  Save
    os.makedirs(os.path.dirname(paths['dcc_topo_reg']), exist_ok=True)
    np.save(paths['dcc_topo_reg'], {
        'a_seq':       a_seq.detach().numpy(),
        'b_seq':       b_seq.detach().numpy(),
        'R_seq':       R_seq.detach().numpy(),
        'll_history':  ll_hist,
        'll_final':    ll_final,
        'lambda_l2':   best_lambda,
        'w_a':         model.w_a.detach().numpy(),
        'w_b':         model.w_b.detach().numpy(),
        'lambda_results': results.to_dict(),
        'X_mean':      X_mean,
        'X_std':       X_std,
    })
    print(f"Saved to {paths['dcc_topo_reg']}")
