"""
Permutation test for TopoDCC: does the topology landscape information
carry real signal, or is the ~183 nat in-sample improvement over baseline
just ~370 free parameters fitting noise?

Procedure:
  1. Fit TopoDCC once on the real (correctly time-aligned) landscape
     features -> real_ll.
  2. Shuffle the *rows* of the feature matrix n_permutations times. This
     breaks the day <-> topology link (each day gets a random other day's
     features) while keeping every feature's own marginal distribution,
     scale, and cross-feature correlation structure intact. Same ~370
     parameters, same optimizer, same n_iter -- the only thing that
     changes is whether the features carry information about *when*
     things happened.
  3. Refit TopoDCC from scratch on each shuffled version -> permuted_ll[i].
  4. Compare real_ll against the distribution of permuted_ll.

Reading the result:
  - real_ll far above the permuted distribution -> the topology features
    carry real, time-aligned information; the model is doing more than
    just exploiting extra parameters.
  - real_ll inside or barely above the permuted distribution -> the
    ~370 extra parameters are doing most of the work regardless of what
    you feed them; the topology content itself isn't earning its keep.

Usage:
  python src/models/permutation_test.py
"""

import os
import sys
import time
import concurrent.futures
import multiprocessing as mp

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.dcc_topo import fit_dcc_topo


def _permuted_fit_worker(seed, z_np, X_np, n_iter, lr):
    """
    Runs in a separate process. Shuffles the rows of X with this seed,
    fits TopoDCC via the exact same fit_dcc_topo used for the real run,
    returns only the final ll (cheap to ship back across the process
    boundary -- no need to return the whole model or history).
    """
    torch.set_num_threads(1)  # avoid every worker fighting for all cores

    rng = np.random.default_rng(seed)
    perm = rng.permutation(X_np.shape[0])
    X_shuffled = X_np[perm]

    z_df = pd.DataFrame(z_np)
    X_df = pd.DataFrame(X_shuffled)

    _, _, _, _, ll_hist = fit_dcc_topo(z_df, X_df, n_iter=n_iter, lr=lr, verbose=False)
    return {'seed': seed, 'll_final': ll_hist[-1]}


def run_permutation_test(garch_residuals_df, tda_features_df,
                         n_permutations=10, n_iter=500, lr=0.01,
                         n_jobs=None, seed0=0, real_ll=None):
    """
    Returns: real_ll (float), permuted_lls (np.ndarray), results_df (pd.DataFrame)

    real_ll: if you already have a final ll from a previous identical run
      (same features, same n_iter/lr, same dcc_topo.py code), pass it here
      to skip refitting on real features. Note this was one particular
      random init's result, not "the" answer for real features -- a fresh
      run could land a bit differently. Fine as a time-saver if nothing
      about the code/data/settings has changed since you got that number.
    """
    z_np = garch_residuals_df.values.astype('float32')
    X_np = tda_features_df.values.astype('float32')

    if real_ll is None:
        print("Fitting on REAL (correctly aligned) features...")
        t0 = time.time()
        _, _, _, _, real_ll_hist = fit_dcc_topo(
            garch_residuals_df, tda_features_df, n_iter=n_iter, lr=lr, verbose=False
        )
        real_ll = real_ll_hist[-1]
        print(f"  real_ll = {real_ll:.2f}  ({time.time() - t0:.1f}s)")
    else:
        print(f"Using cached real_ll = {real_ll:.2f} (skipping real fit)")

    if n_jobs is None:
        n_jobs = max(1, min(n_permutations, os.cpu_count() or 1))

    seeds = [seed0 + i for i in range(n_permutations)]
    rows = []

    print(f"\nFitting {n_permutations} SHUFFLED versions across {n_jobs} processes...")
    t0 = time.time()
    if n_jobs > 1:
        ctx = mp.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
            futures = {
                ex.submit(_permuted_fit_worker, s, z_np, X_np, n_iter, lr): s
                for s in seeds
            }
            for fut in concurrent.futures.as_completed(futures):
                row = fut.result()
                rows.append(row)
                print(f"    seed={row['seed']:3d}  ll_final={row['ll_final']:.2f}")
    else:
        for s in seeds:
            row = _permuted_fit_worker(s, z_np, X_np, n_iter, lr)
            rows.append(row)
            print(f"    seed={row['seed']:3d}  ll_final={row['ll_final']:.2f}")
    print(f"  done in {time.time() - t0:.1f}s")

    rows.sort(key=lambda r: r['seed'])
    permuted_lls = np.array([r['ll_final'] for r in rows])

    print("\nRESULTS")
    print("_" * 60)
    print(f"  real_ll              = {real_ll:.2f}")
    print(f"  permuted mean        = {permuted_lls.mean():.2f}")
    print(f"  permuted std         = {permuted_lls.std():.2f}")
    print(f"  permuted min / max   = {permuted_lls.min():.2f} / {permuted_lls.max():.2f}")
    print(f"  real - permuted mean = {real_ll - permuted_lls.mean():.2f}")
    n_as_good = int((permuted_lls >= real_ll).sum())
    print(f"  permuted runs >= real_ll: {n_as_good}/{n_permutations}")
    if n_as_good >= 1:
        print("  -> at least one shuffled-feature run matched or beat the real")
        print("     features. That's a warning sign the gap is more about")
        print("     parameter count than topology content.")
    else:
        print("  -> real features clearly outperformed every shuffled version.")
        print("     Consistent with the topology features carrying real signal.")

    return real_ll, permuted_lls, pd.DataFrame(rows)


if __name__ == "__main__":
    from src.data.loader import load_config

    config = load_config()
    paths = config['paths']
    topo_cfg = config['models']['topo']

    garch_residuals = pd.read_parquet(paths['garch_residuals'])
    tda_features    = pd.read_parquet(paths['tda_features_landscape'])

    if 'betti_0' in tda_features.columns:
        tda_features = tda_features.drop(columns=['betti_0'])

    assert (garch_residuals.index == tda_features.index).all(), \
        "Index mismatch — re-run tda_pipeline.py"

    print(f"Residuals: {garch_residuals.shape}")
    print(f"Features:  {tda_features.shape}")

    real_ll, permuted_lls, results_df = run_permutation_test(
        garch_residuals,
        tda_features,
        n_permutations=topo_cfg.get('n_permutations', 10),
        n_iter=topo_cfg['n_iter'],
        lr=topo_cfg['lr'],
        real_ll=-4603.30,  # reuse the known real-features result instead of refitting
    )

    out_path = paths.get('permutation_test', 'data/processed/permutation_test_results.npy')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, {
        'real_ll':      real_ll,
        'permuted_lls': permuted_lls,
        'results':      results_df.to_dict(),
    })
    print(f"\nSaved to {out_path}")