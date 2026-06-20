import torch
import torch.nn as nn
import numpy as np
import pandas as pd

def compute_Q_bar(z_t):
    T = z_t.shape[0]
    return (z_t.T @ z_t) / T

def dcc_recursion_torch(z_t, a, b, Q_bar):
    T, N = z_t.shape
    
    Q_t = Q_bar.clone()
    Q_seq = torch.zeros(T, N, N, dtype=z_t.dtype)
    Q_seq[0] = Q_t
    
    for t in range(1, T):
        z_outer = torch.outer(z_t[t-1], z_t[t-1])
        Q_t = (1 - a - b) * Q_bar + a * z_outer + b * Q_t
        Q_seq[t] = Q_t

    diag_Q = torch.sqrt(torch.diagonal(Q_seq, dim1=1, dim2=2))
    R_seq = Q_seq / torch.einsum('ti,tj->tij', diag_Q, diag_Q)

    sign, log_det = torch.linalg.slogdet(R_seq)
    R_inv = torch.linalg.inv(R_seq)
    mahal = torch.sum(z_t * (R_inv @ z_t.unsqueeze(-1)).squeeze(-1), dim=1)
    ll = -0.5 * (log_det.sum() + mahal.sum())

    return R_seq, ll

def fit_dcc_baseline(garch_residuals_df, n_iter=500, lr=0.01, verbose=True):
    z_t = torch.tensor(garch_residuals_df.values, dtype=torch.float32)
    T, N = z_t.shape
    Q_bar = compute_Q_bar(z_t)

    # Stationarity fix: a and b used to be capped independently (a<=0.3,
    # b<=0.97) with nothing stopping a+b >= 1, which breaks the DCC
    # stationarity condition and can make Q_t non-PSD without erroring.
    # Reparameterize as total persistence `s = a+b` and split fraction `p`,
    # so a = s*p, b = s*(1-p) -> a+b = s < max_sum is guaranteed by
    # construction, for every value the raw params can take. No clamping,
    # no masking, no kink in the gradient.
    max_sum = 0.9998  # matches the cap used in TopoDCC for comparability
    total_raw = nn.Parameter(torch.tensor(4.0,  dtype=torch.float32))  # sigmoid(4.0)*0.9998 ≈ 0.980
    split_raw = nn.Parameter(torch.tensor(-3.5, dtype=torch.float32))  # sigmoid(-3.5)       ≈ 0.029
    optimizer = torch.optim.Adam([total_raw, split_raw], lr=lr)
    ll_history = []

    for i in range(n_iter):
        optimizer.zero_grad()
        s = torch.sigmoid(total_raw) * max_sum
        p = torch.sigmoid(split_raw)
        a = s * p
        b = s * (1 - p)
        R_seq, ll = dcc_recursion_torch(z_t, a, b, Q_bar)
        loss = -ll
        loss.backward()
        optimizer.step()
        ll_history.append(ll.item())

        if verbose and (i % 50 == 0 or i == n_iter - 1):
            print(f"  Iter {i:4d} | a={a.item():.6f} | b={b.item():.6f} | "
                  f"a+b={a.item()+b.item():.6f} | ll={ll.item():.2f}")

    with torch.no_grad():
        s = torch.sigmoid(total_raw) * max_sum
        p = torch.sigmoid(split_raw)
        a_final = s * p
        b_final = s * (1 - p)
        R_seq, _ = dcc_recursion_torch(z_t, a_final, b_final, Q_bar)

    return a_final.item(), b_final.item(), R_seq, ll_history

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.data.loader import load_config

    config = load_config()
    model_cfg = config['models']['baseline']
    paths = config['paths']

    garch_residuals = pd.read_parquet(paths['garch_residuals'])
    a, b, R_seq, ll_history = fit_dcc_baseline(
        garch_residuals,
        n_iter=model_cfg['n_iter'],
        lr=model_cfg['lr'],
        verbose=True
    )
    os.makedirs(os.path.dirname(paths['dcc_baseline']), exist_ok=True)
    np.save(paths['dcc_baseline'], {
        'a': a,
        'b': b,
        'll_final': ll_history[-1],
        'll_history': ll_history,
        'R_seq': R_seq.detach().numpy()
    })
    print(f" Saved to {paths['dcc_baseline']}")