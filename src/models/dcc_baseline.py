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

    a_raw = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
    b_raw = nn.Parameter(torch.tensor(0.5,  dtype=torch.float32))
    optimizer = torch.optim.Adam([a_raw, b_raw], lr=lr)
    ll_history = []

    for i in range(n_iter):
        optimizer.zero_grad()
        a = torch.sigmoid(a_raw) * 0.3
        b = torch.sigmoid(b_raw) * 0.97
        R_seq, ll = dcc_recursion_torch(z_t, a, b, Q_bar)
        loss = -ll
        loss.backward()
        optimizer.step()
        ll_history.append(ll.item())

        if verbose and (i % 50 == 0 or i == n_iter - 1):
            print(f"  Iter {i:4d} | a={a.item():.6f} | b={b.item():.6f} | "
                  f"a+b={a.item()+b.item():.6f} | ll={ll.item():.2f}")

    with torch.no_grad():
        a_final = torch.sigmoid(a_raw) * 0.3
        b_final = torch.sigmoid(b_raw) * 0.97
        R_seq, _ = dcc_recursion_torch(z_t, a_final, b_final, Q_bar)

    return a_final.item(), b_final.item(), R_seq, ll_history

if __name__ == "__main__":
    garch_residuals = pd.read_parquet('data/processed/garch_residuals.parquet')
    a, b, R_seq, ll_history = fit_dcc_baseline(garch_residuals, n_iter=500, lr=0.01, verbose=True)
    np.save('data/processed/dcc_baseline_results.npy', {
        'a': a, 'b': b,
        'll_final': ll_history[-1],
        'll_history': ll_history,
        'R_seq': R_seq.detach().numpy()
    })
    print(f"✅ Saved")
