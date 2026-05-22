import torch
import torch.nn as nn
import numpy as np
import pandas as pd

def compute_Q_bar(z_t):
    T = z_t.shape[0]
    return (z_t.T @ z_t) / T

def dcc_topo_recursion(z_t, a_seq, b_seq, Q_bar):
    T, N = z_t.shape
    Q_t = Q_bar.clone()
    Q_seq = torch.zeros(T, N, N, dtype=z_t.dtype)
    Q_seq[0] = Q_t
    for t in range(1, T):
        a_t = a_seq[t]
        b_t = b_seq[t]
        z_outer = torch.outer(z_t[t-1], z_t[t-1])
        Q_t = (1 - a_t - b_t) * Q_bar + a_t * z_outer + b_t * Q_t
        Q_seq[t] = Q_t
    diag_Q = torch.sqrt(torch.diagonal(Q_seq, dim1=1, dim2=2))
    R_seq = Q_seq / torch.einsum('ti,tj->tij', diag_Q, diag_Q)
    sign, log_det = torch.linalg.slogdet(R_seq)
    R_inv = torch.linalg.inv(R_seq)
    mahal = torch.sum(z_t * (R_inv @ z_t.unsqueeze(-1)).squeeze(-1), dim=1)
    ll = -0.5 * (log_det.sum() + mahal.sum())
    return R_seq, ll

class TopoDCC(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.w_a = nn.Parameter(torch.randn(n_features) * 0.01)
        self.w_b = nn.Parameter(torch.randn(n_features) * 0.01)
        self.bias_a = nn.Parameter(torch.tensor(-3.0))
        self.bias_b = nn.Parameter(torch.tensor(0.0))
        self.bias_slack = nn.Parameter(torch.tensor(0.0))

    def forward(self, X_t):
        logit_a = X_t @ self.w_a + self.bias_a
        logit_b = X_t @ self.w_b + self.bias_b
        logit_slack = torch.full_like(logit_a, 0.0) + self.bias_slack
        logits = torch.stack([logit_a, logit_b, logit_slack], dim=1)
        probs = torch.softmax(logits, dim=1)
        return probs[:, 0], probs[:, 1]

def fit_dcc_topo(garch_residuals_df, tda_features_df, n_iter=500, lr=0.01, verbose=True):
    z_t = torch.tensor(garch_residuals_df.values, dtype=torch.float32)
    X_raw = tda_features_df.values
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0) + 1e-8
    X_t = torch.tensor((X_raw - X_mean) / X_std, dtype=torch.float32)
    T, N = z_t.shape
    n_features = X_t.shape[1]
    Q_bar = compute_Q_bar(z_t)
    model = TopoDCC(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ll_history = []
    for i in range(n_iter):
        optimizer.zero_grad()
        a_seq, b_seq = model(X_t)
        R_seq, ll = dcc_topo_recursion(z_t, a_seq, b_seq, Q_bar)
        loss = -ll
        loss.backward()
        optimizer.step()
        ll_history.append(ll.item())
        if verbose and (i % 50 == 0 or i == n_iter - 1):
            print(f"  Iter {i:4d} | a_t mean={a_seq.mean().item():.4f} | b_t mean={b_seq.mean().item():.4f} | a+b mean={(a_seq+b_seq).mean().item():.4f} | ll={ll.item():.2f}")
    with torch.no_grad():
        a_seq, b_seq = model(X_t)
        R_seq, _ = dcc_topo_recursion(z_t, a_seq, b_seq, Q_bar)
    return model, a_seq, b_seq, R_seq, ll_history

if __name__ == "__main__":
    garch_residuals = pd.read_parquet('data/processed/garch_residuals.parquet')
    tda_features = pd.read_parquet('data/processed/tda_features.parquet')
    assert (garch_residuals.index == tda_features.index).all()
    model, a_seq, b_seq, R_seq, ll_history = fit_dcc_topo(garch_residuals, tda_features, n_iter=500, lr=0.01, verbose=True)
    np.save('data/processed/dcc_topo_results.npy', {
        'a_seq': a_seq.detach().numpy(),
        'b_seq': b_seq.detach().numpy(),
        'R_seq': R_seq.detach().numpy(),
        'll_history': ll_history,
        'll_final': ll_history[-1],
        'w_a': model.w_a.detach().numpy(),
        'w_b': model.w_b.detach().numpy(),
    })
    print("Saved")
