# Topological Dynamic Conditional Correlations: A Persistence Landscape Approach to Time-Varying Dependence in Financial Markets

**MSc Asset Pricing Thesis — King's College London**

---

## Overview

This repository implements the **TopoDCC** model — a novel extension of the Dynamic Conditional Correlation (DCC) framework that uses topological features derived from persistence landscapes to make the DCC shock-sensitivity and persistence parameters time-varying.

The core idea, motivated by Gidea & Katz (2018), is that the geometric structure of the multivariate return point cloud — captured via persistent homology — carries information about market regime changes that is not explained by volatility, autocorrelation, or realized correlation alone. This topological signal is used to drive the DCC parameters `a_t` and `b_t` at each point in time, rather than fixing them as constants.

---

## Methodology

### Pipeline

```
Raw Prices
    └── Log Returns  (loader.py)
        └── GARCH(1,1) Residuals  (garch.py)
            └── TDA Pipeline  (tda_pipeline.py)
                │   - Delay embedding (embed_dim=5)
                │   - PCA reduction (pca_dim=10)
                │   - Vietoris-Rips persistence via Ripser
                │   - Persistence landscapes on fixed global grid
                └── L² Norm Features  (lp_norm_features.py)
                    │   - 3 H1 landscape level norms (lh1_k0, k1, k2)
                    │   - 6 scalar features (betti_1, entropy, wasserstein, ...)
                    │   - 9 features total
                    └── TopoDCC  (dcc_topo.py)
                        └── Permutation Test  (permutation_test_lpnorm.py)
```

### Key Design Choices

- **Assets:** SPY, EEM, GLD, TLT, DBC — chosen for cross-asset geometric diversity (equity, EM, gold, rates, commodities)
- **Homology degree:** H1 only (loop structure), following Gidea & Katz (2018)
- **Norm:** L² norm via trapezoidal integration over the global filtration grid — the exact quantity in Gidea & Katz
- **DCC input:** GARCH(1,1) standardised residuals per asset
- **Window:** 250 trading days (~1 year), matching Gidea & Katz

### TopoDCC Model

The DCC recursion is extended so that `a_t` and `b_t` are functions of the topology features at each time step:

```
a_t = sigmoid(X_t @ w_a + bias_a)
b_t = sigmoid(X_t @ w_b + bias_b)
```

where `X_t` is the normalised 9-dimensional topology feature vector at time `t`. The model is initialised to reproduce baseline DCC values (`sigmoid(-3.5) ≈ 0.03`, `sigmoid(2.9) ≈ 0.95`) and trained end-to-end by maximising the DCC log-likelihood.

---

## Results

| Model | Log-Likelihood |
|---|---|
| Baseline DCC (constant a, b) | -8,247.96 |
| TopoDCC (topology-driven a_t, b_t) | -7,827.58 |
| **Improvement** | **+420.38 nats** |

### Permutation Test
To verify the improvement reflects genuine topological signal rather than parameter flexibility:

- **Real features:** ll = -7,827.58
- **Shuffled features mean:** -7,838.79 (std = 7.41)
- **Real beats permuted mean by:** +11.21 nats
- **Permuted runs beating real:** 1/10

### Stationarity (ADF Test)
All three landscape L² norm series are stationary (p < 0.0001), confirming their validity as regressors.

### Crisis Period Behaviour
The topology-driven `a_t` (shock sensitivity) rises during endogenous crises:

| Period | a_t mean | vs. full sample (0.0239) |
|---|---|---|
| GFC (2008-09) | 0.0266 | +11% |
| EU Debt (2011-12) | 0.0378 | +58% |
| COVID (2020) | 0.0209 | -13% |
| Rate Hikes (2022) | 0.0229 | -4% |

The asymmetry between endogenous crises (GFC, EU Debt — rising `a_t`) and exogenous shocks (COVID — falling `a_t`) is economically meaningful and consistent with the loop-collapse mechanism: during sudden exogenous shocks, cross-asset correlations spike to 1 and the point cloud geometry degenerates, giving the model less topological structure to react to.

### Multitrack Diagnostics
The L² norm series is not a proxy for simpler quantities:

| Candidate | lh1_k0 r | lh1_k1 r | lh1_k2 r |
|---|---|---|---|
| Realized volatility | -0.02 | -0.23 | -0.32 |
| Rolling autocorrelation | +0.06 | +0.11 | +0.09 |

No candidate exceeds the |r| > 0.5 threshold — the topology captures genuinely distinct information.

---

## Repository Structure

```
MSCTHESIS/
├── config/
│   └── config.yaml              # all parameters in one place
├── data/
│   ├── raw/                     # gitignored
│   └── processed/               # gitignored (regenerable)
├── src/
│   ├── data/
│   │   └── loader.py            # downloads prices, computes log returns
│   ├── volatility/
│   │   └── garch.py             # GARCH(1,1) per asset -> standardised residuals
│   ├── topology/
│   │   ├── tda_pipeline.py      # persistence landscapes (two-phase, parallel)
│   │   └── lp_norm_features.py  # L² norm reduction -> 9 features
│   ├── models/
│   │   ├── dcc_baseline.py      # constant-parameter DCC
│   │   └── dcc_topo.py          # TopoDCC model
│   └── evaluation/
│       └── stats_tests.py       # stationarity, crisis diagnostics (growing)
├── scripts/
│   ├── diagnostic_lpnorm_vs_crashes.py   # crash period plot
│   └── multitrack.py                     # candidate explanation checks
└── README.md
```

---

## Reproducing Results

```bash
# 1. Install dependencies
pip install numpy pandas torch ripser persim arch statsmodels scikit-learn matplotlib joblib yfinance

# 2. Download data
python src/data/loader.py

# 3. GARCH filtering
python src/volatility/garch.py

# 4. TDA pipeline (takes ~15 minutes)
python src/topology/tda_pipeline.py

# 5. L² norm reduction
python src/topology/lp_norm_features.py

# 6. Fit TopoDCC
python src/models/dcc_topo.py

# 7. Permutation test
python src/models/permutation_test_lpnorm.py

# 8. Diagnostics
python src/evaluation/stats_tests.py
python scripts/diagnostic_lpnorm_vs_crashes.py
python scripts/multitrack.py
```

---

## Dependencies

- Python 3.14
- PyTorch
- Ripser / Persim (TDA)
- arch (GARCH)
- statsmodels
- scikit-learn
- pandas, numpy, matplotlib, joblib, yfinance

---

## References

- Engle, R. (2002). Dynamic Conditional Correlation. *Journal of Business & Economic Statistics.*
- Gidea, M. & Katz, Y. (2018). Topological Data Analysis of Financial Time Series: Landscapes of Crashes. *Physica A.*
- Bubenik, P. (2015). Statistical Topological Data Analysis using Persistence Landscapes. *JMLR.*

---

*King's College London — MSc Asset Pricing — 2026*
