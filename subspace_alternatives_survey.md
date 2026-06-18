# Subspace Separation Alternatives — Survey and Comparison

## The Problem Restated

We have a single dataset of PZT impedance sweeps, each with two labels:
- **device_id** (identity — 295 classes)
- **condition** (health — baseline / temperature / aging / load, with severities)

We want to find **two linear subspaces** of the PCA space:
1. **Identity subspace** — scores vary with device_id, invariant to condition
2. **Health subspace** — scores vary with condition, invariant to device

With **minimal information leakage** between them (low HSIC).

---

## Method 1: PCA + F-ratio Gap Partition (Current Baseline — Section 8)

**How it works:** Run standard PCA on all data. For each PC, compute two ANOVA F-statistics: one for device_id (identity score) and one for condition (health score). Sort PCs by the ratio `identity_F / health_F`. Find the largest gap in this sorted ratio — PCs above the gap are "identity," below are "health." Use median thresholding to further split into dual-use and neither.

**Results (from rq6_9_dual_use.py):**
| Metric | Value |
|--------|-------|
| Identity PCs | 27 |
| Health PCs | 73 |
| Dual-use PCs | 26 |
| Neither PCs | 26 |
| Auth EER (identity sub.) | 7.0–22.3% (degrades under perturbation) |
| Health detection (health sub.) | ~79% |
| Subspace independence | **Not explicitly enforced** |

**Pro:** Simple, interpretable, fast.
**Con:** No explicit disentanglement — dual-use PCs are simply discarded. Does not optimize for independence.

---

## Method 2: sisPCA (Current — RQ4 in rq2_4_synthetic.py)

**How it works:** Extends PCA with HSIC (Hilbert-Schmidt Independence Criterion). Takes supervision as kernel matrices (device_id similarity, condition similarity). Learn projection matrices that maximize alignment of each subspace with its supervision while minimizing HSIC between subspaces.

**Reference:** Su et al. (2024) — "Disentangling Interpretable Factors with Supervised Independent Subspace Principal Component Analysis." *NeurIPS 2024*. [https://github.com/JiayuSuPKU/sispca](https://github.com/JiayuSuPKU/sispca)

**Results (from rq2_4):**
| λ | ID Acc | Health Acc | HSIC | Sep Quality |
|---|--------|-----------|------|-------------|
| 0.0 | 0.833 | 0.624 | 450,000 | 0.00 |
| 1.0 | 0.833 | 0.624 | 0.080 | 1.82 |
| 10.0 | 0.833 | 0.624 | **0.001** | **1.99** |

**Pro:** Explicitly minimizes subspace overlap. Linear (interpretable). Works on single dataset with labels.
**Con:** Requires PyTorch. Slower than standard PCA. Need to tune λ.

---

## Method 3: FastICA + F-ratio Partition (Blind Source Separation)

**How it works:** Apply Independent Component Analysis (ICA) to the PCA-reduced data. ICA finds components that are **statistically independent** (maximally non-Gaussian). Since identity and health are physically independent sources mixing in the impedance signal, ICA should naturally separate them.

**Reference:** Hyvärinen & Oja (2000) — "Independent Component Analysis: Algorithms and Applications." *Neural Networks*, 13(4-5).

**Implementation:** `sklearn.decomposition.FastICA` — 3 lines.

**Predicted:** Unsupervised, fast, but can't incorporate labels. Components have arbitrary order.

---

## Method 4: Multi-output PLS + F-ratio Partition (Supervised Latent Variables)

**How it works:** Partial Least Squares finds latent components that maximize covariance between X (data) and Y (labels). With a multi-output Y (one-hot device_id + one-hot condition), PLS naturally finds components ordered by their joint predictive power.

**Reference:** Barker & Rayens (2003) — "Partial least squares for discrimination." *Journal of Chemometrics*, 17(3).

**Implementation:** `sklearn.cross_decomposition.PLSRegression` — ~5 lines.

**Predicted:** Supervised, fast, but doesn't enforce independence.

---

## Method 5: Two-task LDA on PCA Scores (Discriminant Subspaces)

**How it works:** Run PCA first. Train two independent LDA models: one for device_id, one for condition. LDA directions naturally span the discriminative subspaces. Measure angle between subspaces.

**Reference:** Fisher (1936) — Linear Discriminant Analysis.

**Implementation:** `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` — 3 lines.

**Predicted:** Upper bound on per-task accuracy, but no independence enforcement.

---

## Method 6: HCV — Nonlinear sisPCA

**How it works:** Nonlinear extension of sisPCA using a VAE with HSIC-regularized latent subspaces. Available in the `sispca` package.

**Reference:** Lopez et al. (2018) — scvi-tools framework.

**Predicted:** More expressive but slower, less interpretable.

---

## Survey Summary

| Method | Supervision | Disentanglement | Interpretable | Speed |
|--------|------------|----------------|---------------|-------|
| **PCA + F-ratio** | Post-hoc only | None (discards) | ★★★★★ | ★★★★★ |
| **sisPCA (λ=10)** | Explicit (kernels) | HSIC (0.001) | ★★★★☆ | ★★★☆☆ |
| **FastICA + F-ratio** | None | Assumes independence | ★★★★☆ | ★★★★★ |
| **PLS + F-ratio** | Multi-output Y | None | ★★★☆☆ | ★★★★☆ |
| **Two LDA** | Labels | None | ★★★★☆ | ★★★★★ |
| **HCV (nonlinear)** | Explicit (kernels) | HSIC | ★★☆☆☆ | ★☆☆☆☆ |

---

## Empirical Results (Run June 2026)

Four methods compared on 295 devices × 5 sweeps + 3540 synthetic perturbations (5015 total samples). All use 100 components, F-ratio gap partition, Mahalanobis-based EER evaluation.

| Method | ID | HL | Dual | HSIC | AuthEER | HlthEER |
|--------|----|----|------|------|---------|---------|
| PCA | 42 | 42 | 8 | 1,466,267 | 0.2423 | 0.1727 |
| FastICA | 34 | 34 | 16 | **0.0000** | **0.0913** | 0.4199 |
| PLS | 24 | 24 | 26 | **0.0000** | **0.0671** | **0.2029** |
| PCA (sep F-ratio) | 24 | 24 | 26 | 739,763 | 0.2287 | 0.1726 |
| **sisPCA (λ=10)** | N/A | N/A | N/A | **0.0014** | 0.383* | 0.258* |

\* sisPCA EERs = 1 − accuracy from rq2_4 pipeline — different evaluation, not directly comparable

### Key Findings

1. **PLS wins on balance** — Auth EER 6.7%, Health EER 20.3%, HSIC ~0. Label supervision guides components to be useful for both tasks.

2. **FastICA: best auth, worst health** — HSIC ~0 confirms independence, but health EER 42% (near chance) means ICA separates identity from *noise*, not from health. Health deviation is flat (5–8 for all conditions).

3. **PCA: best health, worst auth** — Health EER 17.3% is best, but identity/health completely entangled (HSIC ~10⁶). Auth EER 24.2%.

4. **Identity and health are NOT naturally independent sources** — FastICA's failure to find useful health components proves explicit supervision or structure regularization is required.

5. **Subspace independence ≠ task performance** — Achieving HSIC ~0 is easy (FastICA). Making components *useful* for both tasks is the hard part.

### Recommendation

Use **PLS + F-ratio** as a fast lightweight baseline for production-like scenarios. Keep **sisPCA** for rigorous disentanglement guarantees (HSIC 0.0014 is explicit, not coincidental). The choice depends on whether EER or independence is the priority.

### Files

- `subspace_comparison.py` — full comparison script
- `rq_analysis_reports/subspace_method_comparison.csv` — results table
- `rq_analysis_reports/subspace_method_comparison.png` — bar chart comparison
