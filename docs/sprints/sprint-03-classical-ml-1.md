# Sprint 3 — Classical ML I: Regression, Regularization & Gradient Descent

**Theme:** Supervised learning fundamentals — regression and classification with manual gradient descent
**Duration:** 1.5 weeks
**Phase:** Phase 1 — Classical ML

---

## Goal

Implement the core supervised-learning algorithms using only NumPy and the math from Sprints 1-2. By end of sprint you can derive and implement linear regression, logistic regression, and regularized variants — including gradient descent from scratch — and explain bias/variance tradeoffs from first principles. This sprint is the bridge between "math on paper" and "models that fit data," before computation graphs are introduced in Sprint 5.

---

## Concepts

### Linear Regression

- Model: `ŷ = X @ w + b`. Loss: MSE (Sprint 2).
- **Normal equation** (closed form): `w = (XᵀX)⁻¹ Xᵀy`. Exact but `O(n³)` and unstable for ill-conditioned `X`.
- **Gradient descent**: `w -= lr * dMSE/dw`, where `dMSE/dw = (2/n) Xᵀ(ŷ - y)`.
- Feature scaling (standardization) matters — gradient descent converges much faster on normalized features.

### Logistic Regression

- Model: `ŷ = sigmoid(X @ w + b)`. Loss: binary cross-entropy (Sprint 2).
- Decision boundary is linear in feature space; sigmoid maps the linear score to a probability.
- Gradient has the same clean form as softmax + cross-entropy: `dL/dw = (1/n) Xᵀ(ŷ - y)`.
- Multiclass extension: **softmax regression** (one-vs-rest vs. true multinomial).

### Regularization

- **L2 (Ridge):** add `λ * ||w||²` to the loss. Shrinks weights smoothly, never to exactly zero. Closed form: `w = (XᵀX + λI)⁻¹Xᵀy`.
- **L1 (Lasso):** add `λ * |w|`. Produces sparse weights (feature selection) — no closed form, needs subgradient or coordinate descent.
- **Elastic Net:** weighted combination of L1 and L2.
- Regularization strength `λ` is a hyperparameter — tune via validation set, not training loss.

### Bias-Variance Tradeoff

- **Bias:** error from overly simple models (underfitting).
- **Variance:** error from overly sensitive models (overfitting to training noise).
- Total error ≈ bias² + variance + irreducible noise.
- Regularization trades a bit of bias for a lot of variance reduction — usually a net win.

### Train/Validation/Test Splits & Cross-Validation

- **Train** to fit parameters, **validation** to tune hyperparameters, **test** to report final performance — never touch test until the very end.
- **k-fold cross-validation:** split data into k folds, train on k-1, validate on the held-out fold, rotate. Average the validation metric.
- Data leakage: any preprocessing (scaling, imputation) must be fit on train only, then applied to val/test.

### Evaluation Metrics

- **Regression:** MSE, RMSE, MAE, R².
- **Classification:** accuracy, precision, recall, F1, confusion matrix, ROC curve, AUC.
- Accuracy is misleading on imbalanced datasets — precision/recall/F1 matter more.

---

## Tasks

- [ ] Implement `linear_regression_normal_eq(X, y)` — closed-form solution with `np.linalg.solve` (not `inv` directly)
- [ ] Implement `linear_regression_gd(X, y, lr, epochs)` — batch gradient descent, track loss per epoch
- [ ] Verify gradient descent converges to the same `w` as the normal equation on a synthetic dataset
- [ ] Implement `ridge_regression(X, y, lam)` — closed form with L2 penalty
- [ ] Implement `lasso_regression_coordinate_descent(X, y, lam)` — coordinate descent with soft-thresholding
- [ ] Implement `logistic_regression_gd(X, y, lr, epochs)` — sigmoid + BCE + gradient descent
- [ ] Implement `softmax_regression_gd(X, y, lr, epochs)` — multinomial logistic regression for k classes
- [ ] Implement `train_val_test_split(X, y, ratios)` and `k_fold_split(X, y, k)`
- [ ] Implement metrics: `accuracy`, `precision_recall_f1`, `confusion_matrix`, `roc_auc`
- [ ] Run linear regression (plain, ridge, lasso) on a synthetic dataset with noise — plot weight magnitudes vs. `λ`
- [ ] Run logistic/softmax regression on a small classification dataset (e.g., Iris or a synthetic 2D blob set) — plot decision boundary
- [ ] Write `notebooks/bias_variance.ipynb` — fit polynomial regression at degrees 1, 4, 15; plot train/val error vs. degree to show under/overfitting
- [ ] **Ship:** `ml_classical/linear_models.py` + `ml_classical/model_selection.py` + `ml_classical/metrics.py` + tests

---

## Interview Topics

- **Why does gradient descent on linear regression converge to the same solution as the normal equation?** MSE is convex — single global minimum. GD with a small enough learning rate converges to it; normal equation jumps there directly.
- **Why does L1 produce sparsity but L2 doesn't?** Geometrically, the L1 ball has corners on the axes — the loss contour is more likely to intersect at a corner (a zero coefficient). The L2 ball is smooth, so intersections are generically at non-zero points.
- **What's the difference between precision and recall, and when do you optimize for each?** Precision = of predicted positives, how many are correct (minimize false positives — e.g., spam filter). Recall = of actual positives, how many are found (minimize false negatives — e.g., cancer screening).
- **Why is the softmax regression gradient identical in form to logistic regression's?** Both derive from cross-entropy over an exponential-family distribution — the gradient `(ŷ - y)` form is a property of the exponential family's natural parameters (canonical link function).
- **Explain k-fold cross-validation and why it gives a less noisy estimate than a single train/val split.** Every example is used for validation exactly once across folds; averaging reduces variance from any single unlucky split.

---

## Definition of Done

- [ ] Gradient descent linear regression matches normal-equation weights within 1e-3 on a synthetic dataset
- [ ] Ridge regression weight norms shrink monotonically as `λ` increases — shown in a plot
- [ ] Lasso produces exactly-zero coefficients for irrelevant features on a synthetic dataset with known sparse ground truth
- [ ] Logistic/softmax regression reaches >90% accuracy on Iris (or equivalent simple dataset)
- [ ] `k_fold_split` produces non-overlapping folds covering the full dataset
- [ ] Bias-variance notebook clearly shows underfitting at degree 1 and overfitting at degree 15
- [ ] All tests pass with `pytest ml_classical/`
