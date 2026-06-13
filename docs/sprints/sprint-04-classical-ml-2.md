# Sprint 4 — Classical ML II: Trees, Ensembles & Unsupervised Learning

**Theme:** Tree-based models, ensembling, and unsupervised algorithms
**Duration:** 1.5 weeks
**Phase:** Phase 1 — Classical ML

---

## Goal

Implement decision trees, ensemble methods (bagging, random forest, gradient boosting), and the core unsupervised algorithms (k-means, PCA) from scratch. By end of sprint you can explain how ensembles reduce variance vs. bias, derive PCA from eigendecomposition, and have a small library of classical ML models that don't depend on autograd.

---

## Concepts

### Decision Trees

- Recursive partitioning: at each node, pick the feature/threshold split that best separates the data.
- **Splitting criteria:** Gini impurity (`1 - Σp_i²`) or entropy (Sprint 2) for classification; variance reduction (MSE) for regression.
- **Information gain:** reduction in impurity/entropy after a split — pick the split that maximizes it.
- Stopping criteria: max depth, min samples per leaf, min impurity decrease — prevent overfitting (a fully grown tree memorizes training data).

### Bagging & Random Forest

- **Bagging (Bootstrap Aggregating):** train many models on bootstrap-resampled (sample with replacement) subsets of the data, average predictions. Reduces variance.
- **Random Forest:** bagging + at each split, only consider a random subset of features. Decorrelates trees further.
- Out-of-bag (OOB) samples (the ~37% left out of each bootstrap) give a free validation estimate.
- Feature importance: average decrease in impurity (or permutation importance) across all trees.

### Gradient Boosting

- Sequential ensemble: each new tree fits the **residual** (negative gradient of the loss) of the current ensemble's predictions.
- `F_m(x) = F_{m-1}(x) + lr * h_m(x)` where `h_m` is fit to `-∂L/∂F_{m-1}`.
- For MSE, the residual is literally `y - ŷ`. For other losses, it's the negative gradient — this connects directly back to Sprint 2's loss derivatives.
- Learning rate (shrinkage) trades training speed for generalization — small `lr` + many trees usually generalizes better.
- This is conceptually the bridge to how DL training works: iterative gradient-based improvement, just on functions (trees) instead of weight tensors.

### K-Means Clustering

- Iterative algorithm: (1) assign each point to nearest centroid, (2) recompute centroids as the mean of assigned points, repeat until convergence.
- Sensitive to initialization — **k-means++** picks initial centroids spread out probabilistically to avoid bad local minima.
- Choosing `k`: elbow method (plot inertia vs. k) or silhouette score.
- Converges to a local minimum of within-cluster sum of squares — not guaranteed global.

### PCA (Principal Component Analysis)

- Goal: find an orthogonal basis that captures maximum variance in fewer dimensions.
- Steps: center the data (subtract mean), compute covariance matrix `C = (1/n) XᵀX`, eigendecompose `C = VΛVᵀ`, project `X_reduced = X @ V[:, :k]`.
- Eigenvalues = variance explained by each principal component. Sort descending, keep top-k.
- Equivalent to SVD of the centered data matrix — SVD is more numerically stable for high-dimensional data.
- Used for dimensionality reduction, visualization, denoising, and as a preprocessing step before clustering.

### Naive Bayes (bonus — quick implementation)

- Assumes feature independence given the class: `P(y|x) ∝ P(y) * Π P(x_i|y)`.
- Fast, strong baseline for text classification — directly connects to the NLP track (Sprint 11).

---

## Tasks

- [ ] Implement `gini(y)` and `entropy(y)` impurity functions, and `information_gain(parent, left, right)`
- [ ] Implement `DecisionTreeClassifier` — recursive split-finding (brute force over thresholds), max_depth, min_samples_leaf
- [ ] Implement `DecisionTreeRegressor` — same structure, variance-reduction splitting
- [ ] Verify decision tree overfits with no depth limit (100% train accuracy) and generalizes better with max_depth=3-5
- [ ] Implement `RandomForestClassifier` — bagging + feature subsampling + majority vote
- [ ] Compute and plot feature importances from the random forest on a synthetic dataset with known important/irrelevant features
- [ ] Implement `GradientBoostingRegressor` — sequential tree fitting on residuals, with learning rate
- [ ] Plot training loss vs. number of boosting rounds for 2-3 different learning rates
- [ ] Implement `KMeans(k)` — Lloyd's algorithm with k-means++ initialization
- [ ] Plot inertia vs. k (elbow method) on a synthetic clustered dataset
- [ ] Implement `PCA(n_components)` — via eigendecomposition of the covariance matrix; verify against `np.linalg.svd`
- [ ] Apply PCA to a small image dataset (e.g., digits) — visualize first 2 components and reconstruct images from top-k components
- [ ] Implement `GaussianNaiveBayes` — fit per-class mean/variance, classify via log-likelihood
- [ ] **Ship:** `ml_classical/trees.py` + `ml_classical/ensembles.py` + `ml_classical/clustering.py` + `ml_classical/pca.py` + `ml_classical/naive_bayes.py` + tests

---

## Interview Topics

- **Why does bagging reduce variance but not bias?** Averaging independent (or weakly correlated) high-variance estimators reduces the variance of the average without changing its expected value — bias of the average equals bias of individuals.
- **How does random forest decorrelate trees beyond bagging alone?** Restricting the feature subset at each split prevents all trees from picking the same dominant feature at the top, so errors are less correlated across trees.
- **Derive why gradient boosting fits residuals for MSE loss.** `∂(1/2)(y-ŷ)²/∂ŷ = -(y-ŷ)`. The negative gradient is exactly the residual — fitting a new model to the residual is gradient descent in function space.
- **Why does k-means converge but not necessarily to the global optimum?** Each step (assignment, update) monotonically decreases inertia, so the algorithm converges — but to whichever local minimum the initialization basin leads to. Hence k-means++ and multiple restarts.
- **What's the relationship between PCA and SVD?** PCA on centered data `X` is equivalent to SVD `X = UΣVᵀ` — the columns of `V` are the principal components, and `Σ²/n` gives the eigenvalues of the covariance matrix.
- **Why does Naive Bayes work well despite the independence assumption being almost always false?** It only needs to rank the correct class highest, not estimate `P(y|x)` accurately — the decision boundary can still be correct even if the probability estimates are miscalibrated.

---

## Definition of Done

- [ ] Decision tree reaches near-100% train accuracy unbounded, and improved val accuracy with depth limits — both shown
- [ ] Random forest outperforms a single decision tree on a noisy synthetic dataset
- [ ] Feature importance plot correctly ranks known-important features above known-irrelevant ones
- [ ] Gradient boosting training loss decreases monotonically and lower `lr` shows smoother, slower convergence
- [ ] K-means elbow plot shows a visible "elbow" at the true number of clusters in a synthetic dataset
- [ ] PCA eigendecomposition matches `np.linalg.svd` components within sign flip
- [ ] All tests pass with `pytest ml_classical/`
