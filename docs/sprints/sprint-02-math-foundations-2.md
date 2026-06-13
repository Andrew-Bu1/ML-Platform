# Sprint 2 — Math Foundations II

**Theme:** Probability, information theory, and optimization math
**Duration:** 2 weeks
**Phase:** Phase 0 — Math Foundations

---

## Goal

Implement every loss function and probability-based operation you will encounter across NLP, CV, and generative models. By end of sprint you can derive softmax, cross-entropy, and KL divergence from first principles and implement them with correct gradients.

---

## Concepts

### Probability Distributions

- A **probability distribution** assigns a non-negative value to each outcome, summing to 1.
- **Softmax** converts a vector of raw scores (logits) into a probability distribution: `p_i = exp(z_i) / sum(exp(z_j))`.
- Numerical stability: subtract `max(z)` before exponentiating. `exp(z - max(z))` produces the same probabilities but avoids overflow.

### Cross-Entropy Loss

- Measures the difference between a predicted distribution `q` and the true distribution `p`.
- For classification with one-hot `p`: `L = -log(q[true_class])`.
- Gradient with respect to logits (before softmax): `dL/dz_i = q_i - p_i`. This is why softmax + cross-entropy has such a clean backward pass.
- Binary cross-entropy (BCE): `L = -(y·log(p) + (1-y)·log(1-p))`. Used for binary classification and multi-label problems.

### KL Divergence

- `KL(P || Q) = sum(P * log(P/Q))` — measures how much distribution P diverges from Q.
- Not symmetric: `KL(P||Q) ≠ KL(Q||P)`.
- Always ≥ 0. Equals 0 iff P = Q.
- Appears in VAE loss (regularize latent space toward standard normal), language model evaluation (perplexity), and distillation.

### Entropy

- `H(P) = -sum(P * log(P))` — measures uncertainty in a distribution.
- Maximum entropy = uniform distribution. Zero entropy = certain outcome.
- Cross-entropy = entropy + KL divergence: `H(P, Q) = H(P) + KL(P || Q)`.

### Mean Squared Error

- `MSE = (1/n) * sum((y - ŷ)²)` — average squared difference between predictions and targets.
- Gradient: `dMSE/dŷ = (2/n) * (ŷ - y)`.
- Used for regression tasks, autoencoders (image reconstruction), and diffusion model denoising objectives.

### Gradient Descent and Learning Rate

- Update rule: `θ = θ - lr * ∇L(θ)`.
- **Learning rate** controls step size. Too high → diverge. Too low → slow convergence.
- **Gradient clipping** — cap gradient norm to prevent exploding gradients in RNNs and deep networks.

---

## Tasks

- [ ] Implement `softmax(z)` — numerically stable version with max subtraction. Verify outputs sum to 1.
- [ ] Implement `softmax_backward(grad_out, softmax_out)` — Jacobian-vector product
- [ ] Implement `cross_entropy(logits, target_class)` — log-softmax formulation for numerical stability
- [ ] Verify cross-entropy gradient `q - p` with finite difference check
- [ ] Implement `binary_cross_entropy(pred, target)` — clamp pred to avoid log(0)
- [ ] Implement `mse_loss(pred, target)` and `mse_backward(pred, target)`
- [ ] Implement `kl_divergence(P, Q)` — verify it is ≥ 0 for 10 random distribution pairs
- [ ] Implement `entropy(P)` — verify H(uniform) = log(n), H(certain) = 0
- [ ] Write notebook `notebooks/information_theory.ipynb` — plot two distributions, their KL, and cross-entropy visually
- [ ] **Ship:** `engine/losses.py` + `engine/test_losses.py` — all pass finite difference check

---

## Interview Topics

- **Why subtract max before softmax?** exp(1000) overflows float32. Subtracting max shifts all values to ≤ 0, bounding exp outputs, without changing the output distribution.
- **Why is the softmax + cross-entropy gradient q - p?** Derive it: chain rule through log(softmax). The simplification is why they are always used together.
- **What is perplexity?** `exp(cross_entropy_per_token)`. Measures how surprised a language model is on average. Lower is better.
- **When would you use MSE vs cross-entropy?** MSE for continuous outputs (regression, image reconstruction). Cross-entropy for categorical outputs (classification, language modeling).
- **What does KL divergence measure physically?** Extra bits needed to encode samples from P using a code optimized for Q.

---

## Definition of Done

- [ ] `softmax` outputs sum to 1.0 (within 1e-6) for all test inputs including large logits
- [ ] `cross_entropy` and `binary_cross_entropy` pass finite difference gradient check
- [ ] `kl_divergence(P, Q) >= 0` verified for 20 random distribution pairs
- [ ] `entropy(uniform(n))` ≈ `log(n)` and `entropy(one_hot)` ≈ 0
- [ ] Notebook renders two distribution plots with labeled KL and entropy values
- [ ] All tests pass with `pytest engine/test_losses.py`
