# Sprint 17 — Adversarial Robustness & Explainability

**Theme:** Attacking, defending, and explaining your own models
**Duration:** 2 weeks
**Phase:** Phase 6 — Special Topics

---

## Goal

Use the models you built in Sprints 7-16 as targets: implement adversarial attacks against them, then defenses, then explainability tools. By end of sprint you can craft adversarial examples that fool your CV classifier, measurably harden it with adversarial training, and produce saliency/attribution maps that explain individual predictions.

---

## Concepts

### Threat Models

- **White-box:** attacker has full access to model weights and gradients (worst case for the defender, best case for understanding robustness).
- **Black-box:** attacker only sees outputs (queries) — attacks via transferability or query-based gradient estimation.
- **Perturbation budget:** attacks are constrained to an `ε`-ball around the input (L∞, L2, or L0 norm) — otherwise "adversarial" just means "a different image."

### Gradient-Based Attacks

- **FGSM (Fast Gradient Sign Method):** `x_adv = x + ε * sign(∇_x L(model(x), y))` — one step in the direction that increases loss.
- **PGD (Projected Gradient Descent):** iterative FGSM with projection back into the `ε`-ball after each step — much stronger than FGSM.
- **Targeted vs. untargeted:** untargeted just wants any misclassification; targeted wants a specific wrong class — requires minimizing loss toward the target class instead.
- These attacks reuse exactly the backward pass you built in Sprints 5-6 — gradients w.r.t. *input* instead of weights.

### Adversarial Training

- Train on a mix of clean and adversarially-perturbed examples (generated on-the-fly with FGSM/PGD during training).
- Robustness-accuracy tradeoff: adversarially trained models are usually slightly less accurate on clean data but far more robust to attacks within the training `ε`.
- Evaluate robustness with **accuracy under attack** at various `ε`, not just clean accuracy.

### Saliency & Attribution Methods

- **Vanilla gradient saliency:** `|∂output/∂input|` — which input pixels/tokens most affect the prediction.
- **Grad-CAM:** weight feature-map activations by the gradient of the class score w.r.t. those activations — produces a coarse heatmap localizing what the CNN "looked at."
- **Integrated Gradients:** average gradients along a path from a baseline (e.g., all-zero image) to the input — satisfies axioms that vanilla gradients don't (e.g., sensitivity).
- **Attention visualization:** for Transformers, attention weights themselves are a (imperfect) form of interpretability — visualize which tokens attend to which.

### Model-Agnostic Explainability

- **LIME:** locally approximate the model with an interpretable linear model around a single prediction by perturbing the input and observing output changes.
- **SHAP:** game-theoretic feature attributions (Shapley values) — fairly distribute the prediction "credit" across features. Connects to cooperative game theory; expensive but principled.
- These methods work on *any* model — including your classical ML models from Sprints 3-4 — making them useful for comparing tree ensembles vs. neural nets on the same dataset.

### Failure Modes & Robustness Beyond Adversarial Examples

- **Distribution shift:** model trained on one data distribution, evaluated on another (e.g., different lighting, demographics, time period).
- **Spurious correlations:** model learns shortcuts (e.g., background texture instead of object shape) — Grad-CAM often reveals this.
- **Calibration:** does the model's confidence match its actual accuracy? Reliability diagrams and temperature scaling.

---

## Tasks

- [ ] Implement `fgsm_attack(model, x, y, eps)` — single-step gradient sign attack against your Sprint 12 CV classifier
- [ ] Implement `pgd_attack(model, x, y, eps, alpha, steps)` — iterative attack with projection
- [ ] Measure clean accuracy vs. accuracy-under-attack at `ε` ∈ {0, 0.01, 0.05, 0.1} — plot the robustness curve
- [ ] Generate and visualize 5 adversarial examples — show original, perturbation (amplified), and adversarial image side by side
- [ ] Implement targeted PGD — force misclassification into a specific chosen class
- [ ] Implement adversarial training loop — mix clean + PGD-generated examples each batch
- [ ] Re-measure the robustness curve for the adversarially trained model — compare to baseline
- [ ] Implement vanilla gradient saliency maps for your CV classifier
- [ ] Implement Grad-CAM for your ResNet/CNN (Sprint 12) — overlay heatmap on input image
- [ ] Implement Integrated Gradients — verify the completeness axiom (attributions sum to `output(x) - output(baseline)`)
- [ ] Visualize attention maps from your Transformer (Sprint 10) on a few input sequences
- [ ] Implement a minimal LIME — perturb a tabular input from Sprint 3/4 models, fit a local linear surrogate, show feature weights
- [ ] Write `notebooks/adversarial_and_explainability.ipynb` summarizing all results with visualizations
- [ ] **Ship:** `security/adversarial.py` + `security/explainability.py` + notebook + robustness comparison plot

---

## Interview Topics

- **Why does FGSM use the *sign* of the gradient rather than the raw gradient?** Constrains the perturbation to a fixed L∞ budget per pixel — every pixel moves by exactly `ε` in the direction that increases loss, regardless of gradient magnitude.
- **Why is PGD considered a much stronger attack than FGSM?** Multiple small steps with projection can navigate non-linear loss landscapes that a single large step overshoots or undershoots — PGD is closer to the true worst-case perturbation within the `ε`-ball.
- **What is the robustness-accuracy tradeoff, and why does it exist?** Adversarial training effectively smooths the decision boundary / forces the model to rely on more robust (often lower-frequency) features, which can discard some predictive signal that was non-robust but useful for clean accuracy.
- **How does Grad-CAM differ from vanilla saliency, and why is it usually more interpretable for CNNs?** Grad-CAM operates on a late convolutional feature map (spatially coarse but semantically rich) rather than raw input pixels (spatially fine but noisy gradients) — produces smoother, more localized heatmaps.
- **Why do attention weights not fully explain a Transformer's decision?** Attention shows where information flows, not how it's used downstream — value vectors, residual connections, and MLP layers all transform that information further. Attention is correlational evidence, not a causal explanation.
- **What is the difference between LIME and SHAP?** LIME fits an arbitrary local linear model and depends on the perturbation sampling and kernel choice. SHAP's Shapley values are uniquely determined by a set of fairness axioms (efficiency, symmetry, dummy, additivity) from game theory — more principled but more expensive to compute exactly.

---

## Definition of Done

- [ ] FGSM and PGD measurably drop your CV classifier's accuracy at `ε=0.05` (e.g., from >90% to <30%)
- [ ] Adversarially trained model shows a flatter robustness curve than the baseline at the cost of some clean accuracy
- [ ] Grad-CAM heatmaps visually localize the relevant object region for at least 5 correctly classified images
- [ ] Integrated Gradients attributions satisfy completeness within 1e-3
- [ ] Attention visualization shows sensible patterns (e.g., attending to relevant tokens) for at least 2 example sequences
- [ ] LIME surrogate's top weighted features match intuition for a simple tabular dataset
- [ ] Notebook runs end-to-end and all generated figures are saved to `results/`
