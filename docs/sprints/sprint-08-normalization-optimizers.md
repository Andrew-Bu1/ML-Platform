# Sprint 8 — Normalization + Optimizers + First Real Training

**Theme:** Complete the layer library and train a full model
**Duration:** 2 weeks
**Phase:** Phase 3 — Layer Library

---

## Goal

Add normalization layers and optimizers, then train a complete MLP on MNIST using only your library. By end of sprint you ship a working `mnist_mlp.py` that reaches >97% accuracy and plots loss + accuracy curves.

---

## Concepts

### Batch Normalization

- Normalize each feature across the batch: `x̂ = (x - μ_B) / √(σ²_B + ε)`.
- Scale and shift: `y = γ * x̂ + β` where γ and β are learnable parameters.
- At training time: use batch statistics. At eval time: use running mean/variance accumulated during training.
- **Why it works:** reduces internal covariate shift, smooths loss landscape, allows higher learning rates.
- Backward: complex — involves Jacobian through the normalization. Implement carefully.

### Layer Normalization

- Normalize across features (not batch): `x̂_i = (x_i - μ_L) / √(σ²_L + ε)`.
- Mean and variance computed over the feature dimension, not the batch dimension.
- **No running statistics** — same computation at train and eval time.
- Used in Transformers because sequence lengths vary (batch norm is awkward with variable-length sequences).

### SGD Optimizer

- `param.data -= lr * param.grad`
- With momentum: `v = momentum * v + grad`, then `param.data -= lr * v`.
- Momentum accumulates gradients in consistent directions, dampens oscillations.

### Adam Optimizer

- Tracks first moment (mean of gradients): `m = β1 * m + (1-β1) * g`
- Tracks second moment (mean of squared gradients): `v = β2 * v + (1-β2) * g²`
- Bias correction: `m̂ = m / (1-β1ᵗ)`, `v̂ = v / (1-β2ᵗ)`
- Update: `param -= lr * m̂ / (√v̂ + ε)`
- Default: β1=0.9, β2=0.999, ε=1e-8, lr=1e-3.

### Training Loop Structure

```
for epoch in range(epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()        # reset gradients
        pred = model(batch_x)        # forward pass
        loss = loss_fn(pred, batch_y) # compute loss
        loss.backward()              # backward pass
        optimizer.step()             # update params
```

---

## Tasks

- [ ] Implement `BatchNorm1d(num_features)` — train/eval modes, running stats, γ/β parameters
- [ ] Grad check BatchNorm with batch size 8, feature size 16
- [ ] Implement `LayerNorm(normalized_shape)` — normalize over last dimension(s)
- [ ] Grad check LayerNorm for shapes (batch, seq, features)
- [ ] Implement `SGD(params, lr, momentum=0)` optimizer
- [ ] Implement `Adam(params, lr, betas, eps)` optimizer with bias correction
- [ ] Implement `zero_grad()` on optimizer — sets all param grads to zero
- [ ] Write `DataLoader` utility — batches numpy arrays, optional shuffle
- [ ] Write training loop function with epoch loop, batch loop, loss + accuracy tracking
- [ ] Train `Sequential(Linear(784,256), BatchNorm1d(256), ReLU(), Dropout(0.2), Linear(256,10))` on MNIST
- [ ] Reach >97% test accuracy — plot loss curve and accuracy curve with matplotlib
- [ ] Compare: train same model with SGD vs Adam — plot both loss curves on same figure
- [ ] **Ship:** `layers/batchnorm.py` + `layers/layernorm.py` + `layers/optim.py` + `engine/mnist_mlp.py` + `results/mnist_curves.png`

---

## Interview Topics

- **BatchNorm vs LayerNorm — when to use which?** BatchNorm over batch dimension — works well for CNNs with large fixed-size batches. LayerNorm over feature dimension — works for Transformers and variable-length sequences, no batch dependence.
- **Explain Adam's bias correction.** m and v start at zero. Early in training, they are biased toward zero. Dividing by (1-β^t) corrects for this cold-start problem.
- **Why call zero_grad() before each backward?** PyTorch (and your engine) accumulate gradients by default. Not zeroing means gradients from previous batch add to current — wrong update direction.
- **What causes training loss to spike?** Learning rate too high, bad batch (outliers), gradient explosion, or a bug in gradient accumulation.

---

## Definition of Done

- [ ] BatchNorm and LayerNorm pass grad check
- [ ] Adam optimizer reproduces known behavior on a convex quadratic (verify convergence)
- [ ] MLP trains to >97% test accuracy on full MNIST in < 20 epochs
- [ ] Loss and accuracy curves saved as PNG in `results/`
- [ ] SGD vs Adam comparison plot shows Adam converges faster
- [ ] All tests pass with `pytest layers/`
