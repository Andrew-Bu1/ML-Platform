# Sprint 7 — Core Layer Primitives

**Theme:** Neural network building blocks using your tensor engine
**Duration:** 2 weeks
**Phase:** Phase 3 — Layer Library

---

## Goal

Implement every foundational layer type that every architecture in NLP, CV, and beyond is built from. By end of sprint you have a `Module` base class, `Linear`, all activations, and `Dropout` — all using your Tensor engine and all passing grad check.

---

## Concepts

### Module Base Class

- All layers inherit from `Module`.
- `Module` tracks parameters via `parameters()` — returns all `Tensor` objects with `requires_grad=True`.
- `__call__` invokes `forward()`. Subclasses implement `forward()`.
- `train()` / `eval()` toggle `self.training` — affects Dropout and BatchNorm behavior.

### Linear Layer

- `y = x @ Wᵀ + b` where W is (out_features, in_features) and b is (out_features,).
- Weight initialization: **Kaiming uniform** — `W ~ Uniform(-√(1/in), √(1/in))` scaled by activation type. Prevents vanishing/exploding activations in deep networks.
- Bias initialized to zeros.

### Activation Functions

| Activation | Formula | Backward |
|---|---|---|
| ReLU | `max(0, x)` | `1 if x > 0 else 0` |
| Sigmoid | `1 / (1 + exp(-x))` | `σ(x) * (1 - σ(x))` |
| Tanh | `(exp(x) - exp(-x)) / (exp(x) + exp(-x))` | `1 - tanh(x)²` |
| GELU | `x * Φ(x)` approx `x * 0.5 * (1 + tanh(√(2/π)(x + 0.044715x³)))` | via autograd |
| Softmax | see Sprint 2 | `p_i(δ_ij - p_j)` |

### Dropout

- During training: randomly zero out each element with probability `p`. Scale surviving values by `1/(1-p)` (inverted dropout).
- During eval: pass through unchanged. No scaling needed.
- Backward: pass gradient through only the non-zeroed positions (same mask as forward).

### Sequential Container

- Stores an ordered list of modules.
- `forward(x)`: passes `x` through each module in order.
- `parameters()`: yields parameters from all child modules.

---

## Tasks

- [ ] Implement `Module` base class with `parameters()`, `train()`, `eval()`, `__call__`
- [ ] Implement `Linear(in_features, out_features)` with Kaiming init and bias
- [ ] Grad check `Linear` forward and backward for shapes (8,16) → (8,32)
- [ ] Implement `ReLU`, `Sigmoid`, `Tanh` as `Module` subclasses
- [ ] Implement `GELU` — use the tanh approximation
- [ ] Grad check all activations with random inputs including negative values
- [ ] Implement `Dropout(p)` — correct train/eval behavior, inverted scaling
- [ ] Verify Dropout: in eval mode output equals input exactly
- [ ] Implement `Sequential(*modules)` — chain forward passes, aggregate parameters
- [ ] Build and test: `Sequential(Linear(4,8), ReLU(), Linear(8,2))` forward + backward
- [ ] **Ship:** `layers/linear.py` + `layers/activations.py` + `layers/dropout.py` + `layers/module.py` + tests

---

## Interview Topics

- **Why Kaiming initialization?** Random weights too small → vanishing gradients. Too large → exploding. Kaiming scales by `1/√fan_in`, keeping variance constant through ReLU layers.
- **What is inverted dropout?** Scale surviving activations by `1/(1-p)` at train time so the expected value is unchanged. No adjustment needed at inference — cleaner deployment.
- **Why does Dropout act as a regularizer?** Forces the network to learn redundant representations — any single neuron can be dropped. Equivalent to training an ensemble of 2^n sub-networks.
- **Explain GELU vs ReLU.** GELU is smoother (differentiable everywhere), weights inputs by their probability of being positive under a Gaussian. Used in Transformers. ReLU is simpler and faster.

---

## Definition of Done

- [ ] All layers pass grad check with relative error < 1e-5
- [ ] `Dropout` output is zero-mean when averaged over many forward passes in train mode
- [ ] `Sequential` correctly chains gradients through 5+ layers
- [ ] `parameters()` returns the correct count for any Sequential combination
- [ ] All tests pass with `pytest layers/`
