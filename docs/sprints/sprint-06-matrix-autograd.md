# Sprint 6 — Matrix Autograd Engine

**Theme:** Extend scalar autograd to tensor (matrix) operations
**Duration:** 2 weeks
**Phase:** Phase 2 — Engine

---

## Goal

Extend the scalar `Value` engine to handle n-dimensional tensors. By end of sprint you have a `Tensor` class with matrix autograd, a working finite difference checker for matrices, and an MLP trained on MNIST-small using zero external ML libraries.

---

## Concepts

### From Scalar to Matrix Gradients

- Scalar: `grad` is a float. Matrix: `grad` is a numpy array of the same shape as `data`.
- Matrix multiply backward: for `C = A @ B`, `dL/dA = dL/dC @ Bᵀ` and `dL/dB = Aᵀ @ dL/dC`.
- Sum backward: `dL/dA = dL/dout * ones_like(A)` (broadcast gradient to input shape).
- Broadcasting backward: sum gradients along broadcast dimensions.

### Shape Tracking

- Every `Tensor` stores its `.shape` at creation.
- Every backward function must return a gradient of the same shape as the input.
- Broadcasting in forward means summing along broadcast axes in backward.

### Finite Difference Check for Matrices

- Perturb each element `x[i,j]` independently by ±ε.
- Compare `(f(x+ε_ij) - f(x-ε_ij)) / 2ε` to `analytical_grad[i,j]`.
- Use relative error `|numerical - analytical| / max(|numerical|, |analytical|, 1)` < 1e-5.

### Gradient Accumulation with Broadcasting

- When a scalar bias is added to a (batch, features) tensor, gradient flows back through broadcasting.
- Backward: sum gradient over the batch dimension to recover the bias gradient shape.

---

## Tasks

- [ ] Implement `Tensor` class: wraps numpy array, stores `data`, `grad`, `_backward`, `_prev`
- [ ] Implement `__add__`, `__mul__` elementwise with correct broadcasting backward
- [ ] Implement `matmul` with `dL/dA = dL/dC @ Bᵀ`, `dL/dB = Aᵀ @ dL/dC`
- [ ] Implement `sum(axis)`, `mean(axis)` with correct backward shapes
- [ ] Implement `reshape`, `transpose` with backward passing gradient through same transform
- [ ] Implement `exp`, `log`, `relu` elementwise on tensors
- [ ] Implement `backward()` — same topological sort as scalar engine
- [ ] Write `grad_check(f, inputs, eps=1e-5)` — checks every element of every input tensor
- [ ] Run grad check on: `matmul`, `add`, `sum`, `relu`, `exp`, `log` — all must pass
- [ ] Train 2-layer MLP `[784, 128, 10]` on 1000 MNIST samples — reach > 85% accuracy
- [ ] **Ship:** `engine/tensor.py` + `engine/tensor_ops.py` + `engine/grad_check.py` + `engine/mnist_tensor_mlp.py`

---

## Interview Topics

- **Derive the backward pass for matrix multiply.** `C = A @ B`. `dL/dA = dL/dC @ Bᵀ`. Show the shape math: if A is (m,k), B is (k,n), dL/dC is (m,n), then dL/dA must be (m,k) = (m,n)@(n,k). ✓
- **Why must gradient shapes match input shapes?** Gradient descent updates `param -= lr * grad`. If grad has wrong shape, the update is wrong or crashes.
- **How do you handle broadcasting in backward?** Identify which dimensions were broadcast in forward, sum grad along those dimensions in backward.
- **What is relative error and why use it instead of absolute error?** Gradients can be very large or very small. Relative error normalizes by magnitude, giving a scale-independent check.

---

## Definition of Done

- [ ] Grad check passes for all implemented ops with relative error < 1e-5
- [ ] MLP trains on MNIST small and reaches > 85% accuracy
- [ ] `backward()` works for computation graphs with fan-in (shared tensors)
- [ ] No PyTorch or other ML framework imports anywhere in `engine/`
- [ ] All tests pass with `pytest engine/`
