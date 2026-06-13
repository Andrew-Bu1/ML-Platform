# Sprint 1 — Math Foundations I

**Theme:** Linear algebra + calculus you will use in code
**Duration:** 2 weeks
**Phase:** Phase 0 — Math Foundations

---

## Goal

Build the mathematical vocabulary to implement every neural network operation without looking up formulas. By the end of this sprint you can implement matrix multiply, dot product, transpose, and the chain rule from memory — and verify them with unit tests.

---

## Concepts

### Matrix Multiplication

- `C = A @ B` where A is (m×k) and B is (k×n) gives C of shape (m×n).
- Entry `C[i,j] = sum(A[i,:] * B[:,j])` — dot product of row i of A with column j of B.
- Naive implementation: three nested loops. This is correct but slow. Understand it before optimizing.
- Matrix multiply is the backbone of every Linear layer, every attention score computation, every weight update.

### Elementwise Operations and Broadcasting

- Elementwise: same shape tensors, operate position by position. `C[i,j] = A[i,j] + B[i,j]`.
- Broadcasting: numpy extends a smaller tensor to match a larger shape by replicating along size-1 dimensions.
- Rule: align shapes from the right. Dimensions of size 1 expand. Mismatched sizes that aren't 1 are an error.
- Critical for understanding why `bias` in a Linear layer (shape `[out]`) can be added to output (shape `[batch, out]`).

### Transpose

- `A.T` swaps axes: shape (m×n) becomes (n×m).
- Appears everywhere: `QKᵀ` in attention, `Wᵀ` in the backward pass of a Linear layer.

### The Chain Rule

- For composed functions `z = f(g(x))`: `dz/dx = dz/dg · dg/dx`.
- In a neural network, the chain rule is applied layer by layer from the loss back to the inputs.
- In code: each operation stores a `backward_fn` closure that implements its local derivative contribution.
- The full backward pass chains these closures together in reverse topological order.

### Finite Differences

- Numerical gradient approximation: `df/dx ≈ (f(x+ε) - f(x-ε)) / 2ε` with ε = 1e-5.
- Use this to verify any analytical gradient implementation. If they disagree beyond floating point tolerance (~1e-5), the backward pass is wrong.
- This is your most important debugging tool for the next four sprints.

### Jacobians

- For vector-valued functions, the gradient generalizes to a **Jacobian matrix** `J[i,j] = df_i/dx_j`.
- Most neural network ops have structured Jacobians (diagonal, or decomposable) — you rarely compute them explicitly, but understanding the shape matters for backprop.

---

## Tasks

- [ ] Implement `matrix_mul(A, B)` with three nested loops — no numpy `@` — with shape assertions
- [ ] Implement `dot(a, b)`, `transpose(A)`, `elementwise_add(A, B)`, `elementwise_mul(A, B)`
- [ ] Write unit tests: verify `matrix_mul` output matches `np.matmul` for (3×4)@(4×5)
- [ ] Implement broadcasting manually for a (3,1) + (1,4) case — then verify numpy matches
- [ ] Derive the chain rule for `z = sin(x²)` on paper — implement `dz_dx(x)` in Python
- [ ] Write `finite_diff_check(f, x, eps=1e-5)` — returns numerical gradient for scalar f
- [ ] Verify chain rule implementation passes finite difference check for 3 different functions
- [ ] Watch 3Blue1Brown "Essence of Linear Algebra" series — write one paragraph of notes per video in `papers/linear_algebra_notes.md`
- [ ] **Ship:** `engine/matrix_ops.py` + `engine/test_matrix_ops.py` — all tests pass

---

## Interview Topics

- **Why is matrix multiply O(n³)?** Three nested loops, each 0..n. Can be reduced with Strassen (O(n^2.81)) but cache behavior matters more in practice.
- **What does broadcasting actually do?** Explain the shape alignment rules. Give an example where it silently produces a wrong result if shapes are accidentally compatible.
- **Explain the chain rule in terms of a computation graph.** Why do you traverse in reverse order? What does "accumulate gradients" mean for fan-in nodes?
- **Why finite differences and not symbolic differentiation?** Symbolic is exact but slow and complex. Finite differences are fast to implement and sufficient for verifying analytical grads.

---

## Definition of Done

- [ ] `matrix_mul` correctly handles arbitrary (m,k) @ (k,n) — tested for at least 5 shape combinations
- [ ] `finite_diff_check` returns gradient that matches analytical result to within 1e-5
- [ ] Chain rule implemented and verified for: `f(x) = x²`, `g(x) = sin(x²)`, `h(x) = exp(x³ + 1)`
- [ ] All tests in `engine/test_matrix_ops.py` pass with `pytest`
- [ ] `papers/linear_algebra_notes.md` has at least one paragraph per 3B1B video
