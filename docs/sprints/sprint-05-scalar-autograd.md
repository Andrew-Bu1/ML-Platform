# Sprint 5 — Scalar Autograd Engine

**Theme:** Backpropagation from scratch — scalar computation graph
**Duration:** 2 weeks
**Phase:** Phase 2 — Engine

---

## Goal

Build a working scalar autograd engine modeled on Karpathy's micrograd. By end of sprint you can compute exact gradients for any scalar computation graph and train a 2-layer MLP on a toy dataset using zero external ML dependencies.

---

## Concepts

### Computation Graph

- Every operation on a `Value` creates a new `Value` node with references to its inputs.
- This forms a **DAG** — nodes are values, edges are operations.
- Forward pass: evaluate each operation to get the output.
- Backward pass: traverse in reverse topological order, calling each node's `backward_fn`.

### Topological Sort

- Order nodes so each node appears after all its inputs.
- Algorithm: DFS — push node to list after visiting all children.
- Reverse the list. This is your backward pass execution order.

### Backward Function Closures

- Each operation stores a closure capturing its inputs.
- When called, the closure computes `d_output/d_input` and adds it to `input.grad`.
- Example for `z = x * y`: `x.grad += y.data * z.grad`, `y.grad += x.data * z.grad`.
- **Adding** (not setting) gradients handles fan-in: multiple paths to the same node accumulate.

### Supported Operations

- `+` (add), `*` (mul), `**` (power), `-` (neg/sub), `/` (div)
- `exp(x)`, `log(x)`, `tanh(x)`, `relu(x)`
- Each needs: a forward computation and a backward function.

---

## Tasks

- [ ] Implement `Value` class: stores `data`, `grad=0.0`, `_backward=lambda:None`, `_prev=set()`
- [ ] Implement `__add__`, `__mul__`, `__pow__` with backward closures
- [ ] Derive and implement `__neg__`, `__sub__`, `__truediv__` from above
- [ ] Implement `exp()`, `log()`, `tanh()`, `relu()` as methods with backward closures
- [ ] Implement `backward()` method: topological sort then call `_backward` in reverse
- [ ] Write `test_scalar_grad.py`: verify `d(x*y)/dx = y` and `d(x**2)/dx = 2x`
- [ ] Implement gradient check: compare `backward()` result vs finite differences for 5 expressions
- [ ] Build 2-layer MLP using `Value` — `[2, 4, 1]` architecture, manual parameter init
- [ ] Train MLP on XOR dataset (4 points) — loss should reach < 0.01 within 1000 steps
- [ ] **Ship:** `engine/value.py` + `engine/engine.py` + `engine/test_scalar_grad.py` + `engine/xor_mlp.py`

---

## Interview Topics

- **What is a computation graph?** Directed graph where nodes are values and edges are operations. Essential for automatic differentiation.
- **Why topological sort for backward pass?** Guarantees a node's gradient is fully accumulated before it is used to compute its inputs' gradients.
- **Why accumulate (+=) instead of assign (=) gradients?** A node can be used multiple times in a graph (fan-in, weight sharing). Each path contributes a gradient — they must be summed.
- **What is the difference between `.backward()` and calling each `_backward` manually?** `.backward()` orchestrates the topological sort. Each `_backward` is one operation's local derivative.

---

## Definition of Done

- [ ] `Value` supports `+`, `*`, `**`, `-`, `/`, `exp`, `log`, `tanh`, `relu`
- [ ] `backward()` produces gradients matching finite differences to within 1e-5 for all ops
- [ ] XOR MLP trains to loss < 0.01 within 1000 gradient steps
- [ ] No numpy in `value.py` — pure Python scalars only
- [ ] All tests pass with `pytest`
