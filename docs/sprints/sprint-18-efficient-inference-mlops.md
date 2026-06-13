# Sprint 18 — Efficient Inference & MLOps

**Theme:** Making models smaller, faster, and production-ready
**Duration:** 2 weeks
**Phase:** Phase 6 — Special Topics

---

## Goal

Take a trained model from Sprints 7-16 and make it efficient and deployable: quantize it, prune it, distill it into a smaller model, benchmark the tradeoffs, and wrap it in a served, monitored, versioned pipeline. By end of sprint you can explain the compute/memory/accuracy tradeoffs of each compression technique and have a minimal but real MLOps pipeline (experiment tracking, model registry, served endpoint, monitoring).

---

## Concepts

### Quantization

- Reduce numeric precision of weights/activations: FP32 → FP16 → INT8 (or lower).
- **Post-training quantization (PTQ):** quantize a trained model directly — fast, some accuracy loss.
- **Quantization-aware training (QAT):** simulate quantization during training (fake-quantize ops in the forward pass) so the model adapts — better accuracy at low precision.
- INT8 quantization: `x_int8 = round(x_fp32 / scale) + zero_point`. Scale/zero_point computed per-tensor or per-channel from calibration data.
- Tradeoff: ~4x memory reduction (FP32→INT8) and faster inference on hardware with INT8 kernels, at the cost of some accuracy.

### Pruning

- **Unstructured pruning:** zero out individual weights below a magnitude threshold — high sparsity possible but needs sparse-aware hardware/libraries to realize speedups.
- **Structured pruning:** remove entire channels/filters/heads — directly reduces compute (FLOPs) and works on any hardware, but coarser-grained.
- **Iterative pruning + fine-tuning:** prune a bit, fine-tune to recover accuracy, repeat — better than one-shot pruning to a high sparsity.
- Lottery Ticket Hypothesis (conceptual): sparse subnetworks found by pruning a trained network can sometimes be retrained from scratch (with the same init) to similar accuracy.

### Knowledge Distillation

- Train a small **student** model to match a large **teacher** model's outputs, not just the hard labels.
- Distillation loss: combination of standard cross-entropy with true labels and KL divergence (Sprint 2) between student and teacher **soft** probabilities (with temperature `T` to soften the distribution).
- Soft labels carry more information than one-hot labels ("dark knowledge") — e.g., relative similarity between wrong classes.
- Works across architectures — e.g., distill a Transformer into a smaller Transformer or even a different architecture.

### Efficient Architectures (Survey-Level)

- **Depthwise separable convolutions** (MobileNet): split a standard conv into depthwise + pointwise — drastically fewer FLOPs.
- **Grouped/multi-query attention:** reduce KV cache size for Transformer inference by sharing key/value heads across query heads.
- **KV-cache:** for autoregressive generation (your Sprint 10 char-GPT), cache previously computed keys/values so each new token doesn't recompute attention over the whole sequence — O(1) per-token instead of O(n).

### Benchmarking

- Measure: latency (p50/p99), throughput (samples/sec), memory footprint, and accuracy — together, not in isolation.
- **Batching:** larger batches improve throughput but increase latency per request — the classic serving tradeoff.
- Profile where time actually goes (e.g., is matmul the bottleneck, or data loading/preprocessing?) before optimizing.

### MLOps Pipeline Basics

- **Experiment tracking:** log hyperparameters, metrics, and artifacts for every training run (even a simple CSV/JSON logger counts — the principle is reproducibility).
- **Model registry/versioning:** every trained model is an immutable, versioned artifact with metadata (training data version, hyperparameters, metrics) — connects to the `GraphVersion` concept from the platform (Sprint 24).
- **Serving:** wrap a model in a minimal API (FastAPI) with a `/predict` endpoint, input validation, and a health check.
- **Monitoring:** log prediction latency, input distribution stats (basic drift detection — compare live input stats to training stats), and error rates.
- **CI for ML:** beyond unit tests — a "model test" that checks the model still meets a minimum accuracy threshold on a fixed eval set before it's marked deployable.

---

## Tasks

- [ ] Implement `quantize_tensor(x, num_bits)` and `dequantize_tensor(x_q, scale, zero_point)` — verify round-trip error is bounded
- [ ] Apply post-training INT8 quantization to your Sprint 12 CV classifier — measure accuracy drop and memory reduction
- [ ] Implement a simple quantization-aware training wrapper (fake-quant in forward pass) — compare accuracy to PTQ at the same bit-width
- [ ] Implement `magnitude_prune(model, sparsity)` — zero out smallest-magnitude weights per layer
- [ ] Run iterative prune + fine-tune on your Sprint 12 CV classifier — plot accuracy vs. sparsity (0%, 50%, 70%, 90%)
- [ ] Implement structured channel pruning for a CNN — remove lowest-importance filters, verify the model still runs with reduced shapes
- [ ] Implement knowledge distillation: train a small student MLP/CNN using a larger trained teacher's soft labels + temperature
- [ ] Compare student-with-distillation vs. student-trained-alone at the same architecture/size — plot accuracy gap
- [ ] Implement KV-cache for your Sprint 10 char-GPT — benchmark generation speed with vs. without cache
- [ ] Write a benchmarking harness: measure latency (p50/p99) and throughput for original vs. quantized vs. pruned vs. distilled models
- [ ] Build a minimal experiment tracker — JSON/SQLite logger that records hyperparams, metrics, and artifact paths per run
- [ ] Wrap one compressed model in a FastAPI `/predict` endpoint with input validation and a `/health` check
- [ ] Add basic monitoring middleware: log request latency and a simple input-stats drift check (compare batch mean/std to training stats)
- [ ] Write a "model gate" test: pytest test that loads the latest registered model and asserts accuracy ≥ threshold on a fixed eval set
- [ ] **Ship:** `efficiency/quantization.py` + `efficiency/pruning.py` + `efficiency/distillation.py` + `efficiency/kv_cache.py` + `mlops/tracker.py` + `mlops/serve_api/` + benchmark report

---

## Interview Topics

- **Why does INT8 quantization usually preserve accuracy reasonably well, but INT4 often doesn't without QAT?** Weight/activation distributions are roughly bell-shaped with most values near zero; INT8 (256 levels) has enough resolution to represent this distribution's range with small quantization error, while INT4 (16 levels) introduces error large enough to shift decision boundaries — QAT lets the model adapt its weights to be robust to that coarser grid.
- **Structured vs. unstructured pruning — why does structured pruning give real speedups on commodity hardware while unstructured often doesn't?** Unstructured sparsity produces irregular memory access patterns that dense matrix-multiply hardware/kernels can't exploit without specialized sparse kernels; structured pruning produces smaller *dense* tensors that standard kernels run faster on directly.
- **Why does knowledge distillation with a temperature-softened teacher distribution help more than training on hard labels alone?** Soft labels encode the teacher's relative confidence across all classes ("dark knowledge") — e.g., a "7" is more like a "1" than a "0" — giving the student a richer training signal per example than a single correct-class bit.
- **Explain why KV-caching changes autoregressive generation from O(n²) to O(n) total attention compute.** Without caching, generating token `t` recomputes K/V for all `t` previous tokens — O(n) work per token, O(n²) total. With caching, only the new token's K/V are computed each step and appended — O(1) per token (for the new K/V), O(n) total.
- **What's the difference between latency and throughput, and why can optimizing one hurt the other?** Latency = time for one request; throughput = requests/sec across many. Batching improves throughput (amortizes overhead) but increases the latency of individual requests (they wait for the batch to fill) — a direct tradeoff tuned via max batch size / timeout.
- **What does a "model gate" in CI protect against that a normal unit test doesn't?** Unit tests check code correctness (the training/inference code runs without error); a model gate checks the *artifact's* quality (the trained weights actually perform above a threshold) — code can be correct but produce a bad model due to data issues, hyperparameters, or training instability.

---

## Definition of Done

- [ ] INT8 quantized model is ~4x smaller than FP32 with documented accuracy delta
- [ ] Pruning-vs-accuracy plot shows graceful degradation up to at least 70% sparsity with fine-tuning
- [ ] Distilled student outperforms the same-architecture student trained without distillation
- [ ] KV-cache generation is measurably faster than non-cached for sequences of length ≥ 100 tokens
- [ ] Benchmark report compares latency/throughput/memory/accuracy across all 4 variants (original, quantized, pruned, distilled) in one table
- [ ] FastAPI `/predict` endpoint returns correct predictions and `/health` returns 200
- [ ] Model gate test fails when pointed at a deliberately under-trained model, and passes for the real one
- [ ] All tests pass with `pytest efficiency/ mlops/`
