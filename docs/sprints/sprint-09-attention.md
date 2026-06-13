# Sprint 9 — Attention Mechanism

**Theme:** Self-attention from scratch — the core of modern NLP and CV
**Duration:** 2 weeks
**Phase:** Phase 4 — Transformer

---

## Goal

Implement scaled dot-product attention and multi-head attention from scratch following the original paper equations. By end of sprint you have a working multi-head attention module that passes grad check and matches PyTorch's `nn.MultiheadAttention` output on identical inputs.

---

## Concepts

### Scaled Dot-Product Attention

- Given queries Q (n×d_k), keys K (m×d_k), values V (m×d_v):
- `Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V`
- Scaling by `1/√d_k` prevents dot products from growing large (which pushes softmax into saturation/tiny-gradient regions) for large embedding dimensions.
- Output shape: (n×d_v) — each query position gets a weighted sum of values.

### Causal (Autoregressive) Masking

- For language modeling, position i should not attend to positions j > i.
- Apply an upper-triangular mask of `-inf` before softmax: `softmax(-inf) = 0`.
- After masking, each position only attends to itself and previous positions.

### Multi-Head Attention

- Instead of one attention with d_model dimensions, use h heads each with d_k = d_model/h.
- Project Q, K, V separately for each head with learned projections W_Q, W_K, W_V (each d_model × d_k).
- Run attention independently for each head.
- Concatenate all head outputs → (n × d_model), then apply output projection W_O.
- **Why multiple heads?** Each head can attend to different types of relationships simultaneously — syntactic, semantic, positional.

### Cross-Attention

- Used in encoder-decoder models: queries from decoder, keys+values from encoder.
- Allows the decoder to attend over all encoder positions at each decoding step.
- Same math as self-attention, different sources for Q vs K/V.

---

## Tasks

- [ ] Implement `scaled_dot_product_attention(Q, K, V, mask=None)` — equation from paper
- [ ] Verify output shape: Q(8,10,64) K(8,10,64) V(8,10,64) → output(8,10,64)
- [ ] Implement causal mask generation for sequence length L — upper triangular -inf matrix
- [ ] Verify masked attention: position 0 only sees position 0; position 3 sees positions 0–3
- [ ] Implement `MultiHeadAttention(d_model, num_heads)` with W_Q, W_K, W_V, W_O projections
- [ ] Grad check `MultiHeadAttention` with (batch=2, seq=8, d_model=64, heads=8)
- [ ] Verify output matches `torch.nn.MultiheadAttention` on identical weights + inputs (within 1e-5)
- [ ] Implement cross-attention: `MultiHeadAttention.forward(x_dec, x_enc, x_enc)` where K,V come from encoder
- [ ] Read "Attention is All You Need" — annotate every equation in `papers/attention_is_all_you_need.md`
- [ ] **Ship:** `transformer/attention.py` + `transformer/mask.py` + `transformer/test_attention.py` + paper notes

---

## Interview Topics

- **Why divide by √d_k?** In high dimensions, dot products grow with d_k. Large logits → near-zero softmax gradients → slow learning. √d_k keeps variance of dot products roughly 1.
- **What is attention complexity?** O(n²·d) in sequence length n. Quadratic — reason why long-context models are expensive. Flash Attention uses tiling to reduce memory to O(n).
- **What is the difference between self-attention and cross-attention?** Self: Q,K,V all come from same sequence. Cross: Q from one sequence (decoder), K,V from another (encoder output).
- **Why is masking applied before softmax rather than after?** -inf → 0 probability after softmax. Masking after softmax would need explicit zeroing and renormalization — more complex and less numerically stable.

---

## Definition of Done

- [ ] `scaled_dot_product_attention` output matches manual calculation for a (2,4,8) input
- [ ] Causal mask zeros out all upper-triangular attention weights
- [ ] `MultiHeadAttention` output matches `torch.nn.MultiheadAttention` within 1e-5
- [ ] Grad check passes for all attention components
- [ ] Paper annotations cover equations 1–3 and the architecture diagram
- [ ] All tests pass with `pytest transformer/`
