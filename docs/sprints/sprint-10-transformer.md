# Sprint 10 — Full Transformer + Character Language Model

**Theme:** Complete Transformer architecture and train a generative model
**Duration:** 2 weeks
**Phase:** Phase 4 — Transformer

---

## Goal

Build a complete Transformer decoder (GPT-style) using your layer library and train a character-level language model on Shakespeare. By end of sprint the model generates coherent English text. This is the milestone where everything built since Sprint 1 comes together.

---

## Concepts

### Positional Encoding

- Transformers have no built-in notion of sequence order (attention is permutation-invariant).
- Sinusoidal positional encoding: `PE[pos, 2i] = sin(pos / 10000^(2i/d_model))`, `PE[pos, 2i+1] = cos(...)`.
- Learned positional embeddings (GPT style): a `(max_seq_len, d_model)` embedding table trained end-to-end.
- Add positional encoding to token embeddings before the first Transformer block.

### Transformer Block (Decoder-Only)

```
x = x + MultiHeadAttention(LayerNorm(x), causal=True)  # pre-norm
x = x + FFN(LayerNorm(x))
```

- Pre-norm (LayerNorm before sublayer) is more stable than original post-norm.
- FFN: `Linear(d_model, 4*d_model) → GELU → Linear(4*d_model, d_model)`.
- Residual connections: allow gradients to flow directly to early layers.

### BPE Tokenizer

- Byte Pair Encoding: start with characters, iteratively merge the most frequent adjacent pair.
- Builds a vocabulary of subword units. Balances vocabulary size vs sequence length.
- For character-level LM: skip BPE, use character vocabulary directly.
- For word-level: implement BPE merge table and encode/decode functions.

### Language Model Training

- Input: token sequence `[t0, t1, ..., tN-1]`.
- Target: shifted by one `[t1, t2, ..., tN]`.
- Loss: cross-entropy at each position, averaged over sequence and batch.
- Sampling: at inference, feed generated tokens back as input autoregressively.

### Sampling Strategies

- **Greedy:** always pick highest probability token. Repetitive.
- **Top-k:** sample from top k tokens by probability. More varied.
- **Temperature:** divide logits by T before softmax. T>1 = more random, T<1 = more peaked.

---

## Tasks

- [ ] Implement sinusoidal `PositionalEncoding(d_model, max_len)` module
- [ ] Implement learned `PositionalEmbedding(max_len, d_model)` as alternative
- [ ] Implement `TransformerBlock(d_model, num_heads, ffn_dim, dropout)` with pre-norm + residuals
- [ ] Implement `GPT(vocab_size, d_model, num_heads, num_layers, max_len)` — stack of blocks + lm_head
- [ ] Implement `generate(prompt, max_tokens, temperature, top_k)` sampling method
- [ ] Implement character-level tokenizer: `encode(text) → List[int]`, `decode(ids) → str`
- [ ] Implement BPE tokenizer: `train(corpus, vocab_size)`, `encode`, `decode`
- [ ] Load Shakespeare dataset, create character-level training batches
- [ ] Train GPT (6 layers, 6 heads, d_model=384) for 5000 steps — loss should reach < 1.5
- [ ] Generate 500 characters of Shakespeare-style text from "To be or not" prompt
- [ ] Compare sinusoidal vs learned positional encoding — loss curves in same plot
- [ ] **Ship:** `transformer/transformer_block.py` + `transformer/embedding.py` + `transformer/bpe_tokenizer.py` + `transformer/char_gpt.py` + `results/shakespeare_sample.txt`

---

## Interview Topics

- **Why pre-norm instead of post-norm (original paper)?** Pre-norm (LayerNorm before sublayer) produces more stable gradients in deep networks. Post-norm requires warmup and careful lr scheduling.
- **Why 4x expansion in FFN?** Empirically works well — the bottleneck forces useful compression. Larger ratios (8x) work in very large models. The "4x" is convention from the original paper.
- **What is perplexity?** `exp(cross_entropy_per_token)`. For character-level models, ~3 is human-level. Your model at 1.5 nats loss has perplexity ≈ 4.5.
- **Temperature sampling vs top-k.** Temperature is global — scales all logits. Top-k is local — cuts off the long tail. Often used together. Nucleus (top-p) samples from smallest set of tokens covering p% of probability.

---

## Definition of Done

- [ ] `TransformerBlock` output matches PyTorch equivalent within 1e-4
- [ ] Character GPT trains to loss < 1.5 on Shakespeare within 5000 steps
- [ ] `generate()` produces coherent (not random) character sequences
- [ ] BPE tokenizer correctly encodes and decodes a test sentence round-trip
- [ ] Loss curves saved as PNG, sample output saved as TXT
- [ ] All tests pass with `pytest transformer/`
