# Sprint 16 — Multimodal Track

**Theme:** CLIP, image captioning, visual Q&A, LoRA fine-tuning
**Duration:** 3 weeks
**Phase:** Phase 5 — Domain Specializations

---

## Goal

Build systems that bridge vision and language. By end of sprint you have a multimodal search API: upload an image or type a query, retrieve matching results from both modalities.

---

## Concepts

### Contrastive Learning (CLIP)

- Learn a shared embedding space for images and text.
- **InfoNCE loss:** for a batch of (image, text) pairs, pull matching pairs together, push non-matching apart.
- `L = -mean(diag(log_softmax(similarity_matrix / temperature)))`.
- Similarity matrix: `S[i,j] = cosine_similarity(image_embed_i, text_embed_j) / τ`.
- Symmetric: compute loss in both image→text and text→image directions.

### Cross-Modal Retrieval

- Index a database of images and texts with their embeddings.
- At query time: embed query → compute cosine similarity to all indexed embeddings → rank.
- Image query → text results. Text query → image results.

### Image Captioning (Vision-Language)

- Encoder: ViT → image token embeddings.
- Decoder: GPT-style Transformer autoregressively generates caption tokens.
- Cross-attention: decoder attends over image token embeddings at each generation step.
- Training: teacher forcing on ground-truth captions.

### Visual Question Answering (VQA)

- Input: image + question text.
- Encode image with ViT, encode question with text Transformer.
- Fuse: concatenate image tokens + question tokens → self-attention over joint sequence.
- Output head: classification over answer vocabulary (most VQA datasets have fixed answer set).

### LoRA (Low-Rank Adaptation)

- Instead of updating `W` (d×d), learn `ΔW = A·B` where A is (d×r) and B is (r×d), with r << d.
- Inject: `W' = W + ΔW = W + A·B`. Only A and B are trained.
- Memory: `2·d·r` params vs `d²`. For d=1024, r=16: 32k vs 1M — 30× reduction.
- Apply to: Q, V projection matrices in attention (most effective).

---

## Tasks

- [ ] Implement `contrastive_loss(image_embeds, text_embeds, temperature=0.07)` — InfoNCE, symmetric
- [ ] Verify loss: identical pairs should give low loss, shuffled pairs high loss
- [ ] Build CLIP model: `ViTEncoder + TransformerTextEncoder + shared projection heads`
- [ ] Train CLIP on Flickr30k (30k image-caption pairs) — verify image-text retrieval improves
- [ ] Implement cosine similarity retrieval with numpy — index 1000 items, query in < 10ms
- [ ] Build image captioner: ViT encoder + GPT decoder with cross-attention — train on MSCOCO subset
- [ ] Evaluate captioning: compute BLEU-4 and CIDEr on validation set
- [ ] Build VQA model: joint image+text encoding → answer classification on VQA v2 subset
- [ ] Implement `LoRA(module, rank=16)` — wraps a Linear, injects A·B into forward pass
- [ ] Wrap LoRA around Q and V projections in your Transformer attention — verify grad only flows to A, B
- [ ] Fine-tune a HuggingFace model (e.g., BERT) using your LoRA — compare memory to full fine-tune
- [ ] Read CLIP paper — annotate in `papers/clip.md`
- [ ] Build multimodal search API: `/embed/image`, `/embed/text`, `/search` (cross-modal retrieval)
- [ ] Build pgvector index for embeddings — store and search via PostgreSQL
- [ ] **Ship:** `multimodal/` + `multimodal/multimodal_search_api/` + `results/multimodal_demo/`

---

## Interview Topics

- **Why temperature τ in contrastive loss?** Controls sharpness of softmax. Low τ → hard negatives dominate, unstable. High τ → all pairs treated similarly, weak signal. τ=0.07 is a common default.
- **How is LoRA rank chosen?** Hyperparameter. r=4–64 is typical. Higher rank = more expressive but more parameters. r=16 is a good default for most tasks.
- **What is catastrophic forgetting?** Fine-tuning on new task overwrites pretrained weights. LoRA avoids this: original weights are frozen, only adapters change. Original task performance preserved.
- **VQA vs image captioning as tasks.** Captioning is open-ended generation (harder to evaluate). VQA is closed-set classification (easier to evaluate but may miss nuance).

---

## Definition of Done

- [ ] CLIP retrieval: given a text query, top-1 retrieved image is semantically correct for >50% of queries
- [ ] Captioner produces grammatical captions (qualitative check on 10 images)
- [ ] LoRA fine-tune uses < 5% of parameters of full fine-tune — verified with `param.requires_grad` count
- [ ] pgvector search returns results in < 50ms for 10k indexed embeddings
- [ ] Multimodal search API returns top-5 cross-modal results with similarity scores
- [ ] All tests pass with `pytest multimodal/`
