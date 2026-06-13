# Sprint 23 — Capstone E: Unified Multimodal Search System

**Theme:** Cross-modal retrieval, VQA, image captioning — unified into one product
**Duration:** 3 weeks
**Phase:** Phase 7 — Capstone Systems

---

## Goal

Ship a unified multimodal search product: users query with an image or text and retrieve matching results across both modalities. Add VQA and captioning as bonus endpoints. This is the most complete portfolio piece in the roadmap.

---

## System Design

```
┌────────────────────────────────────────────────┐
│              React Demo Frontend                │
│  Drag image OR type text → see results         │
│  Click result → expand with VQA + caption      │
└──────────────────┬─────────────────────────────┘
                   │ REST
┌──────────────────▼─────────────────────────────┐
│             FastAPI Backend                     │
│  /embed/image  /embed/text                     │
│  /search       /caption   /vqa                 │
└──────┬──────────────────────────┬──────────────┘
       │                          │
┌──────▼──────┐          ┌────────▼──────────────┐
│  pgvector   │          │  Model servers         │
│  Embedding  │          │  CLIP · Captioner      │
│  Index      │          │  VQA model             │
└─────────────┘          └───────────────────────┘
```

---

## Tasks

### Embedding Index

- [ ] Design pgvector schema: `embeddings(id, modality, source_path, embedding vector(512), metadata jsonb)`
- [ ] Index 10k images from MSCOCO + 10k captions — store embeddings in pgvector
- [ ] Implement `embed_and_index(image_paths, captions)` batch indexing script
- [ ] Add HNSW index in pgvector for fast approximate nearest neighbor search
- [ ] Benchmark: query latency at 10k, 50k, 100k indexed embeddings
- [ ] **Ship:** `multimodal/embedding_store.py` + `multimodal/index_pipeline.py`

### Cross-Modal Search

- [ ] Implement `search_by_image(image_path, top_k=10)` — embed image, query pgvector for nearest texts
- [ ] Implement `search_by_text(query, top_k=10)` — embed text, query pgvector for nearest images
- [ ] Implement `search_by_either(query, modality)` — unified interface
- [ ] Add filters: `modality`, `min_similarity`, `metadata_filter` (e.g., source dataset)
- [ ] **Ship:** `multimodal/cross_modal_search.py`

### Captioning + VQA Endpoints

- [ ] Load your image captioner from Sprint 16 — wrap in inference function
- [ ] `POST /caption` — accepts image, returns 3 caption candidates with beam search
- [ ] Load your VQA model from Sprint 16
- [ ] `POST /vqa` — accepts image + question, returns top-3 answer candidates with confidence
- [ ] **Ship:** `multimodal/captioning_endpoint.py` + `multimodal/vqa_endpoint.py`

### React Demo Frontend

- [ ] Implement drag-and-drop image upload with preview
- [ ] Implement text search input with debounced query
- [ ] Display results grid: thumbnail + similarity score + source modality badge
- [ ] Click result → expand panel showing caption + VQA interface
- [ ] Add loading states, error handling, empty state illustration
- [ ] **Ship:** `multimodal/multimodal_demo/` (React app)

### Production Hardening

- [ ] Add Redis caching: cache embedding results for 1 hour by input hash
- [ ] Add rate limiting: max 60 requests/minute per IP
- [ ] Add input validation: max image size 10MB, max text 512 tokens
- [ ] Write load test: 20 concurrent search requests, measure p50/p95/p99 latency
- [ ] Write `README.md` with setup instructions, example requests, architecture diagram
- [ ] **Ship:** `multimodal/multimodal_search_api/` + load test results

---

## Definition of Done

- [ ] Text query retrieves semantically correct images for > 70% of test queries
- [ ] Image query retrieves semantically correct captions for > 65% of test images
- [ ] VQA endpoint answers factual questions about images correctly (qualitative, 10 test cases)
- [ ] Search latency p95 < 200ms for 10k indexed embeddings
- [ ] React demo runs locally with `npm start` and all interactions work
- [ ] Full system documented in README with architecture diagram
