# Sprint 19 — Capstone A: Production NLP System

**Theme:** Multilingual sentiment analysis + trend prediction + explainability
**Duration:** 3 weeks
**Phase:** Phase 7 — Capstone Systems

---

## Goal

Ship a production-grade NLP system serving Vietnamese and English content. It classifies sentiment, predicts rating trends over time, and exposes attention-based explainability — all behind a typed, tested FastAPI service.

---

## System Design

```
┌──────────────────────────────────────┐
│            FastAPI Service           │
│  /predict/sentiment                  │
│  /predict/trend                      │
│  /explain  (attention visualization) │
│  /health + /metrics                  │
└──────────────┬───────────────────────┘
               │
┌──────────────▼───────────────────────┐
│         Model Pipeline               │
│  mBERT tokenizer                     │
│  → LoRA-fine-tuned mBERT encoder     │
│  → [CLS] representation              │
│  → Sentiment head (binary)           │
│  → Trend head (regression over seq)  │
└──────────────┬───────────────────────┘
               │
┌──────────────▼───────────────────────┐
│         Data Pipeline                │
│  Raw Vietnamese + English reviews    │
│  → Cleaning + normalization          │
│  → mBERT tokenization                │
│  → PostgreSQL storage                │
└──────────────────────────────────────┘
```

---

## Tasks

### Data Pipeline

- [ ] Scrape or download Vietnamese product review dataset (e.g., AIVIVN sentiment dataset)
- [ ] Combine with English IMDb / Amazon review dataset
- [ ] Implement cleaning pipeline: normalize unicode, remove HTML, strip excessive whitespace
- [ ] Store cleaned reviews in PostgreSQL with schema: `(id, text, language, label, timestamp)`
- [ ] Write dataset stats: language distribution, label balance, text length distribution
- [ ] **Ship:** `nlp/data_pipeline.py` + `nlp/dataset_stats.ipynb`

### Model

- [ ] Load `bert-base-multilingual-cased` tokenizer and encoder via HuggingFace
- [ ] Wrap Q and V projections with your `LoRA(rank=16)` implementation
- [ ] Add binary sentiment head: `Linear(768, 2)`
- [ ] Add trend head: `Linear(768, 1)` — predicts rating value (regression)
- [ ] Fine-tune on combined dataset — early stopping on validation F1
- [ ] Evaluate: accuracy, F1, precision, recall per language
- [ ] **Ship:** `nlp/multilingual_sentiment.py` + `nlp/trend_predictor.py` + `results/nlp_eval.json`

### Explainability

- [ ] Extract per-token attention weights from the last Transformer layer
- [ ] Implement `explain(text) → List[(token, attention_weight)]`
- [ ] Render HTML visualization: tokens colored by attention intensity
- [ ] **Ship:** `nlp/explainability.py` + `results/attention_viz_example.html`

### API

- [ ] `POST /predict/sentiment` — `{"text": "...", "language": "vi|en"}` → `{"label": 0|1, "confidence": float}`
- [ ] `POST /predict/trend` — `{"reviews": [...], "timestamps": [...]}` → `{"predicted_rating": float}`
- [ ] `POST /explain` — `{"text": "..."}` → `{"tokens": [...], "weights": [...]}`
- [ ] `GET /health` — returns model status and last inference latency
- [ ] Add Prometheus metrics: request count, latency histogram, error rate
- [ ] Write full evaluation suite: precision, recall, F1, confusion matrix notebook
- [ ] Write integration tests for all endpoints with pytest
- [ ] **Ship:** `nlp/nlp_pipeline_api/` + `results/eval_suite.ipynb`

---

## Definition of Done

- [ ] Vietnamese sentiment: >80% accuracy (mBERT is multilingual, not Vietnamese-specialized)
- [ ] English sentiment: >90% accuracy
- [ ] Trend predictor: Pearson correlation > 0.6 on validation set
- [ ] Attention visualization renders correctly in browser for both languages
- [ ] All 4 API endpoints respond correctly, integration tests pass
- [ ] README in `nlp/nlp_pipeline_api/README.md` explains how to run the service
