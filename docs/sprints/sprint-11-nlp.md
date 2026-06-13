# Sprint 11 — NLP Track

**Theme:** NLP subfields — classification, NER, translation, recommendation, summarization
**Duration:** 4 weeks
**Phase:** Phase 5 — Domain Specializations

---

## Goal

Build five distinct NLP systems end-to-end, each using your Transformer as the backbone. By end of sprint you have a working NLP pipeline API serving classification, NER, translation, recommendations, and summarization from a single FastAPI service.

---

## Concepts

### Text Classification (Encoder + CLS Token)

- BERT-style: prepend a `[CLS]` token. After encoding, the `[CLS]` representation is passed to a classification head.
- Classification head: `Linear(d_model, num_classes) → Softmax`.
- Fine-tuning: load pretrained encoder weights, train only the head (or the whole model at lower lr).

### Named Entity Recognition (Token Classification)

- Instead of one output per sequence, predict one label per token.
- Output: `Linear(d_model, num_entity_classes)` applied to each token position independently.
- Entity types: PER (person), ORG (organization), LOC (location), O (outside).
- Evaluation: entity-level F1 score (strict span match).

### Sequence-to-Sequence Translation

- Encoder encodes source sentence. Decoder generates target token by token.
- Cross-attention in decoder: attends over encoder outputs at each decode step.
- Teacher forcing during training: feed ground-truth target tokens as decoder input.
- BLEU score: precision-based metric for evaluating translation quality.

### Embedding-Based Recommendation

- Learn embeddings for items (products, articles, songs) from co-occurrence or interaction data.
- At inference: embed query item → find nearest neighbors in embedding space by cosine similarity.
- Training: skip-gram or contrastive objective (positive pairs vs negative samples).

### Extractive Summarization

- Rank sentences by importance. Select top-k.
- Importance signal: attention scores from a pretrained encoder, or a trained binary classifier.
- No text generation — just sentence selection and concatenation.

---

## Tasks

- [ ] Fine-tune Transformer encoder on IMDb sentiment (25k train) — binary classification on `[CLS]`
- [ ] Evaluate: accuracy, precision, recall, F1 on test set — save `results/sentiment_eval.json`
- [ ] Implement NER token classification head — train on CoNLL-2003 (English NER)
- [ ] Implement entity-level F1 evaluation — distinguish span-level from token-level accuracy
- [ ] Implement seq2seq Transformer with cross-attention decoder for En→Vi translation
- [ ] Implement beam search decoding (beam width 4) — compare to greedy decode quality
- [ ] Compute BLEU score on validation set
- [ ] Build item embedding model on MovieLens dataset — train with skip-gram objective
- [ ] Implement cosine similarity search: embed a movie, retrieve top-10 similar movies
- [ ] Implement extractive summarizer: score sentences by mean attention, select top-3
- [ ] Build FastAPI service: `/predict/sentiment`, `/predict/ner`, `/translate`, `/recommend`, `/summarize`
- [ ] Write end-to-end integration test — POST to each endpoint, validate response schema
- [ ] Read BERT paper — annotate in `papers/bert.md`
- [ ] Implement MLM pretraining objective: randomly mask 15% of tokens, predict masked tokens
- [ ] **Ship:** `nlp/` directory + `nlp/nlp_pipeline_api/` + `results/nlp_eval.md`

---

## Interview Topics

- **Why is [CLS] used for classification?** It aggregates information from the entire sequence through self-attention. Alternatively, mean-pool all token representations.
- **What is teacher forcing?** During training, feed the ground-truth previous token as decoder input rather than the model's own prediction. Faster convergence but can cause exposure bias at inference.
- **BLEU score limitations?** Only measures n-gram precision, not recall or fluency. Insensitive to meaning — a perfect paraphrase scores poorly. ROUGE-L and BERTScore are better alternatives.
- **Why cosine similarity for embeddings?** Normalized dot product — invariant to vector magnitude. Two items can have different interaction counts but similar relative patterns.

---

## Definition of Done

- [ ] Sentiment classifier: >90% accuracy on IMDb test set
- [ ] NER: >85% entity-level F1 on CoNLL-2003 validation set
- [ ] Translation: positive BLEU score improving with training (no need for state-of-art)
- [ ] Recommender: top-10 results are semantically related to query item
- [ ] FastAPI service starts and all 5 endpoints return 200 with correct schema
- [ ] Integration tests pass with `pytest nlp/nlp_pipeline_api/test_api.py`
