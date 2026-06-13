# Sprint 26 — Platform Capstone: Full Platform v1

**Theme:** All domain nodes wired, dataset nodes, demo video, paper draft
**Duration:** 4 weeks
**Phase:** Phase 8 — Platform (Final)

---

## Goal

Ship platform v1. Every domain you studied — NLP, CV, Video, 3D, Generative, Multimodal — is available as nodes in the visual canvas. A user with no ML background can wire a ResNet, train on CIFAR-10, see live results, and export the code. Write a technical blog post or paper draft about the platform's architecture.

---

## What v1 Means

- All 35+ layer types are registered, have correct schemas, and work in the executor.
- All 6 domain palettes are visible in the frontend with correct grouping.
- 8+ dataset nodes available (MNIST, CIFAR-10, IMDb, MSCOCO, UCF-101, ModelNet40, CelebA, custom CSV).
- 8+ preset templates covering all domains.
- Export generates runnable Python for every template.
- End-to-end demo: drag → connect → train → export — recorded as video.

---

## Tasks

### NLP Nodes

- [ ] Register and test: `Tokenizer`, `TokenEmbedding`, `PositionalEncoding`, `TransformerEncoderBlock`, `TransformerDecoderBlock`, `MultiHeadAttention`, `ClassificationHead`, `NERHead`, `LMHead`
- [ ] Add NLP palette to frontend — group: Embeddings / Encoder / Decoder / Heads
- [ ] Template: `BERT_Classifier` — Tokenizer → TransformerEncoderBlock×6 → CLS → ClassificationHead
- [ ] Template: `CharGPT` — TokenEmbedding → TransformerDecoderBlock×6 → LMHead
- [ ] **Ship:** `platform/backend/nodes/nlp_nodes.py` + `platform/frontend/src/palettes/NLPPalette.jsx`

### CV Nodes

- [ ] Register and test: `Conv2D`, `MaxPool2D`, `AvgPool2D`, `Flatten`, `ResidualBlock`, `ViTPatchEmbed`, `DetectionHead`, `SegmentationHead`
- [ ] Add CV palette — group: Convolution / Pooling / Architecture Blocks / Heads
- [ ] Template: `LeNet5` — Conv2D×2 → Pool×2 → Flatten → Linear×2 → ClassificationHead
- [ ] Template: `ResNet18` — Conv2D → ResidualBlock×8 → AvgPool → ClassificationHead
- [ ] Template: `ViT_small` — ViTPatchEmbed → TransformerEncoderBlock×6 → ClassificationHead
- [ ] **Ship:** `platform/backend/nodes/cv_nodes.py` + `platform/frontend/src/palettes/CVPalette.jsx`

### Video Nodes

- [ ] Register and test: `Conv3D`, `TemporalAttention`, `FrameEmbedding`, `VideoClassHead`, `OpticalFlowNode`
- [ ] Add Video palette
- [ ] Template: `C3D_Classifier` — Conv3D×5 → Pool3D×3 → Flatten → Linear → VideoClassHead
- [ ] **Ship:** `platform/backend/nodes/video_nodes.py` + `platform/frontend/src/palettes/VideoPalette.jsx`

### 3D Nodes

- [ ] Register and test: `PointNetMLP`, `TNet`, `GlobalMaxPool`, `VoxelGrid`, `NeRFBlock`
- [ ] Add 3D palette
- [ ] Template: `PointNet_Classifier` — PointNetMLP×3 → GlobalMaxPool → Linear×3 → ClassificationHead
- [ ] **Ship:** `platform/backend/nodes/nodes_3d.py` + `platform/frontend/src/palettes/Nodes3DPalette.jsx`

### Generative Nodes

- [ ] Register and test: `VAEEncoder`, `VAEDecoder`, `Reparameterize`, `UNetBlock`, `DiffusionScheduler`, `CLIPEmbedding`
- [ ] Add Generative palette
- [ ] Template: `VAE` — VAEEncoder → Reparameterize → VAEDecoder
- [ ] Template: `DDPM_UNet` — UNetBlock (encoder) × 4 → Bottleneck → UNetBlock (decoder) × 4
- [ ] **Ship:** `platform/backend/nodes/gen_nodes.py` + `platform/frontend/src/palettes/GenerativePalette.jsx`

### Dataset Nodes

- [ ] Implement `DatasetNode` for: MNIST, CIFAR-10, IMDb, MSCOCO, UCF-101, ModelNet40, CelebA
- [ ] Implement `CustomCSVDataset` — upload CSV, specify feature + label columns
- [ ] Implement `CustomImageFolderDataset` — upload zip of images organized by class folder
- [ ] Dataset nodes appear at top of palette as special input-only nodes (no input handles)
- [ ] **Ship:** `platform/backend/nodes/dataset_nodes.py`

### Templates + Polish

- [ ] Create 8 complete template graphs covering all domains — verify each trains end to end
- [ ] Add template preview thumbnails (architecture diagram as SVG)
- [ ] Implement undo/redo in canvas (React Flow history)
- [ ] Implement zoom to fit, zoom in/out keyboard shortcuts
- [ ] Add node search: type to filter palette
- [ ] Add minimap for large graphs
- [ ] **Ship:** `platform/frontend/src/` — polished frontend

### Demo + Documentation

- [ ] Record 5-minute demo video: build ResNet from scratch in canvas, train on CIFAR-10, watch live loss, export Python code
- [ ] Write `platform/README.md` — full setup guide, architecture overview, node reference
- [ ] Write `papers/platform_architecture.md` — technical blog post draft covering: graph IR design, shape inference, SSE streaming, code generation, layer registry pattern
- [ ] Deploy platform to a public URL (Fly.io or Railway) — link in README
- [ ] **Ship:** `platform_v1/` + demo video + README + paper draft

---

## Definition of Done

- [ ] All 35+ layer types work in executor — tested with a graph containing one of each domain
- [ ] All 8 dataset nodes load data correctly
- [ ] All 8 templates load, validate, and train end to end
- [ ] Export produces runnable Python for all 8 templates — verified by executing in subprocess
- [ ] Demo video recorded and linked in README
- [ ] Platform is accessible at a public URL
- [ ] Paper draft is at least 1500 words covering architecture decisions
- [ ] All backend tests pass (`pytest platform/`)
- [ ] All frontend tests pass (`npm test`)
