# ML/DL Mastery — From Math to Visual Low-Code Platform

A bottom-up deep learning platform built to achieve research-grade understanding across NLP, CV, Video, 3D, and Generative AI — culminating in a visual drag-and-drop model builder.

27 sprints across math foundations, classical ML, tensor engines, layer libraries, the Transformer, domain specializations, special topics (adversarial robustness, explainability, efficient inference, MLOps), capstone systems, and a full visual platform.

---

## Project Overview

**Platform:** Visual drag-and-drop ML architecture builder — wire layers into models, train, visualize loss, and export to Python — for NLP, CV, Video, 3D, and Generative pipelines.

**Goal:** Build every component from scratch to achieve deep intuition, then productionize into a real platform. Inspired by tools like Netron, Lobe, and NN-SVG but built with full internal understanding.

**Stack:** Python · NumPy · PyTorch · FastAPI · React · React Flow · Chart.js · PostgreSQL · pgvector · Redis · Docker

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     Frontend (React + React Flow)              │
│        Node Palette · Canvas · Config Panel · Loss Chart       │
└────────────────────┬───────────────────────────────────────────┘
                     │ REST / SSE
┌────────────────────▼───────────────────────────────────────────┐
│                   Platform API (FastAPI)                        │
│     /graph/run · /graph/validate · /graph/export · /metrics    │
└──────────┬─────────────────────────────────────┬──────────────┘
           │ Graph JSON                           │ Metadata
┌──────────▼──────────────┐           ┌──────────▼─────────────┐
│   Execution Engine       │           │   Layer Registry        │
│   DAG executor           │           │   NLP · CV · Video     │
│   Shape inference        │           │   3D · Generative      │
│   Training loop          │           │   node_schema.json     │
└─────────────────────────┘           └────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────┐
│                  Scratch Layer Library                        │
│   engine/ · layers/ · transformer/ · nlp/ · cv/             │
│   video/ · 3d/ · generative/ · multimodal/                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Domain Model

```
Platform
    └── Graph (pipeline JSON)
            ├── Node ──── LayerConfig
            │       └── NodeSchema (type, input_shape, params, output_shape)
            ├── Edge (tensor flow between nodes)
            ├── GraphVersion (immutable snapshot)
            ├── Run
            │       └── RunStep (per-node metrics, shapes, errors)
            └── Export (generated Python code)

LayerRegistry
    ├── NLPNodes    (Tokenizer, Embedding, TransformerBlock, ClassHead...)
    ├── CVNodes     (Conv2D, Pool, ResBlock, ViTPatch, DetectionHead...)
    ├── VideoNodes  (Conv3D, TemporalAttn, FrameEmbedding...)
    ├── 3DNodes     (PointNetMLP, TNet, VoxelGrid, NeRFBlock...)
    └── GenNodes    (VAEEncoder, VAEDecoder, UNetBlock, DiffusionScheduler...)
```

---

## Sprint Map

| # | Sprint | Theme | Learning Focus |
|---|--------|-------|----------------|
| 0 | [Architecture and Scope Freeze](sprints/sprint-00-architecture.md) | System design | Tensor abstraction, graph IR, layer registry design |
| 1 | [Math Foundations I](sprints/sprint-01-math-foundations-1.md) | Linear algebra + calculus | Matrix ops, chain rule, finite differences |
| 2 | [Math Foundations II](sprints/sprint-02-math-foundations-2.md) | Probability + information theory | Softmax, cross-entropy, KL divergence |
| 3 | [Classical ML I — Regression](sprints/sprint-03-classical-ml-1.md) | Supervised learning fundamentals | Linear/logistic regression, regularization, bias-variance |
| 4 | [Classical ML II — Trees & Unsupervised](sprints/sprint-04-classical-ml-2.md) | Ensembles + unsupervised | Decision trees, random forest, gradient boosting, k-means, PCA |
| 5 | [Scalar Autograd Engine](sprints/sprint-05-scalar-autograd.md) | Backpropagation | Computation graph, topological sort, grad accumulation |
| 6 | [Matrix Autograd Engine](sprints/sprint-06-matrix-autograd.md) | Tensor backprop | Matrix grad, broadcast, finite difference check |
| 7 | [Core Layer Primitives](sprints/sprint-07-core-layers.md) | Layer library I | Linear, activations, dropout, Sequential |
| 8 | [Normalization + Optimizers](sprints/sprint-08-normalization-optimizers.md) | Layer library II | BatchNorm, LayerNorm, SGD, Adam, first real training |
| 9 | [Attention Mechanism](sprints/sprint-09-attention.md) | Transformer core | Scaled dot-product, multi-head, causal masking |
| 10 | [Transformer + Character LM](sprints/sprint-10-transformer.md) | Full Transformer | Positional encoding, BPE, nanoGPT, text generation |
| 11 | [NLP Track](sprints/sprint-11-nlp.md) | NLP subfields | Classification, NER, translation, recommendation, summarization |
| 12 | [CV Track](sprints/sprint-12-cv.md) | CV subfields | Conv2D, LeNet, ResNet, ViT, YOLO, U-Net |
| 13 | [Video Track](sprints/sprint-13-video.md) | Temporal modeling | Conv3D, optical flow, Video Transformer, captioning |
| 14 | [3D Track](sprints/sprint-14-3d.md) | 3D understanding | PointNet, voxels, NeRF, marching cubes |
| 15 | [Generative Track](sprints/sprint-15-generative.md) | Generative models | VAE, GAN, DCGAN, DDPM diffusion |
| 16 | [Multimodal Track](sprints/sprint-16-multimodal.md) | Cross-modal learning | CLIP, image captioning, VQA, LoRA fine-tuning |
| 17 | [Adversarial Robustness & Explainability](sprints/sprint-17-adversarial-robustness-explainability.md) | Security + interpretability | FGSM/PGD attacks, adversarial training, Grad-CAM, SHAP/LIME |
| 18 | [Efficient Inference & MLOps](sprints/sprint-18-efficient-inference-mlops.md) | Model compression + serving | Quantization, pruning, distillation, KV-cache, experiment tracking |
| 19 | [Capstone A — NLP System](sprints/sprint-19-capstone-nlp.md) | Production NLP | Multilingual sentiment, trend prediction, explainability API |
| 20 | [Capstone B — CV System](sprints/sprint-20-capstone-cv.md) | Production CV | Real-time detection, object tracking, WebSocket stream |
| 21 | [Capstone C — Video + 3D](sprints/sprint-21-capstone-video-3d.md) | Scene understanding | Depth estimation, SfM, 3D reconstruction from video |
| 22 | [Capstone D — Generative](sprints/sprint-22-capstone-generative.md) | Controllable generation | Conditional DDPM, classifier-free guidance, CLIP-guided diffusion |
| 23 | [Capstone E — Multimodal](sprints/sprint-23-capstone-multimodal.md) | Unified search system | Cross-modal retrieval, VQA API, multimodal demo |
| 24 | [Platform Core](sprints/sprint-24-platform-core.md) | Execution engine | Node schema, layer registry, graph executor, shape inference |
| 25 | [Platform API + UI](sprints/sprint-25-platform-api-ui.md) | Backend + canvas | FastAPI, SSE streaming, React canvas, config panel |
| 26 | [Platform Capstone](sprints/sprint-26-platform-capstone.md) | Full platform v1 | All domain nodes, dataset nodes, export, demo video |

---

## Execution Phases

### Phase 0 — Math Foundations (Sprints 0–2)
Lock architectural decisions and establish the mathematical vocabulary you will use in every line of code that follows. By end of Sprint 2 you can implement any loss function or activation from a paper formula.

### Phase 1 — Classical ML (Sprints 3–4)
Implement regression, regularization, decision trees, ensembles, and unsupervised learning (k-means, PCA) using only NumPy and manual gradient descent. By end of Sprint 4 you have a small classical ML library and a solid intuition for bias/variance, before computation graphs add complexity.

### Phase 2 — Engine (Sprints 5–6)
Build scalar then matrix autograd from scratch. By end of Sprint 6 you have a working tensor engine that computes gradients for arbitrary computation graphs. Train a 2-layer MLP on XOR with zero external ML dependencies.

### Phase 3 — Layer Library (Sprints 7–8)
Implement every primitive layer — Linear, activations, normalization, optimizers — using only your engine. By end of Sprint 8 you train a full MLP on MNIST at >97% accuracy using only your library.

### Phase 4 — Transformer (Sprints 9–10)
Implement attention and the full Transformer block. By end of Sprint 10 you train a character-level GPT on Shakespeare and generate coherent text. This is the moment everything clicks.

### Phase 5 — Domain Specializations (Sprints 11–16)
Six parallel domain tracks: NLP, CV, Video, 3D, Generative, Multimodal. Each is 3–6 weeks of focused implementation and paper reading. By end of Sprint 16 you have implemented 20+ architectures from scratch.

### Phase 6 — Special Topics (Sprints 17–18)
Attack and defend your own models, explain their predictions, then make them small and fast enough to ship. By end of Sprint 18 you can run adversarial attacks/defenses, produce attribution maps, and quantize/prune/distill a model into a served, monitored API.

### Phase 7 — Capstone Systems (Sprints 19–23)
Five production systems — each wraps 2–3 sprints of prior work into a shippable FastAPI backend with a real frontend. By end of Sprint 23 you have five portfolio projects demonstrating end-to-end engineering.

### Phase 8 — Platform (Sprints 24–26)
The capstone of capstones. Build the visual low-code ML platform where every node corresponds to code you personally wrote. By end of Sprint 26 a user can drag Conv2D → ReLU → Pool → Dense, train on CIFAR-10, see live loss, and export clean Python — entirely through your platform.

---

## Repository Structure

```
ml-from-scratch/
├── ml_classical/           # Phase 1: classical ML (regression, trees, ensembles, k-means, PCA)
│   ├── linear_models.py
│   ├── model_selection.py
│   ├── metrics.py
│   ├── trees.py
│   ├── ensembles.py
│   ├── clustering.py
│   ├── pca.py
│   └── naive_bayes.py
├── engine/                 # Phase 2: tensor + autograd
│   ├── value.py
│   ├── tensor.py
│   ├── tensor_ops.py
│   └── grad_check.py
├── layers/                 # Phase 3: layer primitives
│   ├── linear.py
│   ├── activations.py
│   ├── normalization.py
│   ├── dropout.py
│   ├── module.py
│   └── optim.py
├── transformer/            # Phase 4: attention + Transformer
│   ├── attention.py
│   ├── multihead_attn.py
│   ├── transformer_block.py
│   ├── embedding.py
│   ├── bpe_tokenizer.py
│   └── char_gpt.py
├── nlp/                    # Sprint 11 + Capstone A
│   ├── text_classifier.py
│   ├── ner_model.py
│   ├── seq2seq_translation.py
│   ├── embedding_rec.py
│   ├── summarizer.py
│   └── nlp_pipeline_api/
├── cv/                     # Sprint 12 + Capstone B
│   ├── conv2d.py
│   ├── pooling.py
│   ├── lenet.py
│   ├── resnet.py
│   ├── vit.py
│   ├── yolo_head.py
│   ├── unet.py
│   └── cv_pipeline_api/
├── video/                  # Sprint 13 + Capstone C
│   ├── conv3d.py
│   ├── optical_flow.py
│   ├── two_stream.py
│   ├── video_transformer.py
│   ├── video_captioner.py
│   └── video_action_api/
├── 3d/                     # Sprint 14 + Capstone C
│   ├── pointcloud_loader.py
│   ├── pointnet.py
│   ├── tnet.py
│   ├── voxelizer.py
│   ├── nerf.py
│   ├── marching_cubes.py
│   └── pointcloud_api/
├── generative/             # Sprint 15 + Capstone D
│   ├── vae.py
│   ├── gan.py
│   ├── dcgan.py
│   ├── diffusion_forward.py
│   ├── ddpm.py
│   ├── ddpm_sample.py
│   └── gen_api/
├── multimodal/             # Sprint 16 + Capstone E
│   ├── contrastive_loss.py
│   ├── clip_model.py
│   ├── image_captioner.py
│   ├── vqa_model.py
│   ├── lora.py
│   └── multimodal_search_api/
├── security/               # Sprint 17: adversarial robustness + explainability
│   ├── adversarial.py
│   └── explainability.py
├── efficiency/             # Sprint 18: quantization, pruning, distillation, KV-cache
│   ├── quantization.py
│   ├── pruning.py
│   ├── distillation.py
│   └── kv_cache.py
├── mlops/                  # Sprint 18: experiment tracking + serving
│   ├── tracker.py
│   └── serve_api/
├── platform/               # Sprints 24–26
│   ├── backend/
│   │   ├── main.py         # FastAPI app
│   │   ├── executor.py     # Graph execution engine
│   │   ├── shape_inference.py
│   │   ├── codegen.py      # Python export
│   │   ├── registry.py     # Layer registry
│   │   └── nodes/          # All domain node definitions
│   └── frontend/
│       ├── src/
│       │   ├── App.jsx
│       │   ├── NodeConfig.jsx
│       │   └── LossChart.jsx
│       └── package.json
├── notebooks/              # Experiments, ablations, loss plots
├── papers/                 # Annotated paper notes (markdown)
├── _lab/                   # Spike code, benchmarks — never imported
├── docs/
│   └── sprints/            # This directory
├── Makefile
└── docker-compose.yml
```

---

## Continuous Rules

- **Grad check before shipping any layer.** Every backward pass must pass finite difference verification: `(f(x+ε) − f(x−ε)) / 2ε ≈ analytical_grad`. No exceptions.
- **Read the PyTorch source for every layer you implement.** After your scratch version works, open `torch/nn/modules/` and compare. This closes the gap between understanding and production.
- **One paper per sprint minimum.** Read the original paper for every architecture you implement. Annotate it in `papers/` as a markdown file.
- **Keep labs separate.** Experiments, spikes, and benchmarks belong in `_lab/`. Do not pollute production paths.
- **Type hints everywhere.** Every function in the platform layer must be fully typed from Sprint 24 onward.
- **Ship something runnable every sprint.** Every sprint ends with code that executes, produces output, and has at least one test.

---

## Progress Tracker

- [ ] Sprint 0 — Can explain tensor abstraction, graph IR, layer registry design from memory
- [ ] Sprint 1 — Can implement matrix multiply and chain rule without looking anything up
- [ ] Sprint 2 — Can derive softmax gradient and cross-entropy by hand
- [ ] Sprint 3 — Can implement linear/logistic regression with gradient descent and explain regularization
- [ ] Sprint 4 — Can implement decision trees, random forest, gradient boosting, k-means, and PCA from scratch
- [ ] Sprint 5 — Can build scalar autograd from scratch in one sitting
- [ ] Sprint 6 — Can extend autograd to matrices and verify with finite differences
- [ ] Sprint 7 — Can implement Linear + Adam from scratch with correct gradients
- [ ] Sprint 8 — Can train MLP on MNIST >97% using only scratch library
- [ ] Sprint 9 — Can implement multi-head attention from the paper equations alone
- [ ] Sprint 10 — Can train a character-level GPT that generates coherent text
- [ ] Sprint 11 — Can build NER, translation, and recommendation systems end-to-end
- [ ] Sprint 12 — Can build ResNet and ViT from scratch, train on CIFAR-10
- [ ] Sprint 13 — Can explain temporal modeling, implement Conv3D and video Transformer
- [ ] Sprint 14 — Can implement PointNet and a basic NeRF from scratch
- [ ] Sprint 15 — Can implement VAE, GAN, and DDPM diffusion from scratch
- [ ] Sprint 16 — Can implement CLIP contrastive training and LoRA fine-tuning
- [ ] Sprint 17 — Can craft adversarial attacks/defenses and produce model explanations (Grad-CAM, SHAP/LIME)
- [ ] Sprint 18 — Can quantize, prune, and distill a model, and serve it via a monitored API
- [ ] Sprint 19 — Shipped multilingual NLP API with evaluation suite
- [ ] Sprint 20 — Shipped real-time CV detection + tracking with WebSocket stream
- [ ] Sprint 21 — Shipped depth estimation + 3D reconstruction API from video
- [ ] Sprint 22 — Shipped conditional diffusion generation API with CLIP guidance
- [ ] Sprint 23 — Shipped cross-modal multimodal search system
- [ ] Sprint 24 — Platform executor runs any graph, shape inference catches mismatches
- [ ] Sprint 25 — React canvas with live SSE loss chart, config panel, export to Python
- [ ] Sprint 26 — Full platform v1: all domains wired, dataset nodes, working demo video
