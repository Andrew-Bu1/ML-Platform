# Sprint 13 — Video Track

**Theme:** Temporal modeling, optical flow, video understanding
**Duration:** 3 weeks
**Phase:** Phase 5 — Domain Specializations

---

## Goal

Extend CV to the time dimension. Build video classification, optical flow estimation, and a video Transformer. By end of sprint you ship an action recognition API that takes a short video clip and returns top-3 action labels with timestamps.

---

## Concepts

### 3D Convolution (Conv3D)

- Extends Conv2D with a temporal kernel dimension: shape (C_out, C_in, kT, kH, kW).
- Input: (N, C, T, H, W) — batch × channels × time × height × width.
- Captures local spatiotemporal patterns simultaneously.
- Computationally expensive: T×H×W feature maps vs H×W for 2D.

### Two-Stream Networks

- **Spatial stream:** standard CNN on RGB frames — captures appearance.
- **Temporal stream:** CNN on stacked optical flow fields — captures motion.
- Fuse predictions (average or learned fusion) for final classification.
- Each stream is independently pretrained then combined.

### Optical Flow

- Per-pixel 2D motion vectors between consecutive frames: `(u[x,y], v[x,y])`.
- **Lucas-Kanade:** assumes small motion, constant brightness, spatial smoothness. Solves a local linear system per patch.
- Dense optical flow: compute flow at every pixel (Farneback algorithm for classical).
- Flow visualization: encode magnitude as saturation, direction as hue (HSV color wheel).

### Video Transformer

- **Divided Space-Time Attention (TimeSformer):** separate spatial attention (across patches in one frame) and temporal attention (across same patch position across frames).
- Input: video → (T × H/P × W/P) spatiotemporal tokens.
- Positional encoding: 3D (time + height + width) or factorized.

### Temporal Segments

- For long videos: sample T clips from equally spaced segments.
- Aggregate predictions across clips (average, max, or learned aggregation).

---

## Tasks

- [ ] Implement `Conv3D(in_ch, out_ch, kernel_size_t, kernel_size_s)` — extend im2col to 3D
- [ ] Grad check `Conv3D` for input (2, 3, 8, 32, 32)
- [ ] Build `C3D` architecture — stack of Conv3D + Pool3D — train on UCF-101 (16-frame clips)
- [ ] Implement Lucas-Kanade optical flow from scratch — tested on two synthetic frames with known motion
- [ ] Visualize optical flow on a video pair as HSV color image
- [ ] Build two-stream network: ResNet spatial + optical flow CNN temporal — fuse softmax scores
- [ ] Build `VideoTransformer` — ViT patch tokenization over T frames with temporal attention
- [ ] Compare: C3D vs Two-stream vs VideoTransformer on UCF-101 accuracy
- [ ] Implement LSTM-based video captioner: CNN frame encoder → mean pool → LSTM → caption tokens
- [ ] Read TimeSformer paper — annotate in `papers/timesformer.md`
- [ ] Build action recognition API: POST video file → return top-3 actions + confidence scores
- [ ] **Ship:** `video/` directory + `video/video_action_api/` + `results/video_samples/`

---

## Interview Topics

- **Why is Conv3D expensive?** A 3×3×3 kernel has 27 weights vs 9 for Conv2D. With T=16 frames and spatial resolution, the feature map is 16× larger. Memory and compute multiply.
- **What is the brightness constancy assumption in Lucas-Kanade?** Pixel intensity doesn't change between frames: `I(x,y,t) = I(x+u, y+v, t+1)`. Violated by lighting changes, fast motion, occlusion.
- **Divided attention vs joint spatiotemporal attention.** Joint: each token attends to all T×H×W tokens — O((T·H·W)²). Divided: spatial attention per frame + temporal attention per location — O(T·(H·W)² + H·W·T²). Much cheaper.
- **What is action recognition vs action detection?** Recognition: classify clip → one action label. Detection: localize and classify actions in time (temporal detection) and space (spatial).

---

## Definition of Done

- [ ] `Conv3D` passes grad check
- [ ] Optical flow visualization produces correct HSV motion field on synthetic test case
- [ ] At least one model (C3D or VideoTransformer) trains and improves on UCF-101 above random
- [ ] Video captioner produces grammatical captions (qualitative check)
- [ ] API accepts video file, returns JSON with top-3 actions and confidence scores
- [ ] All tests pass with `pytest video/`
