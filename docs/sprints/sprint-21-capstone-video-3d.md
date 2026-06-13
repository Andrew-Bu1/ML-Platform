# Sprint 21 — Capstone C: Video Scene Understanding + 3D Reconstruction

**Theme:** Depth estimation, temporal segmentation, sparse 3D reconstruction from video
**Duration:** 3 weeks
**Phase:** Phase 7 — Capstone Systems

---

## Goal

Ship a video analysis API: POST a short video → receive per-frame segmentation, depth maps, and a sparse 3D point cloud reconstruction — all rendered interactively in a browser.

---

## Tasks

### Temporal Segmentation

- [ ] Apply U-Net segmentation per-frame — naive approach
- [ ] Add temporal consistency: smooth masks across frames using exponential moving average of predictions
- [ ] Evaluate temporal consistency: measure IoU variance across consecutive frames
- [ ] **Ship:** `video/scene_pipeline.py`

### Monocular Depth Estimation

- [ ] Implement depth estimation encoder-decoder: ResNet encoder + U-Net-style decoder predicting per-pixel depth
- [ ] Train on NYU Depth v2 dataset (RGB-D indoor scenes)
- [ ] Loss: scale-invariant log loss + gradient matching loss (penalize depth discontinuities)
- [ ] Evaluate: absolute relative error (AbsRel), RMS error, δ<1.25 threshold accuracy
- [ ] **Ship:** `video/depth_estimator.py` + `results/depth_eval.json`

### Sparse 3D Reconstruction (Structure from Motion lite)

- [ ] Implement feature detection: Harris corner detector from scratch
- [ ] Implement feature matching: brute-force descriptor matching (use SIFT via OpenCV for descriptors)
- [ ] Implement 8-point algorithm for essential matrix estimation
- [ ] Recover relative camera pose from essential matrix (R, t)
- [ ] Triangulate matched points across two frames → 3D points
- [ ] Chain across multiple frames to build sparse point cloud
- [ ] **Ship:** `video/sfm_lite.py`

### Visualization + API

- [ ] Implement 3D point cloud renderer using Three.js — interactive rotate/zoom in browser
- [ ] Implement depth map renderer: colormap (viridis/plasma) applied to depth prediction
- [ ] Build API:
  - `POST /analyze/video` — returns job ID
  - `GET /jobs/{id}/status` — poll for completion
  - `GET /jobs/{id}/segmentation` — per-frame mask PNGs
  - `GET /jobs/{id}/depth` — per-frame depth colormap PNGs
  - `GET /jobs/{id}/pointcloud` — `.ply` download
  - `GET /jobs/{id}/viewer` — HTML page with Three.js viewer
- [ ] Implement async job queue with Redis — video processing runs in background worker
- [ ] **Ship:** `video/video3d_service/` + `results/reconstruction_demo.html`

---

## Definition of Done

- [ ] Depth estimation: AbsRel < 0.2 on NYU Depth validation set
- [ ] Temporal segmentation: IoU variance between consecutive frames < 0.05
- [ ] SfM produces visible point cloud structure for a video with clear parallax
- [ ] Three.js viewer renders point cloud interactively in browser
- [ ] API job queue handles a 30-second video end-to-end without timeout
- [ ] All endpoints respond correctly with integration tests
