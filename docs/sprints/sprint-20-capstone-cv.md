# Sprint 20 — Capstone B: Real-Time CV Detection + Tracking

**Theme:** Object detection + tracking + WebSocket streaming
**Duration:** 3 weeks
**Phase:** Phase 7 — Capstone Systems

---

## Goal

Ship a real-time computer vision system: webcam or video input → detect objects → track across frames → stream annotated output to a browser via WebSocket. Benchmark against YOLO-tiny baseline.

---

## System Design

```
Input (webcam/video file)
    ↓
Frame extractor (OpenCV)
    ↓
YOLO detection model (your implementation)
    ↓
SORT tracker (Kalman filter + Hungarian assignment)
    ↓
Annotator (draw boxes + track IDs + class labels)
    ↓
WebSocket stream → Browser canvas display
    ↓
FastAPI WebSocket endpoint + React frontend
```

---

## Tasks

### Detection Model

- [ ] Train your YOLO detection head on a custom 5-class dataset (define classes yourself — e.g., person/car/bike/dog/chair)
- [ ] Implement NMS with configurable IoU threshold — verify removes duplicate detections
- [ ] Implement mAP@0.5 evaluation — compute per-class AP and mean AP
- [ ] Compare your YOLO vs YOLOv5n (nano) baseline on same dataset
- [ ] **Ship:** `cv/custom_yolo/train.py` + `cv/custom_yolo/eval.py` + `results/detection_benchmark.md`

### Object Tracking

- [ ] Implement Kalman filter for 2D bounding box state prediction: state = (cx, cy, w, h, vx, vy, vw, vh)
- [ ] Implement Hungarian algorithm for optimal assignment of detections to tracks
- [ ] Implement SORT tracker: predict → match → update → create new tracks → remove dead tracks
- [ ] Test tracker on a synthetic sequence: 5 objects with known trajectories, verify IDs persist
- [ ] **Ship:** `cv/tracker.py` + `cv/test_tracker.py`

### Real-Time Pipeline

- [ ] Implement frame extraction loop from webcam or video file using OpenCV
- [ ] Wire detection → tracking → annotation in a single inference pipeline
- [ ] Annotate frames: draw bounding boxes, class labels, track IDs, confidence scores
- [ ] Measure FPS with profiler — identify bottleneck (model inference vs I/O)
- [ ] Optimize: batch inference if multiple frames queued, resize input appropriately
- [ ] **Ship:** `cv/realtime_detect.py` — runs standalone from CLI

### API + Frontend

- [ ] Implement FastAPI WebSocket endpoint: accepts video file, streams annotated JPEG frames
- [ ] Implement React page: video upload → WebSocket connection → display frames on `<canvas>`
- [ ] Add frame counter, FPS display, and detection count overlay in browser
- [ ] Implement HTTP endpoint: `POST /analyze/image` — single image → detection JSON
- [ ] Write load test: simulate 3 concurrent WebSocket streams, measure latency degradation
- [ ] **Ship:** `cv/cv_pipeline_api/` + browser demo + `results/benchmark.ipynb`

---

## Definition of Done

- [ ] Custom YOLO mAP@0.5 > 0.5 on validation set
- [ ] SORT tracker maintains IDs correctly on synthetic test sequence
- [ ] Real-time pipeline runs at > 10 FPS on CPU (model at lowest resolution)
- [ ] WebSocket stream displays annotated video in browser with < 500ms latency
- [ ] Benchmark notebook compares your model vs YOLOv5n on speed + accuracy
- [ ] Integration tests pass for HTTP endpoint
