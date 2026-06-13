# Sprint 12 — CV Track

**Theme:** Computer vision — classification, detection, segmentation
**Duration:** 4 weeks
**Phase:** Phase 5 — Domain Specializations

---

## Goal

Implement the core CV architecture progression from scratch: LeNet → ResNet → ViT → YOLO → U-Net. By end of sprint you have a CV pipeline API that classifies, detects, and segments images.

---

## Concepts

### Conv2D and the Convolution Operation

- Slide a kernel of shape (C_out, C_in, kH, kW) across input (N, C_in, H, W).
- Output at position (i,j): dot product of kernel with input patch centered at (i,j).
- **im2col:** reshape input patches into columns, then matmul with kernel matrix — turns convolution into one big matrix multiply. Efficient on hardware.
- Backward: `dL/d_input` is a convolution of `dL/d_output` with flipped kernel. `dL/d_kernel` is a convolution of input with `dL/d_output`.

### Receptive Field

- The region of the input that affects a given output position.
- Grows with depth: after k conv layers with kernel size 3, receptive field ≈ 2k+1.
- Pooling and strided convolutions grow it faster.

### Residual Connections (ResNet)

- `output = F(x) + x` where F is the conv block.
- Gradient highway: gradient flows directly through the skip connection.
- Solves vanishing gradient in networks > 20 layers.
- If shapes differ: use a 1×1 convolution to project x to the right shape.

### Patch Tokenization (ViT)

- Divide image into P×P patches (e.g., 16×16 from a 224×224 image = 196 patches).
- Flatten each patch → linear projection → token embedding.
- Same Transformer block as NLP — just different input tokenization.

### Object Detection (YOLO Head)

- Divide image into S×S grid. Each cell predicts B bounding boxes + confidence + C class probabilities.
- Box parameterization: (cx, cy, w, h) relative to cell. `cx = sigmoid(tx)`, `w = exp(tw) * anchor_w`.
- Loss: localization (MSE on boxes) + objectness (BCE) + classification (BCE).
- NMS (non-maximum suppression): remove duplicate boxes by IoU overlap threshold.

### Semantic Segmentation (U-Net)

- Encoder path: series of conv blocks + MaxPool (downsample).
- Decoder path: series of upsample (bilinear or transposed conv) + conv blocks.
- Skip connections: concatenate encoder feature maps to decoder at each resolution.
- Output: per-pixel class probabilities of shape (N, num_classes, H, W).

---

## Tasks

- [ ] Implement `Conv2D(in_ch, out_ch, kernel_size, stride, padding)` with im2col + grad check
- [ ] Implement `MaxPool2d(kernel_size, stride)` and `AvgPool2d` with backward
- [ ] Build `LeNet5` — train on MNIST, reach >99% accuracy
- [ ] Build `ResNet18` with `BasicBlock(conv-bn-relu-conv-bn + skip)` — train on CIFAR-10
- [ ] Ablation: ResNet18 with skip connections vs without — plot both loss curves
- [ ] Build `ViT` — patch embed + positional embed + 6 Transformer blocks + classification head
- [ ] Train ViT on CIFAR-10 — compare accuracy with ResNet18 at same parameter count
- [ ] Implement YOLO detection head on top of ResNet backbone
- [ ] Train YOLO on small COCO subset (5 classes) — verify boxes appear on sample images
- [ ] Implement NMS: given boxes + scores, suppress overlapping boxes by IoU threshold
- [ ] Implement `UNet` encoder-decoder with skip connections
- [ ] Train U-Net on Oxford Pets segmentation dataset — visualize predicted masks
- [ ] Read ResNet paper — annotate in `papers/resnet.md`
- [ ] Read ViT paper — annotate in `papers/vit.md`
- [ ] Write 1-page comparison: CNN inductive bias vs ViT attention — `papers/cnn_vs_vit.md`
- [ ] Build CV pipeline API: `/classify`, `/detect`, `/segment` endpoints
- [ ] **Ship:** `cv/` directory + `cv/cv_pipeline_api/` + `results/cv_samples/`

---

## Interview Topics

- **Why does im2col turn convolution into matrix multiply?** Reshaping input patches to columns allows the kernel to be applied as a single matmul — highly optimized on GPU/CPU via BLAS.
- **What is IoU (Intersection over Union)?** Area of overlap / area of union. Used in detection to measure box quality and in NMS to remove duplicate predictions.
- **Why do ViTs need more data than CNNs?** CNNs have translation equivariance built in (weight sharing). ViTs learn this from data. On small datasets CNNs win; on large datasets ViTs match or exceed.
- **U-Net skip connections vs ResNet skip connections.** U-Net skips bridge encoder to decoder across resolution levels — pass fine-grained spatial info. ResNet skips are within same resolution — ease gradient flow.

---

## Definition of Done

- [ ] `Conv2D` passes grad check with relative error < 1e-5
- [ ] LeNet: >99% MNIST accuracy
- [ ] ResNet18: >90% CIFAR-10 accuracy
- [ ] ViT: >85% CIFAR-10 accuracy
- [ ] YOLO detection boxes visible on sample images (qualitative check)
- [ ] U-Net segmentation masks visually reasonable on Oxford Pets
- [ ] CV API starts and all 3 endpoints return correct schema
- [ ] All tests pass with `pytest cv/`
