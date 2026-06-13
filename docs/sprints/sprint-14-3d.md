# Sprint 14 — 3D Track

**Theme:** Point clouds, voxels, implicit representations
**Duration:** 3 weeks
**Phase:** Phase 5 — Domain Specializations

---

## Goal

Build 3D understanding models from scratch. By end of sprint you ship a 3D object classifier that accepts a `.ply` point cloud, classifies it, and renders an interactive 3D visualization.

---

## Concepts

### Point Cloud Representation

- An unordered set of (x,y,z) points sampled from a 3D surface.
- No grid structure — permutation invariant (order of points doesn't matter).
- Typical size: 1024–16384 points per object.
- Common formats: `.ply`, `.obj`, `.pcd`.

### PointNet Architecture

- Key insight: a function on an unordered set can be learned as `f({x_i}) = g(max_i(h(x_i)))`.
- **Per-point MLP** `h`: same weights applied independently to each point — equivariant to permutation.
- **Global max-pool** `g`: produces a single global descriptor — invariant to permutation.
- Classification head: fully connected layers on global descriptor → class logits.

### T-Net (Input/Feature Transform)

- Mini-network that predicts a transformation matrix (3×3 for input, 64×64 for features).
- Apply transform to input points or intermediate features to normalize orientation.
- Regularization: penalize transform matrix away from orthogonality with `||TᵀT - I||²_F`.

### Voxel Grid

- Divide 3D space into a regular grid of cubes (voxels).
- Binary occupancy: `V[i,j,k] = 1` if any point falls in that voxel.
- Apply 3D CNNs on the occupancy grid — analogous to 2D CNNs on images.
- Tradeoff: voxels scale as O(r³) in resolution. Memory intensive.

### Neural Radiance Fields (NeRF)

- Implicit representation: `MLP(x, y, z, θ, φ) → (R, G, B, σ)` where σ is density.
- Volume rendering: integrate color and density along camera rays to produce pixel color.
- Learns a continuous 3D scene from 2D images with known camera poses.
- **Positional encoding:** map (x,y,z) through sinusoidal functions before MLP — helps learn high-frequency detail.

### Marching Cubes

- Extract mesh surface from a scalar field (e.g., NeRF density).
- For each voxel: check corner values against threshold → look up mesh triangle configuration from a 256-entry table.

---

## Tasks

- [ ] Implement `.ply` parser — read vertices and optional colors from binary/ASCII PLY files
- [ ] Implement `farthest_point_sample(points, N)` — sample N points maximally spread in 3D
- [ ] Implement `PointNetMLP(input_dim, hidden_dims)` — per-point shared MLP
- [ ] Implement global max-pool aggregation → classification head
- [ ] Implement `TNet(k)` — predicts k×k transform matrix with regularization loss
- [ ] Train `PointNet` on ModelNet40 (40-class shape dataset) — target >85% accuracy
- [ ] Implement voxelizer: `pointcloud_to_voxels(points, resolution=32)` → binary occupancy grid
- [ ] Verify voxelization: voxelize a sphere, visualize with matplotlib 3D scatter
- [ ] Implement basic NeRF MLP with positional encoding (`sin/cos` at multiple frequencies)
- [ ] Implement volume rendering: cast rays, sample points along ray, integrate color + density
- [ ] Train NeRF on a simple synthetic scene (NeRF Blender dataset, one object) — render novel views
- [ ] Implement marching cubes — extract mesh from trained NeRF density field
- [ ] Read PointNet paper — annotate in `papers/pointnet.md`
- [ ] Read NeRF paper — annotate in `papers/nerf.md`
- [ ] Build 3D classification API: POST `.ply` → return class label + confidence + 3D viz HTML
- [ ] **Ship:** `3d/` directory + `3d/pointcloud_api/` + `results/3d_viz.html`

---

## Interview Topics

- **Why is PointNet permutation invariant?** Per-point MLP applies same function to each point independently. Global max-pool selects the maximum activation across all points — result is the same regardless of input order.
- **NeRF training requires posed images — what does that mean?** Camera extrinsics (position + orientation) for each training image must be known. COLMAP can estimate these from unposed images.
- **Voxels vs point clouds vs implicit.** Voxels: regular grid, 3D CNNs apply, expensive. Point clouds: sparse, irregular, PointNet-style. Implicit: continuous, compact, requires ray casting to render.
- **What is volume rendering?** Sum contributions along a ray: `C = ∫ T(t) · σ(r(t)) · c(r(t)) dt` where T(t) is transmittance (probability of not hitting anything before t).

---

## Definition of Done

- [ ] PointNet trains on ModelNet40 and reaches >85% classification accuracy
- [ ] Voxelization produces visually correct occupancy grids for sphere and cube inputs
- [ ] NeRF renders novel views of training scene (qualitative — looks like the scene)
- [ ] Marching cubes produces a watertight mesh from a sphere density field
- [ ] API accepts `.ply` file, returns JSON + interactive 3D HTML visualization
- [ ] All tests pass with `pytest 3d/`
