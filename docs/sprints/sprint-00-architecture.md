# Sprint 0 — Architecture and Scope Freeze

**Theme:** System design, domain modeling, repository structure
**Duration:** 1 week
**Phase:** Phase 0 — Foundations

---

## Goal

Lock every architectural decision before writing production code. By the end of this sprint you can explain the full platform design from memory, draw the domain model, describe how a visual node graph compiles into a training execution plan, and defend why the layer-registry architecture was chosen.

---

## Concepts

### Tensor Abstraction

- A **tensor** is an n-dimensional array with a shape, a dtype, and optionally a gradient buffer.
- Every layer in a neural network is a function `f: Tensor → Tensor`.
- The scratch implementation wraps numpy arrays. PyTorch wraps CUDA/C++ kernels. The interface is the same.
- Key operations: `matmul`, `add`, `reshape`, `sum`, `mean`, `exp`, `log`, `transpose`, `broadcast`.

### Computation Graph and Autograd

- Every operation on a tensor creates a **node** in a computation graph.
- Each node stores: its output value, a reference to its input nodes, and a `backward_fn` closure.
- **Forward pass** — evaluate the graph from inputs to outputs.
- **Backward pass** — traverse the graph in reverse topological order, calling each `backward_fn` to accumulate gradients.
- **Gradient accumulation** — gradients from multiple downstream nodes must be summed (chain rule for fan-in).

### Layer Registry Design

- The platform needs to know about every layer type: its name, its parameters, its expected input shape, its output shape formula.
- A **layer registry** is a dictionary mapping string names to layer classes and their JSON schema.
- The visual canvas reads the registry to render the node palette.
- The executor reads the registry to instantiate layers from graph JSON.
- Adding a new layer = adding one class + one registry entry. Nothing else changes.

### Graph as Intermediate Representation (IR)

- The canvas produces a **graph JSON** — nodes and edges describing the model architecture.
- The graph is not directly executable. It must be **compiled** into an execution plan:
  1. Validate the graph (no cycles, shapes compatible, required params present).
  2. Topological sort nodes into execution order.
  3. Resolve each node to a concrete layer instance from the registry.
  4. Run forward() through each layer in order.
- Separating the graph from the execution plan means the graph is portable — it can be versioned, exported to Python, or run on any backend.

### DAG Theory

- A **DAG** (directed acyclic graph) has directed edges and no cycles.
- **Topological sort** — an ordering where every node appears after all its dependencies. This is the forward pass execution order.
- **Cycle detection** — DFS with visited/in-progress coloring. A back-edge indicates a cycle. Cycles are invalid: they mean a layer depends on its own output.
- **Shape propagation** — traverse the DAG in topological order, computing output shapes layer by layer. Catch mismatches before any weights are allocated.

### Domain Model

```
Platform
    └── Graph (pipeline JSON)
            ├── Node ──── LayerConfig (type, params)
            │       └── NodeSchema  (input_shape, output_shape formula)
            ├── Edge    (source_node_id → target_node_id)
            ├── GraphVersion (immutable snapshot)
            └── Run
                    └── RunStep (node_id, status, shape, loss, error)

LayerRegistry
    ├── Key: "Conv2D"
    │       Value: { class: Conv2D, schema: { filters: int, kernel_size: int, ... } }
    ├── Key: "TransformerBlock"
    └── ...
```

### Graph JSON Schema

```json
{
  "schema_version": "1",
  "domain": "cv",
  "nodes": [
    {
      "id": "node-1",
      "type": "layer",
      "layer_type": "Conv2D",
      "label": "First conv",
      "position": { "x": 100, "y": 200 },
      "config": { "in_channels": 1, "out_channels": 32, "kernel_size": 3, "padding": 1 }
    },
    {
      "id": "node-2",
      "type": "layer",
      "layer_type": "ReLU",
      "label": "Activation",
      "position": { "x": 300, "y": 200 },
      "config": {}
    }
  ],
  "edges": [
    { "id": "edge-1", "source": "node-1", "target": "node-2" }
  ],
  "training": {
    "optimizer": "Adam",
    "lr": 0.001,
    "epochs": 10,
    "loss": "CrossEntropy"
  }
}
```

### Monorepo Structure

```
ml-from-scratch/
├── engine/         # Tensor + autograd (no deps except numpy)
├── layers/         # Layer primitives (depends on engine)
├── transformer/    # Attention + Transformer (depends on layers)
├── nlp/            # NLP domain (depends on transformer)
├── cv/             # CV domain (depends on layers)
├── video/          # Video domain (depends on cv)
├── 3d/             # 3D domain (depends on layers)
├── generative/     # Generative (depends on layers + cv)
├── multimodal/     # Multimodal (depends on nlp + cv)
├── platform/       # Visual platform (depends on all above)
├── papers/         # Annotated paper notes
├── notebooks/      # Experiments and ablations
└── _lab/           # Spike code — never imported by production paths
```

---

## Tasks

- [ ] Write `docs/architecture.md` — describe all layers (engine, library, platform), data flow, and deployment
- [ ] Draw domain model for `Graph`, `Node`, `Edge`, `Run`, `RunStep`, `LayerRegistry`
- [ ] Define `docs/schemas/graph.json` — valid JSON Schema for the pipeline graph format
- [ ] Define `docs/schemas/node_schema.json` — schema for a single layer node definition
- [ ] Design layer registry interface — write `platform/backend/registry.py` with 3 placeholder entries
- [ ] Write one-page design doc: how graph JSON compiles to a Python forward pass
- [ ] Create repo structure with all top-level directories and `__init__.py` files
- [ ] Set up `Makefile` with `test`, `lint`, `run-platform` targets
- [ ] Set up `docker-compose.yml` with PostgreSQL and Redis services
- [ ] Create `.github/workflows/ci.yml` — runs pytest on push
- [ ] Write `CONVENTIONS.md` — naming, file layout, grad-check requirement, paper-per-sprint rule

---

## Definition of Done

- [ ] Architecture document exists with diagram showing engine → layers → domains → platform
- [ ] Domain model diagram covers all entities with relationships
- [ ] Graph JSON schema is valid JSON Schema — test it against two example graphs
- [ ] Layer registry has a working `register()` and `get()` function with at least 3 entries
- [ ] Repo structure exists with correct folder hierarchy
- [ ] Makefile `test` target runs and exits 0 (even if no tests yet)
- [ ] You can explain from memory how a drag-and-drop graph becomes a running training loop
