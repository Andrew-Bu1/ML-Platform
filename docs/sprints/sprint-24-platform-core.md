# Sprint 24 — Platform Core

**Theme:** Node schema, layer registry, graph execution engine, shape inference
**Duration:** 4 weeks
**Phase:** Phase 8 — Platform

---

## Goal

Build the execution backbone of the visual platform. By end of sprint you can represent any ML architecture as a graph JSON, validate it, run it, and stream training metrics — all without a UI yet. The backend is complete and tested.

---

## Concepts

### Node Schema

Every layer type in the platform needs a machine-readable description:

```json
{
  "type": "Conv2D",
  "display_name": "Conv 2D",
  "domain": "cv",
  "params": [
    {"name": "in_channels",  "type": "int",   "required": true},
    {"name": "out_channels", "type": "int",   "required": true},
    {"name": "kernel_size",  "type": "int",   "default": 3},
    {"name": "stride",       "type": "int",   "default": 1},
    {"name": "padding",      "type": "int",   "default": 0}
  ],
  "output_shape": "lambda in_shape, p: (p['out_channels'], (in_shape[1]+2*p['padding']-p['kernel_size'])//p['stride']+1, ...)"
}
```

### Layer Registry

- Maps `layer_type` string → `(LayerClass, NodeSchema)`.
- `register(type_name, layer_class, schema)` — adds entry.
- `get(type_name)` — returns entry or raises `UnknownLayerError`.
- `list_by_domain(domain)` — returns all node schemas for NLP / CV / Video / 3D / Gen.

### Graph Execution Engine

1. **Parse** graph JSON → list of `Node` objects and `Edge` objects.
2. **Validate** — check for cycles, unknown layer types, missing required params.
3. **Shape inference** — propagate shapes topologically; raise `ShapeMismatchError` if incompatible.
4. **Instantiate** — create layer objects from registry using node configs.
5. **Execute** — call `forward()` in topological order, threading tensors through edges.
6. **Train loop** — wrap in optimizer + loss + epoch loop; yield metrics per step.

### Shape Inference Engine

- Each node schema has an `output_shape` formula — a function of `(input_shape, params)`.
- Traverse nodes in topological order, compute output shapes, store in `node.output_shape`.
- If a node's expected input shape doesn't match the upstream node's output shape: `ShapeMismatchError`.
- Return the full shape map before any weights are allocated — fail fast.

### SSE Streaming Metrics

- Server-Sent Events: HTTP streaming where server pushes events.
- FastAPI: use `StreamingResponse` with `text/event-stream` content type.
- Each training step: emit `data: {"step": 100, "loss": 0.42, "acc": 0.87}\n\n`.
- Client subscribes with `EventSource` — receives events as they are emitted.

---

## Tasks

### Node Schema + Registry

- [ ] Define `node_schema.json` for all layers: NLP (8 types), CV (8 types), Video (4 types), 3D (4 types), Gen (5 types), shared (Linear, Activations, Norm, Dropout, Optim, Loss, Dataset)
- [ ] Implement `LayerRegistry` class with `register`, `get`, `list_by_domain`
- [ ] Pre-register all 35+ layer types from your library into the registry
- [ ] Write registry tests: unknown type raises error, all registered types have valid schemas
- [ ] **Ship:** `platform/backend/node_schema.json` + `platform/backend/registry.py`

### Graph Execution Engine

- [ ] Implement `GraphParser.parse(graph_json)` → `(nodes, edges)`
- [ ] Implement `GraphValidator.validate(nodes, edges)` — cycle detection, unknown types, missing params
- [ ] Implement `ShapeInferencer.infer(nodes, edges)` — topological traversal, shape propagation
- [ ] Implement `GraphExecutor.instantiate(nodes)` → dict of layer instances
- [ ] Implement `GraphExecutor.forward(x, nodes, edges)` — single forward pass through DAG
- [ ] Write executor tests: CNN graph (Conv→ReLU→Pool→Flatten→Linear), Transformer graph
- [ ] **Ship:** `platform/backend/executor.py` + `platform/backend/shape_inference.py`

### Training Loop Node

- [ ] Implement `TrainingLoop(model_graph, optimizer_config, loss_config, epochs, batch_size)`
- [ ] Yield `{"step": int, "loss": float, "val_loss": float, "metrics": dict}` per step
- [ ] Implement `DatasetNode` types: MNIST, CIFAR10, custom CSV, custom image folder
- [ ] Wire training loop through executor: `executor.train(graph, dataset, training_config)` → generator
- [ ] Test: build CNN graph in Python dict, run training, verify loss decreases
- [ ] **Ship:** `platform/backend/train_node.py` + `platform/backend/dataset_nodes.py`

### API Foundation

- [ ] `POST /graph/validate` — returns shape map or list of errors
- [ ] `POST /graph/run` — starts training, returns SSE stream of metrics
- [ ] `POST /graph/export` — returns generated Python code string
- [ ] Implement SSE streaming with FastAPI `StreamingResponse`
- [ ] Write API integration test: submit a CNN graph, verify SSE stream receives 10+ metric events
- [ ] **Ship:** `platform/backend/main.py` + `platform/backend/api/`

---

## Interview Topics

- **How does shape inference work?** Topological traversal — each node's output shape is computed from its params and input shape using the schema formula. Mismatches surface before weight allocation.
- **Why separate graph JSON from execution plan?** JSON is the user's intent — portable, versionable, editable. Execution plan is resolved — concrete layer instances, validated shapes, execution order. Compiling one to the other separates concerns.
- **Why SSE over WebSocket for training metrics?** SSE is unidirectional server→client — simpler for streaming logs. WebSocket is bidirectional — needed for interactive communication. Training metrics only flow one way.
- **How would you handle a training job that runs for hours?** Background task queue (Celery/Redis), job IDs, persistent status storage. Client polls `/jobs/{id}/status` or reconnects to SSE by job ID.

---

## Definition of Done

- [ ] Registry contains all 35+ layer types — `registry.list_by_domain("cv")` returns 8+ nodes
- [ ] Shape inference catches wrong input shapes before any forward pass
- [ ] Executor runs a full CNN training on MNIST and loss decreases
- [ ] SSE stream delivers metric events — verified with `curl` before building UI
- [ ] All platform backend tests pass with `pytest platform/backend/`
