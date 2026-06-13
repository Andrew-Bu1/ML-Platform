# Sprint 25 — Platform API + React Canvas

**Theme:** FastAPI backend polish + React Flow canvas + live loss chart + export
**Duration:** 4 weeks
**Phase:** Phase 8 — Platform

---

## Goal

Build the visual interface on top of Sprint 24's execution engine. By end of sprint a user can drag layers onto a canvas, wire them together, press Run, watch the loss curve update live, and export the architecture as clean Python code.

---

## System Design

```
React App
├── NodePalette    — sidebar listing all layer types grouped by domain
├── Canvas         — React Flow graph editor (drag, drop, connect)
├── NodeConfig     — right panel form for selected node's params
├── Toolbar        — Run / Stop / Validate / Export buttons
├── LossChart      — Chart.js live updating line chart
└── CodeExport     — syntax-highlighted Python code preview

FastAPI
├── GET  /registry           — returns all node schemas by domain
├── POST /graph/validate     — shape inference, returns errors or shape map
├── POST /graph/run          — SSE stream of training metrics
├── POST /graph/export       — returns generated Python string
└── GET  /graph/templates    — preset graphs (LeNet, ResNet, GPT-mini, etc.)
```

---

## Tasks

### Backend Polish

- [ ] `GET /registry` — returns full node schema list grouped by domain for palette rendering
- [ ] `GET /graph/templates` — returns 5 preset graphs: LeNet, ResNet18, MLP, CharGPT, VAE
- [ ] Add CORS middleware — allow React dev server origin
- [ ] Add request validation with Pydantic models for all endpoints
- [ ] Add error handling: return structured JSON errors `{"error": "ShapeMismatch", "node_id": "n3", "detail": "..."}`
- [ ] **Ship:** `platform/backend/api/registry.py` + `platform/backend/api/templates.py`

### Code Export (codegen.py)

- [ ] Traverse graph in topological order — generate Python imports for used layer types
- [ ] Generate class definition: `class MyModel(nn.Module): ...` with `__init__` and `forward`
- [ ] Generate training loop boilerplate: optimizer init, epoch loop, loss computation
- [ ] Produce clean, formatted, runnable Python — verify it executes without modification
- [ ] **Ship:** `platform/backend/codegen.py` + `tests/test_codegen.py`

### React Canvas

- [ ] Set up React app with React Flow, Chart.js, Tailwind
- [ ] Implement `NodePalette` — fetch `/registry`, render grouped list of draggable node cards
- [ ] Implement drag from palette → drop onto canvas → creates React Flow node
- [ ] Implement edge drawing — connect output handle of one node to input handle of another
- [ ] Implement node deletion (delete key), edge deletion (click edge → delete)
- [ ] Implement `NodeConfig` panel — clicking a node opens form generated from node schema params
- [ ] Form field types: number input, string input, select dropdown, boolean toggle
- [ ] Implement graph-to-JSON serialization: read React Flow state → produce platform graph JSON
- [ ] **Ship:** `platform/frontend/src/components/NodePalette.jsx` + `Canvas.jsx` + `NodeConfig.jsx`

### Validate + Run + Chart

- [ ] Implement `Toolbar` with: Validate button (calls `/graph/validate`, shows errors inline on nodes), Run button (opens SSE connection), Stop button (closes SSE)
- [ ] Show shape annotations on edges after validation: display tensor shape as edge label
- [ ] Show error state on invalid nodes: red border + tooltip with error message
- [ ] Implement `LossChart` — subscribes to SSE stream, appends data points to Chart.js line chart
- [ ] Chart displays: training loss (blue), validation loss (orange), accuracy (green) on dual y-axis
- [ ] Chart auto-scrolls to show last 100 steps, full history accessible via zoom
- [ ] **Ship:** `platform/frontend/src/components/Toolbar.jsx` + `LossChart.jsx`

### Export + Templates

- [ ] Implement Export button: calls `/graph/export`, displays result in `<CodePreview>` with syntax highlighting (use Prism.js)
- [ ] Add copy-to-clipboard button on code preview
- [ ] Implement template picker: "Start from template" dropdown → loads preset graph onto canvas
- [ ] Implement save/load: serialize current graph to JSON, download as file or load from file
- [ ] **Ship:** `platform/frontend/src/components/CodeExport.jsx` + `TemplateSelector.jsx`

---

## Interview Topics

- **How does React Flow represent graph state?** Nodes array (id, type, position, data) and edges array (id, source, target). State is managed in React — serialize to JSON by reading this state.
- **How does SSE differ from polling?** Polling: client requests every N seconds — high latency, wasteful. SSE: server pushes when data is available — low latency, efficient. One persistent HTTP connection.
- **How would you handle network reconnection for SSE?** `EventSource` auto-reconnects. Assign each training run a job ID — on reconnect, fetch metrics from that job ID from the last received step.
- **Why generate code rather than export a pickle?** Code is readable, editable, version-controllable. Pickle is tied to exact library versions. Generated code is the best documentation of the architecture.

---

## Definition of Done

- [ ] Full drag-drop-connect-run workflow works end to end for a CNN
- [ ] Shape labels appear on edges after validation
- [ ] LossChart updates in real time during training — tested with 100-step run
- [ ] Code export produces runnable Python verified by executing it in a subprocess
- [ ] Template loader populates canvas with correct graph for all 5 presets
- [ ] All frontend unit tests pass (React Testing Library)
- [ ] All backend API tests pass (pytest)
