# Next3D — Semantic 3D Geometry Cognition System

## Vision

A machine-understandable geometric cognition system that represents 3D CAD geometry
as structured, semantic, relational data — not triangles. Enables AI to reason about
engineering geometry the way humans do: topology, geometry, semantics, relationships.

---

## Architecture Overview

```
STEP File
   ↓
B-Rep Kernel (OpenCascade via PythonOCC)
   ↓
Topology Graph Builder
   ↓
Feature Recognition Engine (rule-based)
   ↓
Semantic Graph (3D DOM)
   ↓
Query + Reasoning Layer (DSL)
   ↓
AI Agent Interface (JSON / Graph embeddings)
```

---

## Technology Stack

| Layer              | Technology                        |
|--------------------|-----------------------------------|
| Language           | Python 3.11+                      |
| CAD Kernel         | OpenCascade via PythonOCC-core    |
| Graph              | NetworkX                          |
| Data Models        | Pydantic v2                       |
| Serialization      | JSON (LLM), MessagePack (fast)    |
| CLI                | Click                             |
| Testing            | pytest                            |
| Build              | pyproject.toml (hatchling)        |

---

## Phasing

### Phase 1 — MVP
- [x] Project structure and configuration
- [x] STEP file parsing via OpenCascade B-Rep kernel
- [x] Topology graph construction (Face/Edge/Vertex nodes, adjacency edges)
- [x] Persistent identity system (stable IDs across transforms)
- [x] Basic feature recognition (holes, fillets, chamfers, slots, bosses, counterbores)
- [x] Semantic graph assembly (3D DOM)
- [x] Query DSL for geometry selection
- [x] JSON export for LLM consumption
- [x] CLI tool (`next3d inspect part.step`)

### Phase 2 — Relationships & Reasoning
- [x] Constraint/relationship engine (parallel, concentric, tangent, offset)
- [x] Spatial reasoning (inside/outside, intersections, clearance)
- [x] Multi-scale representation (Solid → Feature → Face → Edge → Vertex)
- [x] Graph embedding export for GNN/ML models
- [x] Inter-feature relationships (symmetric, pattern_member)

### Phase 3 — Manufacturing & Assembly Intelligence
- [x] Manufacturing semantics (machinability, axis count, process suggestions)
- [x] Assembly mating logic and fit analysis
- [ ] Parametric/procedural history extraction (when available in STEP)
- [x] Physical properties (mass, CoG, moments of inertia)
- [x] Multi-body STEP support (proper solid-to-face mapping)

---

## Strategic Feature Roadmap — Phases 4–7

### Current State (Phases 1–3 complete)
27 tools, feature recognition, semantic graph, manufacturing analysis, physical properties,
STEP I/O. Foundation is solid but missing the operations that make up 80% of a mechanical
designer's daily work.

### Phase 4 — Advanced Modeling Operations (highest ROI)
Operations designers use on almost every part. Unlocks 80% of part geometry.

- [ ] **Revolve** — Rotate a 2D profile around an axis to create solids of revolution
      (~30% of parts: shafts, pulleys, bushings, nozzles)
- [ ] **Sweep** — Extrude a 2D profile along a path (pipes, channels, gasket grooves)
- [ ] **Loft** — Transition between different cross-sections (ducts, bottles, aero shapes)
- [ ] **Shell** — Hollow out a solid to uniform wall thickness (enclosures, housings)
- [ ] **Draft** — Add taper angles to faces (required for injection molding and casting)
- [ ] **STL export** — Tessellated mesh output for 3D printing
- [ ] **3MF export** — Modern 3D printing format with color/material support
- [ ] **Visual feedback** — Render model to PNG so AI can self-correct

### Phase 5 — Multi-Body & Assembly
Single-body modeling can't build real products.

- [ ] Multi-body sessions (work with multiple solids, split/combine bodies)
- [ ] Assembly context (place parts relative to each other with mate constraints)
- [ ] Interference detection (check if parts collide)
- [ ] Bill of Materials (part list with quantities, materials, mass)
- [ ] Standard parts library (ISO metric fasteners, bearings, O-rings)
- [ ] Parametric named dimensions (design intent propagation)

### Phase 6 — Design Intelligence
Where AI surpasses human designers.

- [ ] 2D sketch + constraints (lines, arcs, circles, splines with geometric constraints)
- [ ] Design rules engine (min wall thickness, max overhang, hole spacing rules)
- [ ] DFM auto-check (draft angles for molding, min radii for CNC, undercut detection)
- [ ] GD&T annotation (geometric dimensioning and tolerancing)
- [ ] Topology optimization (given loads/constraints, optimal material distribution)
- [ ] Visual feedback loop (render → AI "sees" → self-corrects)

### Phase 7 — Domain Workflows & Simulation
Specialized features for common design domains.

- [ ] Sheet metal (bend, unfold, flat pattern, K-factor)
- [ ] Weldments/structural (extrude profiles along paths — frames, trusses)
- [ ] Enclosure generator (PCB outline + components → snap-fit enclosure)
- [ ] Gear/cam generation (parametric involute gears, cams, sprockets)
- [ ] FEA linear static (apply loads → check stress/deflection)
- [ ] Motion simulation (verify mechanisms, collision detection)
- [ ] Drawing generation (2D engineering drawings with dimensions, sections)
- [ ] IGES export (legacy CAD interoperability)

### Recommended Build Order
1. **Revolve, Sweep, Loft, Shell, Draft** — unlock 80% of part geometry
2. **STL/3MF export** — enable 3D printing workflow end-to-end
3. **Visual feedback (PNG render)** — let AI self-correct
4. **Multi-body + basic assembly mates**
5. **Standard parts library** (start with ISO metric fasteners)
6. **Parametric named dimensions**
7. **Design rules engine + DFM auto-check**
8. **2D constrained sketches**
9. **Sheet metal, enclosure generator, gear generator**

---

## Core Design Principles

1. **B-Rep is truth** — All reasoning operates on exact geometry (NURBS, analytic surfaces), never on mesh approximations.
2. **Dual representation** — Exact geometry for computation, tessellated mesh only for visualization.
3. **Topology-first** — The topology graph (face adjacency, edge connectivity) is the primary data structure, analogous to DOM.
4. **Persistent identity** — Every topological entity gets a stable hash-based ID that survives transformations.
5. **Context makes semantics** — A cylinder is not a hole. A cylinder + topology + context = hole. Feature recognition is graph pattern matching.
6. **Query-driven** — Geometry is accessed via a declarative DSL, not imperative traversal.
7. **AI-native output** — The system produces structured data optimized for LLM reasoning and ML model consumption.

---

## Requirements

### R1: STEP File Ingestion
- **R1.1**: Parse STEP AP203/AP214 files using OpenCascade
- **R1.2**: Extract complete B-Rep structure (Solids → Shells → Faces → Wires → Edges → Vertices)
- **R1.3**: Support multi-body STEP files (assemblies with multiple solids)
- **R1.4**: Report parse errors with meaningful diagnostics

### R2: Topology Graph
- **R2.1**: Build a directed graph from B-Rep topology
- **R2.2**: Node types: `Solid`, `Shell`, `Face`, `Wire`, `Edge`, `Vertex`
- **R2.3**: Edge types: `contains` (parent→child), `adjacent` (face↔face sharing an edge), `shares_vertex`
- **R2.4**: Each node carries geometric metadata:
  - Face: surface type (plane, cylinder, cone, sphere, torus, bspline), parameters (normal, radius, axis, etc.), area, UV bounds
  - Edge: curve type (line, circle, ellipse, bspline), length, endpoints
  - Vertex: 3D coordinates (x, y, z)
- **R2.5**: Graph must be serializable to JSON and reconstructable

### R3: Persistent Identity
- **R3.1**: Assign deterministic IDs based on geometric + topological hashing
- **R3.2**: IDs must be stable: same geometry → same ID regardless of STEP file ordering
- **R3.3**: ID format: `{entity_type}_{hash}` (e.g., `face_a3f2b1`, `edge_91cc04`)
- **R3.4**: Provide a mapping table from OpenCascade internal labels to persistent IDs

### R4: Feature Recognition (Rule-Based)
- **R4.1**: **Through Holes** — Cylindrical face, two circular edges, both edges shared with other faces, axis passes through solid
- **R4.2**: **Blind Holes** — Cylindrical face + planar bottom face, one open circular edge
- **R4.3**: **Fillets** — Toroidal or cylindrical face tangent to two adjacent faces, constant or variable radius
- **R4.4**: **Chamfers** — Planar face at angle between two other faces, narrow strip geometry
- **R4.5**: **Counterbores / Countersinks** — Concentric hole sequences with step or conical transitions
- **R4.6**: **Slots** — U-shaped profile: planar bottom + two planar sides + two cylindrical ends
- **R4.7**: **Bosses** — Cylindrical or prismatic protrusion from a planar face
- **R4.8**: Each feature stores: type, constituent faces/edges, parameters (diameter, depth, radius, angle), axis/orientation
- **R4.9**: Features reference their constituent topological entities by persistent ID

### R5: Semantic Graph (3D DOM)
- **R5.1**: Layered graph combining topology and features
- **R5.2**: Feature nodes link to their constituent Face/Edge/Vertex nodes
- **R5.3**: Inter-feature relationships: `concentric`, `coaxial`, `coplanar`, `symmetric`, `pattern_member`
- **R5.4**: Hierarchy: Solid → Features → Faces → Edges → Vertices
- **R5.5**: Full graph exportable as JSON document for LLM context injection

### R6: Query DSL
- **R6.1**: Select entities by type: `faces(type="cylinder")`
- **R6.2**: Filter by geometric properties: `faces(type="cylinder", radius=5.0, depth__gt=10.0)`
- **R6.3**: Filter by relationships: `edges(adjacent_to=face_id)`
- **R6.4**: Chain queries: `faces(type="cylinder").adjacent(type="plane")`
- **R6.5**: Feature-level queries: `features(type="hole", diameter__lt=12.0)`
- **R6.6**: Return results as list of entity descriptors with full metadata

### R7: AI Interface
- **R7.1**: Export semantic graph as structured JSON with consistent schema
- **R7.2**: Provide summary mode (features + key dimensions only) and detail mode (full topology)
- **R7.3**: JSON output must be self-contained — an LLM reading it should understand the part without seeing the geometry
- **R7.4**: Include natural-language annotations: `"description": "Through hole, diameter 10mm, along Z axis"`
- **R7.5**: Support streaming partial results for large models

### R8: CLI
- **R8.1**: `next3d inspect <file.step>` — Print semantic summary
- **R8.2**: `next3d graph <file.step>` — Export full semantic graph as JSON
- **R8.3**: `next3d query <file.step> "<query>"` — Run DSL query, print results
- **R8.4**: `next3d features <file.step>` — List recognized features with parameters
- **R8.5**: `next3d validate <file.step>` — Check file integrity and report issues
- **R8.6**: Output formats: `--format json|table|yaml`

---

## Data Schema (Core Types)

### Entity Base
```
{id, type, geometry_type, parameters, persistent_id}
```

### Face
```
{persistent_id, surface_type, normal, area, centroid, radius?, axis?, uv_bounds}
```

### Edge
```
{persistent_id, curve_type, length, start_vertex, end_vertex, radius?, center?}
```

### Vertex
```
{persistent_id, x, y, z}
```

### Feature
```
{persistent_id, feature_type, faces[], edges[], parameters{}, axis, description}
```

### SemanticGraph
```
{solids[], features[], faces[], edges[], vertices[], adjacency[], relationships[]}
```

---

## Directory Structure

```
next3d/
├── pyproject.toml
├── PLANNING.md
├── src/
│   └── next3d/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── brep.py           # STEP → B-Rep parsing
│       │   ├── topology.py       # Topology graph builder
│       │   ├── identity.py       # Persistent ID system
│       │   └── schema.py         # Pydantic data models
│       ├── features/
│       │   ├── __init__.py
│       │   ├── engine.py         # Feature recognition orchestrator
│       │   ├── holes.py          # Hole detection (through, blind, counterbore)
│       │   ├── fillets.py        # Fillet detection
│       │   ├── chamfers.py       # Chamfer detection
│       │   ├── slots.py          # Slot detection
│       │   └── bosses.py         # Boss/protrusion detection
│       ├── graph/
│       │   ├── __init__.py
│       │   ├── semantic.py       # Semantic graph assembly (3D DOM)
│       │   └── query.py          # Query DSL parser + executor
│       ├── ai/
│       │   ├── __init__.py
│       │   └── interface.py      # JSON/embedding export for AI
│       └── cli/
│           ├── __init__.py
│           └── main.py           # Click CLI entry point
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_brep.py
│   ├── test_topology.py
│   ├── test_identity.py
│   ├── test_features.py
│   ├── test_query.py
│   └── test_cli.py
└── examples/
    └── .gitkeep
```

---

## Non-Functional Requirements

- **Performance**: Process a 10k-face STEP file in < 30 seconds
- **Memory**: Topology graph for 10k faces < 500MB RAM
- **Determinism**: Same input → identical output (persistent IDs, graph structure)
- **Extensibility**: New feature recognizers plug in via a registry pattern
- **Error handling**: Never crash on malformed STEP — degrade gracefully with diagnostics
