# next3d — AI-Native 3D CAD Engine

A semantic 3D geometry system that lets AI agents create, modify, analyze, and export parametric 3D parts and assemblies through structured tool calls.

**99 tools** covering modeling, assembly, analysis, sheet metal, FEA, 2D drawings, and export — usable from Claude Code, Claude Desktop, or any MCP client.

```bash
pip install next3d
```

## Quick Start with Claude Code

### 1. Install

```bash
# Clone and install
git clone https://github.com/buvi-org/next3d.git
cd next3d
pip install -e .
```

Requires Python 3.10+ and [CadQuery](https://cadquery.readthedocs.io/) (installed automatically).

### 2. Connect via MCP

**Claude Code (CLI):**

```bash
claude mcp add next3d -- next3d serve
```

**Claude Desktop:**

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "next3d": {
      "command": "next3d",
      "args": ["serve"]
    }
  }
}
```

### 3. Start Designing

Once connected, just ask Claude:

> "Create a 100x60x20mm aluminum bracket with 4 mounting holes and export it as STEP and STL"

> "Design a shaft with bearing seats at both ends, 200mm long, 15mm diameter"

> "Build a stiffened chamber wall panel — 1 foot square, 3mm sheet with 50x50x3 RHS grid — and check if it can handle 0.1 MPa with less than 2mm deflection"

> "Create an electronics enclosure 80x60x40mm, shell it to 2mm walls, add mounting holes, check it against FDM 3D printing design rules, then export STL"

---

## What Claude Can Do

### Create Geometry (9 tools)
```
create_box, create_cylinder, create_sphere, create_extrusion,
create_revolve, create_sweep, create_loft, create_sketch → sketch_extrude/sketch_revolve
```

### Modify Parts (10 tools)
```
add_hole, add_counterbore_hole, add_pocket, add_circular_pocket,
add_boss, add_slot, add_fillet, add_chamfer, add_shell, add_draft
```

### 2D Sketches (8 tools)
```
create_sketch, sketch_add_line, sketch_add_arc, sketch_add_circle,
sketch_add_rect, sketch_add_constraint, sketch_extrude, sketch_revolve
```

### Multi-Body Assembly (10 tools)
```
create_named_body, set_active_body, list_bodies, delete_body,
place_body, add_mate_constraint, check_interference,
add_standard_part, export_assembly, get_bom
```

Standard parts library: ISO metric fasteners M3–M12 (hex bolts, nuts, washers, socket head cap screws).

### Query & Understand (5 tools)
```
get_summary, get_features, find_faces, query_geometry, measure_distance
```

### Design Intelligence (7 tools)
```
check_design_rules, set_parameter, get_parameters,
add_datum, add_tolerance, get_gdt, suggest_gdt
```

Design rules for 6 processes: CNC milling, injection molding, FDM, SLA, sheet metal, casting.
GD&T per ASME Y14.5 with auto-suggest.

### Interactive Sheet Metal (10 tools)
```
sheet_metal_define, sheet_metal_add_flat, sheet_metal_add_bend,
sheet_metal_list_segments, sheet_metal_modify_segment,
sheet_metal_remove_segment, sheet_metal_insert_segment,
sheet_metal_get_flat_pattern, sheet_metal_get_cost, sheet_metal_plan_bending
```

Define sheet metal parts interactively: add flat/bend segments, recompute flat patterns, estimate costs, and plan bending operations.

### Parametric (3 tools)
```
update_parameter, design_table, get_parametric_state
```

### Dimensions & Drawings (6 tools)
```
add_dimension, get_dimensions, auto_dimension,
export_drawing, export_section_drawing, export_dxf
```

2D engineering drawing generation with dimensions, section views, and DXF export.

### Structural Analysis (5 tools)
```
run_fea, run_fea_parametric,
add_load, add_boundary_condition, run_topology_optimization
```

Real FEA solver: plate bending + beam stiffeners, 6 materials, 11 RHS sizes, parametric studies.

### Interactivity (15 tools)
```
remove_datum, remove_tolerance, modify_tolerance,
list_loads, remove_load, modify_load,
list_boundary_conditions, remove_boundary_condition,
list_mates, remove_mate, remove_parameter,
remove_dimension, modify_dimension,
list_standard_parts, list_design_processes
```

List, remove, and modify operations for iterative design workflows.

### Export (7 tools)
```
export_step, export_stl, export_3mf, export_assembly,
export_script, render_png, undo
```

---

## Example Workflows

### Mounting Bracket

```
create_box(length=120, width=80, height=12)
add_hole(center_x=45, center_y=30, diameter=8)
add_hole(center_x=-45, center_y=30, diameter=8)
add_hole(center_x=45, center_y=-30, diameter=8)
add_hole(center_x=-45, center_y=-30, diameter=8)
add_fillet(radius=3, edge_selector="|Z")
check_design_rules(process="cnc_milling")
export_step(output_path="/tmp/bracket.step")
```

### Shelled Electronics Housing (3D Print Ready)

```
create_box(length=80, width=60, height=40)
add_shell(thickness=2.0, face_selector=">Z")
add_hole(center_x=25, center_y=20, diameter=4, face_selector="<Z")
add_hole(center_x=-25, center_y=-20, diameter=4, face_selector="<Z")
add_fillet(radius=3, edge_selector="|Z")
check_design_rules(process="fdm_3d_print")
export_stl(output_path="/tmp/housing.stl")
```

### Multi-Part Assembly with Fasteners

```
create_named_body(name="base_plate", shape_type="box", material="aluminum",
                  length=100, width=80, height=5)
set_active_body(name="base_plate")
add_hole(center_x=30, center_y=25, diameter=6.4)
add_hole(center_x=-30, center_y=25, diameter=6.4)

add_standard_part(name="bolt_1", part_type="hex_bolt", size="M6", length=20)
add_standard_part(name="bolt_2", part_type="hex_bolt", size="M6", length=20)
place_body(name="bolt_1", x=30, y=25, z=5)
place_body(name="bolt_2", x=-30, y=25, z=5)

get_bom()
export_assembly(output_path="/tmp/assembly.step")
```

### Structural FEA — Stiffened Panel

```
run_fea(
  plate_width=304.8, plate_height=304.8,
  plate_thickness=3.0,
  grid_spacing_x=152.4, grid_spacing_y=152.4,
  rhs_size="50x50x3",
  material="steel_mild",
  pressure_mpa=0.1,
  bc_type="fixed_edges",
  weld_type="full"
)
→ max_deflection, stress, safety_factor, pass/fail verdict
```

Compare configurations:

```
run_fea_parametric(
  base_config={plate_width: 300, plate_height: 300, plate_thickness: 3,
               pressure_mpa: 0.05, grid_spacing_x: 150, grid_spacing_y: 150,
               rhs_size: "25x25x2"},
  variations=[
    {rhs_size: "25x25x2", label: "Small RHS"},
    {rhs_size: "50x50x3", label: "Large RHS"},
    {weld_type: "spot", label: "Spot welds"},
  ]
)
```

### Sheet Metal Bracket

```
sheet_metal_define(thickness=1.5, material="steel_mild", k_factor=0.44)
sheet_metal_add_flat(length=100, width=50)
sheet_metal_add_bend(angle=90, radius=3)
sheet_metal_add_flat(length=30, width=50)
sheet_metal_get_flat_pattern()
→ developed length, bend allowance, flat dimensions
sheet_metal_get_cost(quantity=100)
→ material cost, bending cost, total per-unit cost
sheet_metal_plan_bending()
→ ordered bend sequence, tool selection, collision checks
```

### Revolved Shaft

```
create_revolve(
  points=[[5,0], [10,0], [10,15], [8,15], [8,40], [6,40], [6,50], [5,50]],
  angle_degrees=360
)
suggest_gdt()
export_step(output_path="/tmp/shaft.step")
```

---

## CLI Usage

```bash
# Inspect a STEP file
next3d inspect part.step

# Export semantic graph as JSON
next3d graph part.step --format json

# List recognized features
next3d features part.step

# Run a modeling pipeline
next3d build '[{"tool":"create_box","params":{"length":100,"width":60,"height":20}},
               {"tool":"add_hole","params":{"center_x":0,"center_y":0,"diameter":10}}]' \
  -o output.step

# Start the MCP server
next3d serve
```

---

## Architecture

```
AI Agent (Claude, GPT, etc.)
    ↓ tool calls
MCP Server (next3d serve)
    ↓
Tool Executor → validates params, dispatches
    ↓
Modeling Session → stateful, multi-body, undo
    ↓
Kernel (CadQuery / OpenCascade) → exact B-Rep geometry
    ↓
Semantic Graph → topology, features, relationships
    ↓
Export (STEP, STL, 3MF, PNG, CadQuery script)
```

## Tech Stack

99 tools across modeling, assembly, sheet metal, FEA, drawings, and export.

| Layer | Technology |
|-------|-----------|
| CAD Kernel | OpenCascade via CadQuery |
| Graph | NetworkX |
| Data Models | Pydantic v2 |
| FEA Solver | NumPy + SciPy sparse |
| CLI | Click + Rich |
| AI Protocol | MCP (Model Context Protocol) |

---

## Development

```bash
pip install -e ".[dev]"
pytest                     # run all tests (350)
pytest -k "FEA"           # run FEA tests only
pytest -k "TestSketch"    # run sketch tests only
```

## License

MIT
