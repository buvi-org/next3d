# next3d — AI Agent Guide

> Complete reference for AI agents using next3d to create, modify, query,
> and export parametric 3D geometry.

## Quick Start

### Connect via MCP (Claude Code)

```bash
claude mcp add next3d -- next3d serve
```

### Connect via MCP (Claude Desktop)

Add to `claude_desktop_config.json`:
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

### Your First Part

```
1. create_box(length=100, width=60, height=20)
2. add_hole(center_x=25, center_y=15, diameter=6)
3. add_hole(center_x=-25, center_y=-15, diameter=6)
4. add_fillet(radius=2, edge_selector="|Z")
5. export_step(output_path="/tmp/bracket.step")
```

---

## Core Concepts

### Stateful Session

The server maintains a single modeling session. Each tool call modifies or
queries the same shape. There is no need to pass shape references between
calls — the session tracks the current state.

```
create_box → [box exists] → add_hole → [box with hole] → add_fillet → [filleted box with hole]
```

### Units

All dimensions are in **millimeters (mm)** and **degrees** for angles.

### Coordinate System

- **X** = length (left/right)
- **Y** = width (front/back)
- **Z** = height (up/down)
- Origin (0,0,0) is at the center of created shapes by default.

---

## Tools Reference

### CREATE — Build Base Shapes

#### `create_box`

Create a rectangular box centered at a point.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `length` | float | *required* | X dimension in mm |
| `width` | float | *required* | Y dimension in mm |
| `height` | float | *required* | Z dimension in mm |
| `center_x` | float | 0 | X center |
| `center_y` | float | 0 | Y center |
| `center_z` | float | 0 | Z center |

**Returns:** `faces: 6, edges: 12, vertices: 8`

**Example:** `create_box(length=100, width=60, height=20)`

Creates a 100x60x20mm box centered at the origin. The top face is at Z=10,
bottom at Z=-10.

---

#### `create_cylinder`

Create a cylinder along an axis.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `radius` | float | *required* | Radius in mm |
| `height` | float | *required* | Height in mm |
| `center_x` | float | 0 | X center of base |
| `center_y` | float | 0 | Y center of base |
| `center_z` | float | 0 | Z center of base |
| `axis` | string | "Z" | Axis direction: "X", "Y", or "Z" |

**Example:** `create_cylinder(radius=25, height=50)`

Creates a vertical cylinder (radius 25, height 50) centered at origin.

---

#### `create_sphere`

Create a sphere.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `radius` | float | *required* | Radius in mm |
| `center_x` | float | 0 | X center |
| `center_y` | float | 0 | Y center |
| `center_z` | float | 0 | Z center |

---

#### `create_extrusion`

Extrude a 2D polygon along Z to create a solid.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `points` | list[list[float]] | *required* | 2D vertices [[x,y], ...]. Min 3. Auto-closed. |
| `height` | float | *required* | Extrusion height in mm |
| `center_x` | float | 0 | X offset for sketch plane |
| `center_y` | float | 0 | Y offset |
| `center_z` | float | 0 | Z offset |

**Example — L-bracket profile:**
```json
{
  "points": [[0,0], [60,0], [60,10], [10,10], [10,40], [0,40]],
  "height": 20
}
```

---

### MODIFY — Cut, Add, and Blend

All modify tools operate on the **current shape**. They use **face selectors**
to pick which face to work on.

#### Face Selectors

| Selector | Meaning | Example Use |
|----------|---------|-------------|
| `>Z` | **Top** face (highest Z) | Drill from above |
| `<Z` | **Bottom** face (lowest Z) | Add feet |
| `>X` | **Right** face (highest X) | Side mounting hole |
| `<X` | **Left** face (lowest X) | Side pocket |
| `>Y` | **Front** face (highest Y) | Front slot |
| `<Y` | **Back** face (lowest Y) | Back connector |

#### Edge Selectors

| Selector | Meaning | Example Use |
|----------|---------|-------------|
| `\|Z` | **Vertical** edges (parallel to Z) | Fillet corners of a box |
| `\|X` | Edges **along X** | Fillet top-front edge |
| `\|Y` | Edges **along Y** | Fillet top-side edge |
| `>Z` | **Top** edges (highest Z) | Chamfer top perimeter |
| `<Z` | **Bottom** edges | Chamfer bottom |
| `#Z` | Edges **perpendicular to Z** (horizontal) | Fillet horizontal edges |
| `null` | **All** edges | Round everything |

---

#### `add_hole`

Drill a cylindrical hole into a face.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `center_x` | float | *required* | X position on face |
| `center_y` | float | *required* | Y position on face |
| `diameter` | float | *required* | Hole diameter in mm |
| `depth` | float\|null | null | Depth. `null` = through-all |
| `face_selector` | string | ">Z" | Which face to drill |

**Example — through hole in top face:**
```json
{"center_x": 20, "center_y": 10, "diameter": 6}
```

**Example — blind hole 8mm deep from bottom:**
```json
{"center_x": 0, "center_y": 0, "diameter": 4, "depth": 8, "face_selector": "<Z"}
```

---

#### `add_counterbore_hole`

Stepped hole — narrow through-hole with a wider recess for bolt heads.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `center_x` | float | *required* | X position |
| `center_y` | float | *required* | Y position |
| `hole_diameter` | float | *required* | Through-hole diameter |
| `cb_diameter` | float | *required* | Counterbore diameter (> hole_diameter) |
| `cb_depth` | float | *required* | Counterbore depth |
| `depth` | float\|null | null | Through-hole depth. null = through-all |
| `face_selector` | string | ">Z" | Target face |

**Example — M6 bolt hole:**
```json
{
  "center_x": 25, "center_y": 15,
  "hole_diameter": 6.5, "cb_diameter": 11, "cb_depth": 6
}
```

---

#### `add_pocket`

Cut a rectangular pocket into a face.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `center_x` | float | *required* | Pocket center X |
| `center_y` | float | *required* | Pocket center Y |
| `length` | float | *required* | X dimension |
| `width` | float | *required* | Y dimension |
| `depth` | float | *required* | Cut depth |
| `face_selector` | string | ">Z" | Target face |

**Example:** `add_pocket(center_x=0, center_y=0, length=40, width=20, depth=5)`

---

#### `add_circular_pocket`

Cut a circular pocket into a face.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `center_x` | float | *required* | Center X |
| `center_y` | float | *required* | Center Y |
| `diameter` | float | *required* | Pocket diameter |
| `depth` | float | *required* | Cut depth |
| `face_selector` | string | ">Z" | Target face |

---

#### `add_boss`

Add a cylindrical protrusion (boss) on a face.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `center_x` | float | *required* | Center X |
| `center_y` | float | *required* | Center Y |
| `diameter` | float | *required* | Boss diameter |
| `height` | float | *required* | Boss height (extrusion up from face) |
| `face_selector` | string | ">Z" | Target face |

---

#### `add_slot`

Cut a slot (rounded-end rectangle) into a face.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `center_x` | float | *required* | Center X |
| `center_y` | float | *required* | Center Y |
| `length` | float | *required* | Slot length |
| `width` | float | *required* | Slot width |
| `depth` | float | *required* | Cut depth |
| `angle` | float | 0 | Rotation angle in degrees |
| `face_selector` | string | ">Z" | Target face |

**Example — angled slot:** `add_slot(center_x=0, center_y=0, length=30, width=8, depth=3, angle=45)`

---

#### `add_fillet`

Round edges with a fillet.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `radius` | float | *required* | Fillet radius in mm |
| `edge_selector` | string\|null | null | Which edges. null = all edges |

**Example — round vertical corners:** `add_fillet(radius=3, edge_selector="|Z")`

---

#### `add_chamfer`

Bevel edges with a chamfer.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `distance` | float | *required* | Chamfer distance in mm |
| `edge_selector` | string\|null | null | Which edges. null = all |

**Example — bevel top perimeter:** `add_chamfer(distance=1.5, edge_selector=">Z")`

---

### BOOLEAN — Combine Shapes

#### `boolean_cut`

Subtract a primitive from the current shape.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tool_type` | string | *required* | "box", "cylinder", or "sphere" |
| `tool_params` | object | *required* | Same params as the create tool for that type |

**Example — cut a notch:**
```json
{
  "tool_type": "box",
  "tool_params": {"length": 20, "width": 20, "height": 30, "center_x": 40, "center_y": 0, "center_z": 0}
}
```

**Example — cut a cylindrical bore:**
```json
{
  "tool_type": "cylinder",
  "tool_params": {"radius": 15, "height": 100, "center_x": 0, "center_y": 0, "center_z": -10}
}
```

---

### TRANSFORM — Move and Rotate

#### `translate`

Move the shape.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dx` | float | 0 | X displacement |
| `dy` | float | 0 | Y displacement |
| `dz` | float | 0 | Z displacement |

---

#### `rotate`

Rotate the shape around an axis.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `angle_degrees` | float | *required* | Rotation angle |
| `axis_x` | float | 0 | X component of axis |
| `axis_y` | float | 0 | Y component of axis |
| `axis_z` | float | 1 | Z component of axis |
| `center_x` | float | 0 | X rotation center |
| `center_y` | float | 0 | Y rotation center |
| `center_z` | float | 0 | Z rotation center |

**Example — rotate 45 deg around Z:** `rotate(angle_degrees=45)`

**Example — rotate 90 deg around X:** `rotate(angle_degrees=90, axis_x=1, axis_y=0, axis_z=0)`

---

### QUERY — Understand the Geometry

These tools inspect the current shape without modifying it. **Always query
before modifying** to understand what you're working with.

#### `get_summary`

Returns overall part statistics. No parameters.

**Example response:**
```json
{
  "operations": 3,
  "faces": 10,
  "edges": 24,
  "vertices": 16,
  "features": 2,
  "solids": 1,
  "bounding_box": {"x_min": -50, "x_max": 50, "y_min": -30, "y_max": 30, "z_min": -10, "z_max": 10}
}
```

---

#### `get_features`

List recognized manufacturing features.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature_type` | string\|null | null | Filter: `through_hole`, `blind_hole`, `fillet`, `chamfer`, `slot`, `boss`, `counterbore`, `countersink`. null = all |

**Example response (through holes):**
```json
{
  "count": 4,
  "features": [
    {
      "id": "feature_2a0caaef654da8f8",
      "type": "through_hole",
      "parameters": {"diameter": 6.0, "radius": 3.0},
      "axis": {"x": 0, "y": 0, "z": 1},
      "face_ids": ["face_abc123"],
      "description": "Through hole ⌀6.0"
    }
  ]
}
```

---

#### `find_faces`

Find faces by geometric properties.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `surface_type` | string\|null | null | `plane`, `cylinder`, `cone`, `sphere`, `torus`, `bspline` |
| `min_radius` | float\|null | null | Min radius (curved faces) |
| `max_radius` | float\|null | null | Max radius |
| `normal_x` | float\|null | null | Face normal X (planes only) |
| `normal_y` | float\|null | null | Face normal Y |
| `normal_z` | float\|null | null | Face normal Z |

**Common queries:**

| Goal | Parameters |
|------|-----------|
| Top face | `surface_type="plane", normal_z=1.0` |
| Bottom face | `surface_type="plane", normal_z=-1.0` |
| Right face | `surface_type="plane", normal_x=1.0` |
| All cylindrical faces | `surface_type="cylinder"` |
| Large holes (radius > 5) | `surface_type="cylinder", min_radius=5` |
| Small holes (radius < 3) | `surface_type="cylinder", max_radius=3` |

---

#### `query_geometry`

Advanced DSL queries on the semantic graph.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | *required* | DSL query string |

**DSL syntax:** `entity_type(key=value, key__operator=value)`

**Entity types:** `faces`, `edges`, `vertices`, `features`

**Operators:**
| Operator | Meaning | Example |
|----------|---------|---------|
| `=` | Equals | `surface_type="plane"` |
| `__gt` | Greater than | `radius__gt=5` |
| `__gte` | Greater or equal | `area__gte=100` |
| `__lt` | Less than | `radius__lt=3` |
| `__lte` | Less or equal | `diameter__lte=10` |
| `__ne` | Not equal | `surface_type__ne="plane"` |

**Example queries:**
```
faces(surface_type="cylinder")
faces(surface_type="plane", area__gt=500)
features(feature_type="through_hole")
features(feature_type="through_hole", diameter__gt=8)
edges(curve_type="circle")
edges(curve_type="circle", radius__lt=5)
```

---

### SESSION — Load, Export, Undo

#### `load_step`

Load geometry from a STEP file. Replaces the current shape.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | string | *required* | Absolute path to .step/.stp file |

---

#### `export_step`

Export the current geometry to a STEP file.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_path` | string | *required* | Output file path |

---

#### `export_script`

Export the operation history as a standalone CadQuery Python script. No
parameters. Returns the full script as a string — useful for reproducibility,
version control, or handing off to a human engineer.

---

#### `undo`

Revert the last modeling operation. Can be called multiple times to step
back through history.

---

## Workflow Patterns

### Pattern 1: Mounting Bracket

```
create_box(length=120, width=80, height=12)
  → 6 faces, 12 edges

add_hole(center_x=45, center_y=30, diameter=8)
add_hole(center_x=-45, center_y=30, diameter=8)
add_hole(center_x=45, center_y=-30, diameter=8)
add_hole(center_x=-45, center_y=-30, diameter=8)
  → 4 through-holes for M8 bolts

add_pocket(center_x=0, center_y=0, length=60, width=40, depth=6)
  → central lightening pocket

add_fillet(radius=3, edge_selector="|Z")
  → rounded vertical corners

add_chamfer(distance=0.5, edge_selector=">Z")
  → beveled top edges

get_features()
  → verify: 4 through_hole, 1 fillet

export_step(output_path="/tmp/bracket.step")
export_script()
  → CadQuery Python code for reproduction
```

### Pattern 2: Analyze Existing Part

```
load_step(path="/data/housing.step")

get_summary()
  → faces: 42, features: 8, solids: 1

get_features()
  → 4 through_hole, 2 blind_hole, 2 fillet

find_faces(surface_type="cylinder", min_radius=3)
  → find all large cylindrical surfaces

query_geometry(query='features(feature_type="through_hole", diameter__gt=6)')
  → find large mounting holes

find_faces(surface_type="plane", normal_z=1.0)
  → identify top mounting surface
```

### Pattern 3: Iterative Design

```
create_box(length=100, width=60, height=20)

get_summary()
  → understand what we have

add_hole(center_x=0, center_y=0, diameter=10)

get_features(feature_type="through_hole")
  → verify hole was recognized

undo()
  → oops, wrong diameter

add_hole(center_x=0, center_y=0, diameter=8)
  → correct diameter

get_features(feature_type="through_hole")
  → confirm: 1 through_hole, diameter=8

export_step(output_path="/tmp/result.step")
```

### Pattern 4: Complex Enclosure

```
create_box(length=150, width=100, height=60)

add_pocket(center_x=0, center_y=0, length=140, width=90, depth=55)
  → hollow out interior (5mm walls)

add_counterbore_hole(center_x=65, center_y=40, hole_diameter=4, cb_diameter=8, cb_depth=3)
add_counterbore_hole(center_x=-65, center_y=40, hole_diameter=4, cb_diameter=8, cb_depth=3)
add_counterbore_hole(center_x=65, center_y=-40, hole_diameter=4, cb_diameter=8, cb_depth=3)
add_counterbore_hole(center_x=-65, center_y=-40, hole_diameter=4, cb_diameter=8, cb_depth=3)
  → 4 corner screw holes with head recesses

add_slot(center_x=0, center_y=45, length=40, width=12, depth=55, face_selector=">Y")
  → ventilation slot on front face

add_fillet(radius=5, edge_selector="|Z")
  → rounded corners

export_step(output_path="/tmp/enclosure.step")
```

---

## Best Practices for AI Agents

### 1. Query First, Modify Second

Always call `get_summary` or `find_faces` before modifying. This gives you
the face count, feature count, and bounding box so you can plan your
modifications correctly.

### 2. Verify After Each Operation

After adding a feature, call `get_features` to confirm it was recognized
correctly. If something went wrong, use `undo` immediately.

### 3. Use Appropriate Selectors

- Default face selector `>Z` works for most top-down operations.
- For side operations, use `>X`, `<X`, `>Y`, `<Y`.
- For edge operations: `|Z` for vertical edges, `>Z`/`<Z` for top/bottom
  edge loops.

### 4. Hole Positioning

Hole coordinates (`center_x`, `center_y`) are relative to the **selected
face's coordinate system**. For the top face (`>Z`), this is the XY plane.
For a side face (`>X`), coordinates map to the face's local Y and Z.

### 5. Depth vs Through-All

- Set `depth=null` (default) for through-holes — this cuts through the
  entire part regardless of thickness.
- Set explicit depth for blind holes, pockets, and counterbore recesses.

### 6. Fillet/Chamfer Ordering

Apply fillets and chamfers **last**, after all holes, pockets, and bosses.
Filleting early can make face selection unpredictable for later operations.

### 7. Export Both Formats

Always export both STEP (for manufacturing) and script (for reproducibility):
```
export_step(output_path="/tmp/part.step")
export_script()  → returns Python code
```

---

## Data Model Reference

### Surface Types

| Type | Description | Has Normal | Has Radius | Has Axis |
|------|-------------|:---:|:---:|:---:|
| `plane` | Flat face | Yes | No | No |
| `cylinder` | Cylindrical surface | No | Yes | Yes |
| `cone` | Conical surface | No | Yes | Yes |
| `sphere` | Spherical surface | No | Yes | No |
| `torus` | Donut-like surface | No | Yes | Yes |
| `bspline` | Freeform surface | No | No | No |

### Feature Types

| Type | Description | Key Parameters |
|------|-------------|---------------|
| `through_hole` | Hole through entire part | `diameter`, `radius` |
| `blind_hole` | Hole with bottom | `diameter`, `radius`, `depth` |
| `counterbore` | Stepped bolt hole | `hole_diameter`, `cb_diameter`, `cb_depth` |
| `countersink` | Conical screw hole | `diameter`, `angle` |
| `fillet` | Rounded edge blend | `radius` |
| `chamfer` | Beveled edge | `distance` |
| `slot` | Elongated cut | `length`, `width`, `depth` |
| `boss` | Cylindrical protrusion | `diameter`, `height` |

### Geometric Relationships (in semantic graph)

| Relationship | Description |
|-------------|-------------|
| `ADJACENT` | Faces share an edge |
| `TANGENT` | Surfaces touch smoothly |
| `CONCENTRIC` | Same center point |
| `COAXIAL` | Same axis |
| `COPLANAR` | In the same plane |
| `PARALLEL` | Parallel surfaces |
| `PERPENDICULAR` | Right-angle surfaces |

---

## Error Handling

Tool calls return structured results:
```json
{
  "success": true,
  "message": "Created box 100.0x60.0x20.0",
  "data": {"op_id": "abc123", "faces": 6, "edges": 12}
}
```

On failure:
```json
{
  "success": false,
  "message": "No face matches selector '>Z' — shape may not have a top face"
}
```

**Common errors and fixes:**

| Error | Cause | Fix |
|-------|-------|-----|
| "No shape in session" | Called modify before create | Call `create_*` or `load_step` first |
| "No face matches selector" | Wrong face selector | Try different selector, call `find_faces` |
| "Hole extends outside shape" | Hole center outside face | Check coordinates with `get_summary` bounding box |
| "Fillet radius too large" | Radius exceeds edge length | Reduce radius |
| "Nothing to undo" | No operations in history | Nothing to fix |

---

## Alternative Integration Methods

### CLI Pipeline (non-MCP)

```bash
next3d build '[
  {"tool": "create_box", "params": {"length": 100, "width": 60, "height": 20}},
  {"tool": "add_hole", "params": {"center_x": 0, "center_y": 0, "diameter": 10}},
  {"tool": "add_fillet", "params": {"radius": 2, "edge_selector": "|Z"}}
]' -o /tmp/part.step
```

### Python API (embed in your agent)

```python
from next3d.tools.executor import ToolExecutor
from next3d.tools.formats import to_openai_tools, to_anthropic_tools

# Get tool schemas for your AI framework
tools = to_anthropic_tools()  # or to_openai_tools()

# Execute tool calls from the AI
executor = ToolExecutor()
result = executor.call("create_box", {"length": 100, "width": 60, "height": 20})
print(result.success, result.message, result.data)
```

### Export Schemas

```bash
next3d tools --format mcp        # MCP tool definitions
next3d tools --format openai     # OpenAI function calling format
next3d tools --format anthropic  # Anthropic tool use format
```
