"""Tool schemas — the AI-facing API definition.

Each tool is a Pydantic model that defines:
- Name and description (what the AI sees)
- Parameters with types and descriptions
- Validation

These schemas are exported as OpenAI function-calling format,
MCP tool format, or raw JSON Schema.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# CREATE tools
# ---------------------------------------------------------------------------

class CreateBox(BaseModel):
    """Create a rectangular box (block)."""

    length: float = Field(..., description="X dimension in mm", gt=0)
    width: float = Field(..., description="Y dimension in mm", gt=0)
    height: float = Field(..., description="Z dimension in mm", gt=0)
    center_x: float = Field(0, description="X center coordinate")
    center_y: float = Field(0, description="Y center coordinate")
    center_z: float = Field(0, description="Z center coordinate")


class CreateCylinder(BaseModel):
    """Create a cylinder."""

    radius: float = Field(..., description="Cylinder radius in mm", gt=0)
    height: float = Field(..., description="Cylinder height in mm", gt=0)
    center_x: float = Field(0, description="X center of base")
    center_y: float = Field(0, description="Y center of base")
    center_z: float = Field(0, description="Z center of base")
    axis: Literal["X", "Y", "Z"] = Field("Z", description="Axis direction")


class CreateSphere(BaseModel):
    """Create a sphere."""

    radius: float = Field(..., description="Sphere radius in mm", gt=0)
    center_x: float = Field(0, description="X center")
    center_y: float = Field(0, description="Y center")
    center_z: float = Field(0, description="Z center")


class CreateExtrusion(BaseModel):
    """Create a solid by extruding a 2D polygon along Z."""

    points: list[list[float]] = Field(
        ..., description="2D polygon vertices [[x,y], ...]. Auto-closed.", min_length=3,
    )
    height: float = Field(..., description="Extrusion height in mm", gt=0)
    center_x: float = Field(0, description="X offset for sketch plane")
    center_y: float = Field(0, description="Y offset for sketch plane")
    center_z: float = Field(0, description="Z offset for sketch plane")


class CreateRevolve(BaseModel):
    """Create a solid of revolution by rotating a 2D profile around an axis.

    Ideal for shafts, pulleys, bushings, nozzles, and any rotationally symmetric part.
    Profile is defined in the XZ plane."""

    points: list[list[float]] = Field(
        ...,
        description="2D profile vertices [[x, z], ...] in XZ plane. Must be on one side of the axis. Auto-closed.",
        min_length=3,
    )
    angle_degrees: float = Field(360.0, description="Revolution angle in degrees (360 = full)", gt=0, le=360)
    axis_origin_x: float = Field(0, description="X coordinate of a point on the revolution axis")
    axis_origin_z: float = Field(0, description="Z coordinate of a point on the revolution axis")
    axis_direction_x: float = Field(0, description="X component of axis direction")
    axis_direction_z: float = Field(1, description="Z component of axis direction")
    center_x: float = Field(0, description="X offset")
    center_y: float = Field(0, description="Y offset")
    center_z: float = Field(0, description="Z offset")


class CreateSweep(BaseModel):
    """Create a solid by sweeping a 2D cross-section along a 3D path.

    Ideal for pipes, channels, wiring guides, gasket grooves."""

    profile_points: list[list[float]] = Field(
        ..., description="2D cross-section vertices [[x,y], ...]. Auto-closed.", min_length=3,
    )
    path_points: list[list[float]] = Field(
        ..., description="3D path vertices [[x,y,z], ...]. At least 2 points.", min_length=2,
    )
    center_x: float = Field(0, description="X offset for path")
    center_y: float = Field(0, description="Y offset for path")
    center_z: float = Field(0, description="Z offset for path")


class CreateLoft(BaseModel):
    """Create a solid by lofting between cross-sections at different heights.

    Ideal for transitions: ducts, bottles, aerodynamic shapes."""

    sections: list[list[list[float]]] = Field(
        ..., description="List of 2D polygon sections [[[x,y],...], [[x,y],...], ...]", min_length=2,
    )
    heights: list[float] = Field(
        ..., description="Z-height for each section. Must match sections length.",
    )
    ruled: bool = Field(False, description="If true, use straight ruled surfaces between sections")
    center_x: float = Field(0, description="X offset")
    center_y: float = Field(0, description="Y offset")
    center_z: float = Field(0, description="Z offset")


# ---------------------------------------------------------------------------
# MODIFY tools
# ---------------------------------------------------------------------------

class AddHole(BaseModel):
    """Drill a hole into the current shape."""

    center_x: float = Field(..., description="X position on face")
    center_y: float = Field(..., description="Y position on face")
    diameter: float = Field(..., description="Hole diameter in mm", gt=0)
    depth: float | None = Field(None, description="Hole depth. null = through-all")
    face_selector: str = Field(">Z", description="CadQuery face selector (>Z=top, <Z=bottom, >X=right, etc.)")


class AddCounterboreHole(BaseModel):
    """Drill a counterbore hole (stepped hole for bolt heads)."""

    center_x: float = Field(..., description="X position on face")
    center_y: float = Field(..., description="Y position on face")
    hole_diameter: float = Field(..., description="Through-hole diameter in mm", gt=0)
    cb_diameter: float = Field(..., description="Counterbore diameter in mm", gt=0)
    cb_depth: float = Field(..., description="Counterbore depth in mm", gt=0)
    depth: float | None = Field(None, description="Through-hole depth. null = through-all")
    face_selector: str = Field(">Z", description="CadQuery face selector")


class AddPocket(BaseModel):
    """Cut a rectangular pocket into a face."""

    center_x: float = Field(..., description="X center on face")
    center_y: float = Field(..., description="Y center on face")
    length: float = Field(..., description="Pocket length (X) in mm", gt=0)
    width: float = Field(..., description="Pocket width (Y) in mm", gt=0)
    depth: float = Field(..., description="Pocket depth in mm", gt=0)
    face_selector: str = Field(">Z", description="CadQuery face selector")


class AddCircularPocket(BaseModel):
    """Cut a circular pocket into a face."""

    center_x: float = Field(..., description="X center on face")
    center_y: float = Field(..., description="Y center on face")
    diameter: float = Field(..., description="Pocket diameter in mm", gt=0)
    depth: float = Field(..., description="Pocket depth in mm", gt=0)
    face_selector: str = Field(">Z", description="CadQuery face selector")


class AddBoss(BaseModel):
    """Add a cylindrical boss (protrusion) on a face."""

    center_x: float = Field(..., description="X center on face")
    center_y: float = Field(..., description="Y center on face")
    diameter: float = Field(..., description="Boss diameter in mm", gt=0)
    height: float = Field(..., description="Boss height in mm", gt=0)
    face_selector: str = Field(">Z", description="CadQuery face selector")


class AddSlot(BaseModel):
    """Cut a slot (rounded-end rectangle) into a face."""

    center_x: float = Field(..., description="X center on face")
    center_y: float = Field(..., description="Y center on face")
    length: float = Field(..., description="Slot length in mm", gt=0)
    width: float = Field(..., description="Slot width in mm", gt=0)
    depth: float = Field(..., description="Slot depth in mm", gt=0)
    angle: float = Field(0, description="Rotation angle in degrees")
    face_selector: str = Field(">Z", description="CadQuery face selector")


class AddFillet(BaseModel):
    """Add fillets (rounded edges) to the shape."""

    radius: float = Field(..., description="Fillet radius in mm", gt=0)
    edge_selector: str | None = Field(
        None,
        description="CadQuery edge selector. null = all edges. Examples: '|Z' (vertical), '>Z' (top), '#Z' (horizontal)",
    )


class AddChamfer(BaseModel):
    """Add chamfers (beveled edges) to the shape."""

    distance: float = Field(..., description="Chamfer distance in mm", gt=0)
    edge_selector: str | None = Field(
        None, description="CadQuery edge selector. null = all edges.",
    )


class AddShell(BaseModel):
    """Hollow out a solid to uniform wall thickness. Essential for enclosures, housings, containers."""

    thickness: float = Field(..., description="Wall thickness in mm", gt=0)
    face_selector: str = Field(
        ">Z",
        description="CadQuery face selector for face(s) to remove (open). >Z=top, <Z=bottom, etc.",
    )


class AddDraft(BaseModel):
    """Add draft (taper) angles to faces — required for injection molding and casting."""

    angle_degrees: float = Field(..., description="Draft angle in degrees (typically 1-5°)", gt=0, lt=45)
    face_selector: str = Field("|Z", description="Faces to draft (|Z = vertical faces)")
    pull_direction_x: float = Field(0, description="X component of mold pull direction")
    pull_direction_y: float = Field(0, description="Y component of mold pull direction")
    pull_direction_z: float = Field(1, description="Z component of mold pull direction")
    plane_selector: str = Field("<Z", description="Neutral plane (parting surface) selector")


# ---------------------------------------------------------------------------
# BOOLEAN tools
# ---------------------------------------------------------------------------

class BooleanCut(BaseModel):
    """Subtract a primitive shape from the current shape."""

    tool_type: Literal["box", "cylinder", "sphere"] = Field(..., description="Type of cutting tool")
    tool_params: dict[str, Any] = Field(
        ...,
        description="Parameters for the tool shape (same as create_box/create_cylinder/create_sphere)",
    )


# ---------------------------------------------------------------------------
# TRANSFORM tools
# ---------------------------------------------------------------------------

class Translate(BaseModel):
    """Move the shape."""

    dx: float = Field(0, description="X translation in mm")
    dy: float = Field(0, description="Y translation in mm")
    dz: float = Field(0, description="Z translation in mm")


class Rotate(BaseModel):
    """Rotate the shape around an axis."""

    axis_x: float = Field(0, description="X component of rotation axis")
    axis_y: float = Field(0, description="Y component of rotation axis")
    axis_z: float = Field(1, description="Z component of rotation axis")
    angle_degrees: float = Field(..., description="Rotation angle in degrees")
    center_x: float = Field(0, description="X center of rotation")
    center_y: float = Field(0, description="Y center of rotation")
    center_z: float = Field(0, description="Z center of rotation")


# ---------------------------------------------------------------------------
# QUERY tools
# ---------------------------------------------------------------------------

class QueryGeometry(BaseModel):
    """Query the semantic graph using the DSL."""

    query: str = Field(
        ...,
        description='Query DSL string. Examples: \'faces(surface_type="cylinder")\', \'features(feature_type="through_hole")\', \'faces(radius>5)\'',
    )


class GetSummary(BaseModel):
    """Get a summary of the current geometry."""
    pass


class GetFeatures(BaseModel):
    """Get recognized features, optionally filtered by type."""

    feature_type: str | None = Field(
        None,
        description="Filter by type: through_hole, blind_hole, fillet, chamfer, slot, boss, counterbore, countersink",
    )


class FindFaces(BaseModel):
    """Find faces matching criteria."""

    surface_type: str | None = Field(None, description="plane, cylinder, cone, sphere, torus, bspline")
    min_radius: float | None = Field(None, description="Minimum radius filter")
    max_radius: float | None = Field(None, description="Maximum radius filter")
    normal_x: float | None = Field(None, description="Face normal X (for planes)")
    normal_y: float | None = Field(None, description="Face normal Y (for planes)")
    normal_z: float | None = Field(None, description="Face normal Z (for planes)")


class MeasureDistance(BaseModel):
    """Measure the minimum distance between two entities."""

    entity_id_a: str = Field(..., description="Persistent ID of first entity")
    entity_id_b: str = Field(..., description="Persistent ID of second entity")


# ---------------------------------------------------------------------------
# SESSION tools
# ---------------------------------------------------------------------------

class Undo(BaseModel):
    """Undo the last operation."""
    pass


class ExportStep(BaseModel):
    """Export current geometry to a STEP file."""

    output_path: str = Field(..., description="Output file path")


class ExportScript(BaseModel):
    """Export the operation history as a CadQuery Python script."""
    pass


class ExportStl(BaseModel):
    """Export current geometry as STL for 3D printing."""

    output_path: str = Field(..., description="Output STL file path")
    linear_deflection: float = Field(0.1, description="Max chord deviation in mm (lower = finer mesh)", gt=0)
    angular_deflection: float = Field(0.5, description="Max angle deviation in radians", gt=0)


class Export3mf(BaseModel):
    """Export current geometry as 3MF for 3D printing (modern format with color/material support)."""

    output_path: str = Field(..., description="Output 3MF file path")
    linear_deflection: float = Field(0.1, description="Max chord deviation in mm", gt=0)
    angular_deflection: float = Field(0.5, description="Max angle deviation in radians", gt=0)


class RenderPng(BaseModel):
    """Render the current geometry to a PNG/SVG image for visual feedback."""

    output_path: str = Field(..., description="Output image path (.png or .svg)")
    width: int = Field(800, description="Image width in pixels", gt=0)
    height: int = Field(600, description="Image height in pixels", gt=0)


class LoadStep(BaseModel):
    """Load geometry from a STEP file."""

    path: str = Field(..., description="Path to the STEP file")


# ---------------------------------------------------------------------------
# MULTI-BODY tools
# ---------------------------------------------------------------------------

class CreateNamedBody(BaseModel):
    """Create a new named body and make it active. Enables multi-body workflows."""

    name: str = Field(..., description="Unique body name (e.g. 'bracket', 'shaft', 'housing')")
    shape_type: Literal["box", "cylinder", "sphere", "extrusion"] = Field(
        ..., description="Type of base shape",
    )
    material: str = Field("steel", description="Material: steel, aluminum, titanium, brass, copper, nylon, abs, pla")
    length: float | None = Field(None, description="X dimension (box)")
    width: float | None = Field(None, description="Y dimension (box)")
    height: float | None = Field(None, description="Z dimension (box/cylinder/extrusion)")
    radius: float | None = Field(None, description="Radius (cylinder/sphere)")


class SetActiveBody(BaseModel):
    """Switch which body subsequent operations target."""

    name: str = Field(..., description="Body name to make active")


class ListBodies(BaseModel):
    """List all bodies with summary info (faces, material, mass, placement)."""
    pass


class DeleteBody(BaseModel):
    """Delete a named body from the session."""

    name: str = Field(..., description="Body name to delete")


class PlaceBody(BaseModel):
    """Position a body in assembly space."""

    name: str = Field(..., description="Body name to place")
    x: float = Field(0, description="X translation in mm")
    y: float = Field(0, description="Y translation in mm")
    z: float = Field(0, description="Z translation in mm")
    axis_x: float = Field(0, description="X component of rotation axis")
    axis_y: float = Field(0, description="Y component of rotation axis")
    axis_z: float = Field(1, description="Z component of rotation axis")
    angle_degrees: float = Field(0, description="Rotation angle in degrees")


class CheckInterference(BaseModel):
    """Check if two bodies collide/interfere."""

    body_a: str = Field(..., description="First body name")
    body_b: str = Field(..., description="Second body name")


class GetBom(BaseModel):
    """Get bill of materials — part list with materials, volumes, and masses."""
    pass


class AddStandardPart(BaseModel):
    """Insert a standard ISO metric fastener into the assembly."""

    name: str = Field(..., description="Body name for the part (e.g. 'bolt_1')")
    part_type: Literal["hex_bolt", "hex_nut", "flat_washer", "socket_head_cap_screw"] = Field(
        ..., description="Type of standard part",
    )
    size: str = Field(..., description="ISO metric size: M3, M4, M5, M6, M8, M10, M12")
    length: float | None = Field(None, description="Shank length in mm (for bolts/screws)")


class ExportAssembly(BaseModel):
    """Export the full assembly (all bodies with placements) as a STEP file."""

    output_path: str = Field(..., description="Output STEP file path")


class UpdateParameter(BaseModel):
    """Change a parameter value and selectively replay affected operations.

    This is parametric design: change wall_thickness from 3 to 2, and only
    the shell operation re-executes. Everything else stays untouched."""

    name: str = Field(..., description="Parameter name to change")
    new_value: float = Field(..., description="New value")


class DesignTable(BaseModel):
    """Generate design variants from parameter combinations.

    Example: {"wall_t": [2, 3, 4], "bolt_d": [4, 6]} → 6 variants,
    each with fully resolved operation parameters ready for replay."""

    param_ranges: dict[str, list[float]] = Field(
        ...,
        description="Parameter name → list of values to try. All combinations are generated.",
    )


class GetParametricState(BaseModel):
    """Get full parametric state: parameters, operation bindings, dependency graph."""
    pass


class SheetMetalDefine(BaseModel):
    """Initialize interactive sheet metal mode."""
    thickness: float = Field(..., description="Material thickness mm", gt=0)
    bend_radius: float = Field(1.0, description="Inside bend radius mm", gt=0)
    k_factor: float = Field(0.44, description="K-factor (auto from material if default)")
    material: str = Field("steel_mild", description="steel_mild, steel_stainless, aluminum, copper, brass")

class SheetMetalAddFlat(BaseModel):
    """Add a flat segment."""
    length: float = Field(..., description="Length mm", gt=0)
    width: float = Field(..., description="Width mm", gt=0)

class SheetMetalAddBend(BaseModel):
    """Add a bend."""
    angle: float = Field(..., description="Bend angle degrees")

class SheetMetalListSegments(BaseModel):
    """List all segments with bend allowances."""
    pass

class SheetMetalModifySegment(BaseModel):
    """Modify a segment."""
    index: int = Field(..., description="Segment index", ge=0)
    angle: float | None = Field(None)
    length: float | None = Field(None)
    width: float | None = Field(None)

class SheetMetalRemoveSegment(BaseModel):
    """Remove a segment by index."""
    index: int = Field(..., ge=0)

class SheetMetalInsertSegment(BaseModel):
    """Insert a segment at a position."""
    index: int = Field(..., ge=0)
    segment_type: Literal["flat", "bend"] = Field(...)
    angle: float | None = Field(None)
    length: float | None = Field(None)
    width: float | None = Field(None)

class SheetMetalGetFlatPattern(BaseModel):
    """Compute flat pattern from current segments."""
    pass

class SheetMetalGetCost(BaseModel):
    """Estimate cost from current segments."""
    pass


class CreateSheetMetal(BaseModel):
    """Create a flat sheet metal blank."""

    width: float = Field(..., description="Width (X) in mm", gt=0)
    length: float = Field(..., description="Length (Y) in mm", gt=0)
    thickness: float = Field(..., description="Material thickness in mm", gt=0)


class ComputeFlatPattern(BaseModel):
    """Compute the flat pattern (unfolded blank) from segments and bends.

    Define a sheet metal part as alternating flat segments and bends.
    Returns the flat blank geometry with bend line positions — ready for laser cutting."""

    segments: list[dict] = Field(
        ...,
        description='Alternating flat/bend segments: [{"type":"flat","length":50,"width":100}, {"type":"bend","angle":90}, {"type":"flat","length":30,"width":100}]',
    )
    thickness: float = Field(..., description="Material thickness in mm", gt=0)
    bend_radius: float = Field(1.0, description="Inside bend radius in mm", gt=0)
    k_factor: float = Field(0.44, description="K-factor (0.3-0.5). Steel=0.44, Aluminum=0.33")


class EstimateSheetMetalCost(BaseModel):
    """Estimate manufacturing cost for a sheet metal part (material + cutting + bending)."""

    segments: list[dict] = Field(..., description="Same as compute_flat_pattern segments")
    thickness: float = Field(..., description="Material thickness in mm", gt=0)
    bend_radius: float = Field(1.0, description="Inside bend radius", gt=0)
    k_factor: float = Field(0.44, description="K-factor")
    material_cost_per_kg: float = Field(2.0, description="Material cost per kg")
    density: float = Field(0.00785, description="Material density g/mm³ (steel=0.00785)")


class AddDimension(BaseModel):
    """Add a dimension annotation to the active body."""

    dim_type: Literal["linear", "radial", "diametral", "angular"] = Field(
        ..., description="Dimension type",
    )
    value: float = Field(..., description="Measured value in mm (or degrees for angular)")
    entity_ids: list[str] = Field(default_factory=list, description="Persistent IDs of referenced entities")
    label: str = Field("", description="Display label (e.g. '⌀10', 'R5')")
    tolerance_plus: float = Field(0, description="Plus tolerance")
    tolerance_minus: float = Field(0, description="Minus tolerance")


class GetDimensions(BaseModel):
    """Get all dimension annotations on the active body."""
    pass


class AutoDimension(BaseModel):
    """Auto-generate key dimensions from feature analysis (hole diameters, fillet radii, etc.)."""
    pass


class ExportDrawing(BaseModel):
    """Export a multi-view engineering drawing as SVG.

    Generates orthographic projections with hidden lines, laid out on a drawing sheet."""

    output_path: str = Field(..., description="Output SVG file path")
    views: list[str] = Field(
        default_factory=lambda: ["front", "top", "right", "isometric"],
        description="View names: front, top, right, left, back, bottom, isometric, dimetric",
    )
    title: str = Field("", description="Drawing title")
    show_hidden: bool = Field(True, description="Show hidden lines (dashed)")
    page_width: int = Field(1200, description="Page width in pixels")
    page_height: int = Field(800, description="Page height in pixels")


class ExportSectionDrawing(BaseModel):
    """Export a cross-section drawing as SVG.

    Cuts the part along a plane to reveal internal geometry."""

    output_path: str = Field(..., description="Output SVG file path")
    section_plane: Literal["XY", "XZ", "YZ"] = Field("XZ", description="Section cut plane")
    section_offset: float = Field(0, description="Offset along plane normal in mm")
    title: str = Field("", description="Drawing title")


class ExportDxf(BaseModel):
    """Export a 2D projected view as DXF for CAM/manufacturing."""

    output_path: str = Field(..., description="Output DXF file path")
    projection_dir_x: float = Field(0, description="Projection direction X")
    projection_dir_y: float = Field(0, description="Projection direction Y")
    projection_dir_z: float = Field(1, description="Projection direction Z (default: top view)")


class CheckDesignRules(BaseModel):
    """Check the active body against manufacturing design rules.

    Validates hole diameters, spacing, draft angles, overhang limits, fillet radii,
    and more — per manufacturing process."""

    process: str = Field(
        "cnc_milling",
        description="Manufacturing process: cnc_milling, injection_molding, fdm_3d_print, sla_3d_print, sheet_metal, casting",
    )


class SetParameter(BaseModel):
    """Define a named design parameter for parametric design intent.

    Named parameters let the AI express intent like 'wall_thickness=3mm'
    that can be referenced and modified later."""

    name: str = Field(..., description="Parameter name (e.g. 'wall_thickness', 'bolt_diameter')")
    value: float = Field(..., description="Parameter value in mm or degrees")
    description: str = Field("", description="Human-readable description of what this parameter controls")


class GetParameters(BaseModel):
    """Get all named design parameters."""
    pass


class AddMateConstraint(BaseModel):
    """Declare a mate constraint between two bodies (for documentation and validation)."""

    mate_type: Literal["coincident", "concentric", "flush", "distance", "angle"] = Field(
        ..., description="Type of mate constraint",
    )
    body_a: str = Field(..., description="First body name")
    entity_a: str = Field(..., description="Persistent ID of face/edge on body A")
    body_b: str = Field(..., description="Second body name")
    entity_b: str = Field(..., description="Persistent ID of face/edge on body B")
    distance: float | None = Field(None, description="Distance value (for distance mate)")
    angle: float | None = Field(None, description="Angle value in degrees (for angle mate)")


# ---------------------------------------------------------------------------
# SKETCH tools
# ---------------------------------------------------------------------------

class CreateSketch(BaseModel):
    """Start a new 2D sketch on a plane. Required before adding sketch entities."""

    plane: Literal["XY", "XZ", "YZ"] = Field("XY", description="Sketch plane")


class SketchAddLine(BaseModel):
    """Add a line segment to the active sketch."""

    x1: float = Field(..., description="Start X coordinate")
    y1: float = Field(..., description="Start Y coordinate")
    x2: float = Field(..., description="End X coordinate")
    y2: float = Field(..., description="End Y coordinate")


class SketchAddArc(BaseModel):
    """Add a circular arc to the active sketch."""

    cx: float = Field(..., description="Arc center X")
    cy: float = Field(..., description="Arc center Y")
    radius: float = Field(..., description="Arc radius in mm", gt=0)
    start_angle: float = Field(..., description="Start angle in degrees")
    end_angle: float = Field(..., description="End angle in degrees")


class SketchAddCircle(BaseModel):
    """Add a full circle to the active sketch."""

    cx: float = Field(0, description="Circle center X")
    cy: float = Field(0, description="Circle center Y")
    radius: float = Field(..., description="Circle radius in mm", gt=0)


class SketchAddRect(BaseModel):
    """Add a rectangle to the active sketch."""

    cx: float = Field(0, description="Rectangle center X")
    cy: float = Field(0, description="Rectangle center Y")
    width: float = Field(..., description="Rectangle width in mm", gt=0)
    height: float = Field(..., description="Rectangle height in mm", gt=0)


class SketchAddConstraint(BaseModel):
    """Add a geometric or dimensional constraint to the active sketch.

    Geometric: coincident, tangent, parallel, perpendicular, equal_length,
    horizontal, vertical, fixed, concentric.
    Dimensional: distance, angle, radius, diameter."""

    constraint_type: str = Field(
        ...,
        description="Constraint type: coincident, tangent, parallel, perpendicular, "
                    "equal_length, horizontal, vertical, fixed, concentric, "
                    "distance, angle, radius, diameter",
    )
    entity_a: str = Field(..., description="ID of the first entity")
    entity_b: str | None = Field(None, description="ID of the second entity (optional)")
    value: float | None = Field(None, description="Dimensional value (for distance, angle, radius, diameter)")


class SketchExtrude(BaseModel):
    """Extrude the active sketch to create a solid."""

    height: float = Field(..., description="Extrusion height in mm", gt=0)


class SketchRevolve(BaseModel):
    """Revolve the active sketch around an axis to create a solid."""

    angle_degrees: float = Field(360.0, description="Revolution angle in degrees", gt=0, le=360)
    axis_origin_x: float = Field(0, description="X of a point on the revolution axis")
    axis_origin_y: float = Field(0, description="Y of a point on the revolution axis")
    axis_direction_x: float = Field(0, description="X component of axis direction")
    axis_direction_y: float = Field(1, description="Y component of axis direction")


# ---------------------------------------------------------------------------
# GD&T tools
# ---------------------------------------------------------------------------

class AddDatum(BaseModel):
    """Add a GD&T datum reference to the active body.

    Datums are labeled reference features (A, B, C) used as the basis
    for geometric tolerancing per ASME Y14.5 / ISO 1101."""

    label: str = Field(..., description="Datum label — single uppercase letter (e.g. 'A', 'B', 'C')")
    entity_id: str = Field(..., description="Persistent ID of the datum feature (face or edge)")
    description: str = Field("", description="Human-readable description of the datum")


class AddTolerance(BaseModel):
    """Add a GD&T tolerance zone to the active body.

    Defines a geometric tolerance (flatness, position, perpendicularity, etc.)
    on a feature, optionally referencing datums."""

    tolerance_type: str = Field(
        ...,
        description=(
            "Tolerance type: flatness, straightness, circularity, cylindricity, "
            "parallelism, perpendicularity, angularity, position, concentricity, "
            "symmetry, circular_runout, total_runout, profile_of_line, profile_of_surface"
        ),
    )
    value: float = Field(..., description="Tolerance value in mm", gt=0)
    entity_id: str = Field(..., description="Persistent ID of the controlled feature")
    datum_refs: list[str] = Field(
        default_factory=list,
        description="Datum labels this tolerance references (e.g. ['A', 'B'])",
    )
    material_condition: str = Field(
        "",
        description="Material condition modifier: 'MMC', 'LMC', 'RFS', or '' (none)",
    )
    description: str = Field("", description="Human-readable description")


class GetGDT(BaseModel):
    """Get all GD&T annotations (datums and tolerances) for the active body."""
    pass


class SuggestGDT(BaseModel):
    """Auto-suggest GD&T annotations based on feature analysis.

    Analyzes through holes, planar faces, cylindrical surfaces, etc.
    and suggests appropriate datums and tolerances per common engineering practice."""
    pass


# ---------------------------------------------------------------------------
# TOPOLOGY OPTIMIZATION tools
# ---------------------------------------------------------------------------

class AddLoad(BaseModel):
    """Add a load case for topology optimization.

    Define a force vector applied at a specific point on the geometry.
    Multiple loads can be added before running optimization."""

    name: str = Field(..., description="Load case name (e.g. 'downward_force', 'lateral_push')")
    fx: float = Field(0, description="Force X component in Newtons")
    fy: float = Field(0, description="Force Y component in Newtons")
    fz: float = Field(0, description="Force Z component in Newtons")
    px: float = Field(0, description="Application point X coordinate")
    py: float = Field(0, description="Application point Y coordinate")
    pz: float = Field(0, description="Application point Z coordinate")


class AddBoundaryCondition(BaseModel):
    """Add a boundary condition (support) for topology optimization.

    Define how the geometry is constrained. At least one BC is required
    before running optimization."""

    name: str = Field(..., description="Boundary condition name (e.g. 'fixed_base', 'roller_support')")
    bc_type: Literal["fixed", "pinned", "roller"] = Field(
        ..., description="Constraint type: fixed (no movement), pinned (no translation), roller (slides in one direction)",
    )
    face_selector: str = Field(
        ..., description="CadQuery face selector for the constrained face (>Z=top, <Z=bottom, >X=right, etc.)",
    )


class RunTopologyOptimization(BaseModel):
    """Run topology optimization on the active body.

    Requires at least one load and one boundary condition to be defined first.
    Uses a simplified SIMP-like approach with distance-based stress heuristics
    to identify material that can be removed."""

    volume_fraction: float = Field(
        0.3,
        description="Target volume fraction — 0.3 means keep 30% of material. Range: 0.01 to 0.99.",
        gt=0,
        lt=1,
    )
    resolution: int = Field(
        10,
        description="Voxel grid resolution along longest axis. Higher = more detail but slower. Range: 3 to 50.",
        ge=3,
        le=50,
    )


# ---------------------------------------------------------------------------
# FEA tools
# ---------------------------------------------------------------------------

class RunFEA(BaseModel):
    """Run Finite Element Analysis on a plate/stiffener structure.

    General-purpose structural solver for plates with optional RHS stiffener grids.
    Computes deflections, stresses, and safety factors.

    Materials: steel_mild, steel_ss304, steel_ss316, aluminum_6061, aluminum_5052, carbon_steel.
    RHS sizes: 25x25x2, 25x25x3, 40x40x2, 40x40x3, 50x50x3, 50x50x4, 50x25x2, 50x25x3, 75x50x3, 100x50x3, 100x100x4."""

    plate_width: float = Field(..., description="Plate width in mm (X direction)", gt=0)
    plate_height: float = Field(..., description="Plate height in mm (Y direction)", gt=0)
    plate_thickness: float = Field(..., description="Plate/sheet thickness in mm", gt=0)
    grid_spacing_x: float = Field(0, description="Stiffener grid spacing X in mm (0 = no stiffeners)")
    grid_spacing_y: float = Field(0, description="Stiffener grid spacing Y in mm (0 = no stiffeners)")
    rhs_size: str | None = Field(None, description="RHS stiffener section (e.g. '50x50x3'). null = plate only")
    material: str = Field("steel_mild", description="Material name")
    pressure_mpa: float = Field(0, description="Uniform pressure in MPa (N/mm²)")
    point_loads: list[dict] | None = Field(None, description="Point loads: [{x, y, force}, ...]")
    bc_type: str = Field("fixed_edges", description="Boundary condition: fixed_edges, simply_supported, fixed_corners")
    weld_type: str = Field("full", description="Weld type: full, intermittent, spot")
    weld_spacing: float = Field(50, description="Weld spacing in mm (for intermittent/spot)", gt=0)


class RunFEAParametric(BaseModel):
    """Run FEA across multiple configurations for comparison.

    Compare different materials, RHS sizes, weld types, etc. in one call."""

    base_config: dict = Field(..., description="Base configuration dict with plate_width, plate_height, plate_thickness, etc.")
    variations: list[dict] = Field(..., description="List of dicts with fields to override per run, e.g. [{rhs_size: '25x25x2'}, {rhs_size: '50x50x3'}]")


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: dict[str, type[BaseModel]] = {
    # Create
    "create_box": CreateBox,
    "create_cylinder": CreateCylinder,
    "create_sphere": CreateSphere,
    "create_extrusion": CreateExtrusion,
    "create_revolve": CreateRevolve,
    "create_sweep": CreateSweep,
    "create_loft": CreateLoft,
    # Modify
    "add_hole": AddHole,
    "add_counterbore_hole": AddCounterboreHole,
    "add_pocket": AddPocket,
    "add_circular_pocket": AddCircularPocket,
    "add_boss": AddBoss,
    "add_slot": AddSlot,
    "add_fillet": AddFillet,
    "add_chamfer": AddChamfer,
    "add_shell": AddShell,
    "add_draft": AddDraft,
    # Boolean
    "boolean_cut": BooleanCut,
    # Transform
    "translate": Translate,
    "rotate": Rotate,
    # Query
    "query_geometry": QueryGeometry,
    "get_summary": GetSummary,
    "get_features": GetFeatures,
    "find_faces": FindFaces,
    "measure_distance": MeasureDistance,
    # Design intelligence
    # Interactive sheet metal
    "sheet_metal_define": SheetMetalDefine,
    "sheet_metal_add_flat": SheetMetalAddFlat,
    "sheet_metal_add_bend": SheetMetalAddBend,
    "sheet_metal_list_segments": SheetMetalListSegments,
    "sheet_metal_modify_segment": SheetMetalModifySegment,
    "sheet_metal_remove_segment": SheetMetalRemoveSegment,
    "sheet_metal_insert_segment": SheetMetalInsertSegment,
    "sheet_metal_get_flat_pattern": SheetMetalGetFlatPattern,
    "sheet_metal_get_cost": SheetMetalGetCost,
    # Sheet metal (one-shot)
    "create_sheet_metal": CreateSheetMetal,
    "compute_flat_pattern": ComputeFlatPattern,
    "estimate_sheet_metal_cost": EstimateSheetMetalCost,
    # Dimensions & Drawings
    "add_dimension": AddDimension,
    "get_dimensions": GetDimensions,
    "auto_dimension": AutoDimension,
    "export_drawing": ExportDrawing,
    "export_section_drawing": ExportSectionDrawing,
    "export_dxf": ExportDxf,
    # Design intelligence
    "check_design_rules": CheckDesignRules,
    "set_parameter": SetParameter,
    "get_parameters": GetParameters,
    "update_parameter": UpdateParameter,
    "design_table": DesignTable,
    "get_parametric_state": GetParametricState,
    # Multi-body
    "create_named_body": CreateNamedBody,
    "set_active_body": SetActiveBody,
    "list_bodies": ListBodies,
    "delete_body": DeleteBody,
    # Assembly
    "place_body": PlaceBody,
    "add_mate_constraint": AddMateConstraint,
    "check_interference": CheckInterference,
    "get_bom": GetBom,
    "add_standard_part": AddStandardPart,
    "export_assembly": ExportAssembly,
    # Sketch
    "create_sketch": CreateSketch,
    "sketch_add_line": SketchAddLine,
    "sketch_add_arc": SketchAddArc,
    "sketch_add_circle": SketchAddCircle,
    "sketch_add_rect": SketchAddRect,
    "sketch_add_constraint": SketchAddConstraint,
    "sketch_extrude": SketchExtrude,
    "sketch_revolve": SketchRevolve,
    # GD&T
    "add_datum": AddDatum,
    "add_tolerance": AddTolerance,
    "get_gdt": GetGDT,
    "suggest_gdt": SuggestGDT,
    # Topology optimization
    "add_load": AddLoad,
    "add_boundary_condition": AddBoundaryCondition,
    "run_topology_optimization": RunTopologyOptimization,
    # FEA
    "run_fea": RunFEA,
    "run_fea_parametric": RunFEAParametric,
    # Session
    "undo": Undo,
    "export_step": ExportStep,
    "export_stl": ExportStl,
    "export_3mf": Export3mf,
    "render_png": RenderPng,
    "export_script": ExportScript,
    "load_step": LoadStep,
}
