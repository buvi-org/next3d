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


class LoadStep(BaseModel):
    """Load geometry from a STEP file."""

    path: str = Field(..., description="Path to the STEP file")


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: dict[str, type[BaseModel]] = {
    # Create
    "create_box": CreateBox,
    "create_cylinder": CreateCylinder,
    "create_sphere": CreateSphere,
    "create_extrusion": CreateExtrusion,
    # Modify
    "add_hole": AddHole,
    "add_counterbore_hole": AddCounterboreHole,
    "add_pocket": AddPocket,
    "add_circular_pocket": AddCircularPocket,
    "add_boss": AddBoss,
    "add_slot": AddSlot,
    "add_fillet": AddFillet,
    "add_chamfer": AddChamfer,
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
    # Session
    "undo": Undo,
    "export_step": ExportStep,
    "export_script": ExportScript,
    "load_step": LoadStep,
}
