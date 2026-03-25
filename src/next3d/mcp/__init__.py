"""MCP server for next3d — exposes 3D modeling tools to AI agents.

Run with: next3d serve
Or:       python -m next3d.mcp.server

The server exposes all 22 modeling tools over stdio (default) or SSE.
Any MCP client (Claude Code, Claude Desktop, etc.) can connect and
create/modify/query 3D geometry through structured tool calls.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from next3d.modeling.session import ModelingSession
from next3d.tools.executor import ToolExecutor, ToolResult
from next3d.tools.schema import TOOL_SCHEMAS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "next3d",
    instructions=(
        "3D modeling tools for creating, modifying, querying, and exporting "
        "parametric 3D geometry. Start by creating a shape (create_box, "
        "create_cylinder, create_revolve, create_loft, etc.) or loading a STEP file. "
        "Modify with add_hole, add_pocket, add_shell, add_draft, add_fillet, etc. "
        "Query with get_summary, get_features, find_faces. "
        "Export to STEP, STL, 3MF, or PNG/SVG. Use undo to revert. "
        "Face selectors: >Z=top, <Z=bottom, >X=right, <X=left, >Y=front, <Y=back. "
        "Edge selectors: |Z=vertical, |X=along-X, #Z=perpendicular-to-Z."
    ),
)

# Single shared session — stateful across tool calls
_executor = ToolExecutor()


def _format_result(result: ToolResult) -> str:
    """Format a ToolResult as a string for MCP text content."""
    parts = [result.message]
    if result.data:
        parts.append(json.dumps(result.data, indent=2, default=str))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# CREATE tools
# ---------------------------------------------------------------------------

@mcp.tool()
def create_box(
    length: float,
    width: float,
    height: float,
    center_x: float = 0,
    center_y: float = 0,
    center_z: float = 0,
) -> str:
    """Create a rectangular box.

    Args:
        length: X dimension in mm (must be > 0)
        width: Y dimension in mm (must be > 0)
        height: Z dimension in mm (must be > 0)
        center_x: X center coordinate
        center_y: Y center coordinate
        center_z: Z center coordinate
    """
    r = _executor.call("create_box", {
        "length": length, "width": width, "height": height,
        "center_x": center_x, "center_y": center_y, "center_z": center_z,
    })
    return _format_result(r)


@mcp.tool()
def create_cylinder(
    radius: float,
    height: float,
    center_x: float = 0,
    center_y: float = 0,
    center_z: float = 0,
    axis: str = "Z",
) -> str:
    """Create a cylinder.

    Args:
        radius: Cylinder radius in mm
        height: Cylinder height in mm
        center_x: X center of base
        center_y: Y center of base
        center_z: Z center of base
        axis: Axis direction — "X", "Y", or "Z"
    """
    r = _executor.call("create_cylinder", {
        "radius": radius, "height": height,
        "center_x": center_x, "center_y": center_y, "center_z": center_z,
        "axis": axis,
    })
    return _format_result(r)


@mcp.tool()
def create_sphere(
    radius: float,
    center_x: float = 0,
    center_y: float = 0,
    center_z: float = 0,
) -> str:
    """Create a sphere.

    Args:
        radius: Sphere radius in mm
        center_x: X center
        center_y: Y center
        center_z: Z center
    """
    r = _executor.call("create_sphere", {
        "radius": radius,
        "center_x": center_x, "center_y": center_y, "center_z": center_z,
    })
    return _format_result(r)


@mcp.tool()
def create_extrusion(
    points: list[list[float]],
    height: float,
    center_x: float = 0,
    center_y: float = 0,
    center_z: float = 0,
) -> str:
    """Create a solid by extruding a 2D polygon along Z.

    Args:
        points: 2D polygon vertices [[x,y], ...]. Auto-closed. Min 3 points.
        height: Extrusion height in mm
        center_x: X offset for sketch plane
        center_y: Y offset for sketch plane
        center_z: Z offset for sketch plane
    """
    r = _executor.call("create_extrusion", {
        "points": points, "height": height,
        "center_x": center_x, "center_y": center_y, "center_z": center_z,
    })
    return _format_result(r)


@mcp.tool()
def create_revolve(
    points: list[list[float]],
    angle_degrees: float = 360.0,
    axis_origin_x: float = 0,
    axis_origin_z: float = 0,
    axis_direction_x: float = 0,
    axis_direction_z: float = 1,
    center_x: float = 0,
    center_y: float = 0,
    center_z: float = 0,
) -> str:
    """Create a solid of revolution — rotate a 2D profile around an axis.

    Ideal for shafts, pulleys, bushings, nozzles (~30% of all mechanical parts).
    Profile is in XZ plane; default axis is Z (vertical).

    Args:
        points: 2D profile vertices [[x,z], ...] in XZ plane. Must be on one side of axis.
        angle_degrees: Revolution angle (360 = full revolution)
        axis_origin_x: X of a point on the revolution axis
        axis_origin_z: Z of a point on the revolution axis
        axis_direction_x: X component of axis direction
        axis_direction_z: Z component of axis direction
        center_x: X offset
        center_y: Y offset
        center_z: Z offset
    """
    r = _executor.call("create_revolve", {
        "points": points, "angle_degrees": angle_degrees,
        "axis_origin_x": axis_origin_x, "axis_origin_z": axis_origin_z,
        "axis_direction_x": axis_direction_x, "axis_direction_z": axis_direction_z,
        "center_x": center_x, "center_y": center_y, "center_z": center_z,
    })
    return _format_result(r)


@mcp.tool()
def create_sweep(
    profile_points: list[list[float]],
    path_points: list[list[float]],
    center_x: float = 0,
    center_y: float = 0,
    center_z: float = 0,
) -> str:
    """Create a solid by sweeping a 2D cross-section along a 3D path.

    Ideal for pipes, channels, wiring guides, gasket grooves.

    Args:
        profile_points: 2D cross-section vertices [[x,y], ...]
        path_points: 3D path vertices [[x,y,z], ...]. Min 2 points.
        center_x: X offset for path
        center_y: Y offset for path
        center_z: Z offset for path
    """
    r = _executor.call("create_sweep", {
        "profile_points": profile_points, "path_points": path_points,
        "center_x": center_x, "center_y": center_y, "center_z": center_z,
    })
    return _format_result(r)


@mcp.tool()
def create_loft(
    sections: list[list[list[float]]],
    heights: list[float],
    ruled: bool = False,
    center_x: float = 0,
    center_y: float = 0,
    center_z: float = 0,
) -> str:
    """Create a solid by lofting between cross-sections at different heights.

    Ideal for transitions: ducts, bottles, aerodynamic shapes.

    Args:
        sections: List of 2D polygon sections [[[x,y],...], ...]
        heights: Z-height for each section (must match sections length)
        ruled: If true, straight ruled surfaces between sections
        center_x: X offset
        center_y: Y offset
        center_z: Z offset
    """
    r = _executor.call("create_loft", {
        "sections": sections, "heights": heights, "ruled": ruled,
        "center_x": center_x, "center_y": center_y, "center_z": center_z,
    })
    return _format_result(r)


# ---------------------------------------------------------------------------
# MODIFY tools
# ---------------------------------------------------------------------------

@mcp.tool()
def add_hole(
    center_x: float,
    center_y: float,
    diameter: float,
    depth: float | None = None,
    face_selector: str = ">Z",
) -> str:
    """Drill a hole into the current shape.

    Args:
        center_x: X position on the selected face
        center_y: Y position on the selected face
        diameter: Hole diameter in mm
        depth: Hole depth in mm. null = through-all
        face_selector: Which face to drill into. >Z=top, <Z=bottom, >X=right, <X=left
    """
    r = _executor.call("add_hole", {
        "center_x": center_x, "center_y": center_y,
        "diameter": diameter, "depth": depth, "face_selector": face_selector,
    })
    return _format_result(r)


@mcp.tool()
def add_counterbore_hole(
    center_x: float,
    center_y: float,
    hole_diameter: float,
    cb_diameter: float,
    cb_depth: float,
    depth: float | None = None,
    face_selector: str = ">Z",
) -> str:
    """Drill a counterbore hole (stepped hole for bolt heads).

    Args:
        center_x: X position on face
        center_y: Y position on face
        hole_diameter: Through-hole diameter in mm
        cb_diameter: Counterbore diameter in mm (must be > hole_diameter)
        cb_depth: Counterbore depth in mm
        depth: Through-hole depth. null = through-all
        face_selector: Which face to drill into
    """
    r = _executor.call("add_counterbore_hole", {
        "center_x": center_x, "center_y": center_y,
        "hole_diameter": hole_diameter, "cb_diameter": cb_diameter,
        "cb_depth": cb_depth, "depth": depth, "face_selector": face_selector,
    })
    return _format_result(r)


@mcp.tool()
def add_pocket(
    center_x: float,
    center_y: float,
    length: float,
    width: float,
    depth: float,
    face_selector: str = ">Z",
) -> str:
    """Cut a rectangular pocket into a face.

    Args:
        center_x: X center of pocket on face
        center_y: Y center of pocket on face
        length: Pocket length (X direction) in mm
        width: Pocket width (Y direction) in mm
        depth: Pocket depth in mm
        face_selector: Which face to cut into
    """
    r = _executor.call("add_pocket", {
        "center_x": center_x, "center_y": center_y,
        "length": length, "width": width, "depth": depth,
        "face_selector": face_selector,
    })
    return _format_result(r)


@mcp.tool()
def add_circular_pocket(
    center_x: float,
    center_y: float,
    diameter: float,
    depth: float,
    face_selector: str = ">Z",
) -> str:
    """Cut a circular pocket into a face.

    Args:
        center_x: X center on face
        center_y: Y center on face
        diameter: Pocket diameter in mm
        depth: Pocket depth in mm
        face_selector: Which face to cut into
    """
    r = _executor.call("add_circular_pocket", {
        "center_x": center_x, "center_y": center_y,
        "diameter": diameter, "depth": depth, "face_selector": face_selector,
    })
    return _format_result(r)


@mcp.tool()
def add_boss(
    center_x: float,
    center_y: float,
    diameter: float,
    height: float,
    face_selector: str = ">Z",
) -> str:
    """Add a cylindrical boss (protrusion) on a face.

    Args:
        center_x: X center on face
        center_y: Y center on face
        diameter: Boss diameter in mm
        height: Boss height in mm
        face_selector: Which face to add boss on
    """
    r = _executor.call("add_boss", {
        "center_x": center_x, "center_y": center_y,
        "diameter": diameter, "height": height, "face_selector": face_selector,
    })
    return _format_result(r)


@mcp.tool()
def add_slot(
    center_x: float,
    center_y: float,
    length: float,
    width: float,
    depth: float,
    angle: float = 0,
    face_selector: str = ">Z",
) -> str:
    """Cut a slot (rounded-end rectangle) into a face.

    Args:
        center_x: X center on face
        center_y: Y center on face
        length: Slot length in mm
        width: Slot width in mm
        depth: Slot depth in mm
        angle: Rotation angle in degrees
        face_selector: Which face to cut into
    """
    r = _executor.call("add_slot", {
        "center_x": center_x, "center_y": center_y,
        "length": length, "width": width, "depth": depth,
        "angle": angle, "face_selector": face_selector,
    })
    return _format_result(r)


@mcp.tool()
def add_fillet(
    radius: float,
    edge_selector: str | None = None,
) -> str:
    """Add fillets (rounded edges) to the shape.

    Args:
        radius: Fillet radius in mm
        edge_selector: Which edges. null=all, |Z=vertical, >Z=top, #Z=horizontal
    """
    r = _executor.call("add_fillet", {
        "radius": radius, "edge_selector": edge_selector,
    })
    return _format_result(r)


@mcp.tool()
def add_chamfer(
    distance: float,
    edge_selector: str | None = None,
) -> str:
    """Add chamfers (beveled edges) to the shape.

    Args:
        distance: Chamfer distance in mm
        edge_selector: Which edges. null=all, |Z=vertical, >Z=top, #Z=horizontal
    """
    r = _executor.call("add_chamfer", {
        "distance": distance, "edge_selector": edge_selector,
    })
    return _format_result(r)


@mcp.tool()
def add_shell(
    thickness: float,
    face_selector: str = ">Z",
) -> str:
    """Hollow out a solid to uniform wall thickness.

    Essential for enclosures, housings, containers. The selected face is
    removed (left open) and remaining walls are offset inward.

    Args:
        thickness: Wall thickness in mm
        face_selector: Face to remove (open). >Z=top, <Z=bottom, >X=right
    """
    r = _executor.call("add_shell", {
        "thickness": thickness, "face_selector": face_selector,
    })
    return _format_result(r)


@mcp.tool()
def add_draft(
    angle_degrees: float,
    face_selector: str = "|Z",
    pull_direction_x: float = 0,
    pull_direction_y: float = 0,
    pull_direction_z: float = 1,
    plane_selector: str = "<Z",
) -> str:
    """Add draft (taper) angles to faces for mold release.

    Required for injection molding and casting.

    Args:
        angle_degrees: Draft angle in degrees (typically 1-5)
        face_selector: Faces to draft (|Z = vertical faces)
        pull_direction_x: X component of mold pull direction
        pull_direction_y: Y component of mold pull direction
        pull_direction_z: Z component of mold pull direction
        plane_selector: Neutral plane (parting surface) selector
    """
    r = _executor.call("add_draft", {
        "angle_degrees": angle_degrees, "face_selector": face_selector,
        "pull_direction_x": pull_direction_x,
        "pull_direction_y": pull_direction_y,
        "pull_direction_z": pull_direction_z,
        "plane_selector": plane_selector,
    })
    return _format_result(r)


# ---------------------------------------------------------------------------
# BOOLEAN tools
# ---------------------------------------------------------------------------

@mcp.tool()
def boolean_cut(
    tool_type: str,
    tool_params: dict[str, Any],
) -> str:
    """Subtract a primitive shape from the current shape.

    Args:
        tool_type: Type of cutting tool — "box", "cylinder", or "sphere"
        tool_params: Parameters for the tool (same as create_box/create_cylinder/create_sphere)
    """
    r = _executor.call("boolean_cut", {
        "tool_type": tool_type, "tool_params": tool_params,
    })
    return _format_result(r)


# ---------------------------------------------------------------------------
# TRANSFORM tools
# ---------------------------------------------------------------------------

@mcp.tool()
def translate(dx: float = 0, dy: float = 0, dz: float = 0) -> str:
    """Move the shape.

    Args:
        dx: X translation in mm
        dy: Y translation in mm
        dz: Z translation in mm
    """
    r = _executor.call("translate", {"dx": dx, "dy": dy, "dz": dz})
    return _format_result(r)


@mcp.tool()
def rotate(
    angle_degrees: float,
    axis_x: float = 0,
    axis_y: float = 0,
    axis_z: float = 1,
    center_x: float = 0,
    center_y: float = 0,
    center_z: float = 0,
) -> str:
    """Rotate the shape around an axis.

    Args:
        angle_degrees: Rotation angle in degrees
        axis_x: X component of rotation axis
        axis_y: Y component of rotation axis
        axis_z: Z component of rotation axis (default 1 = rotate around Z)
        center_x: X center of rotation
        center_y: Y center of rotation
        center_z: Z center of rotation
    """
    r = _executor.call("rotate", {
        "axis_x": axis_x, "axis_y": axis_y, "axis_z": axis_z,
        "angle_degrees": angle_degrees,
        "center_x": center_x, "center_y": center_y, "center_z": center_z,
    })
    return _format_result(r)


# ---------------------------------------------------------------------------
# QUERY tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_summary() -> str:
    """Get a summary of the current geometry — face count, features, solids."""
    r = _executor.call("get_summary", {})
    return _format_result(r)


@mcp.tool()
def get_features(feature_type: str | None = None) -> str:
    """Get recognized features, optionally filtered by type.

    Args:
        feature_type: Filter — through_hole, blind_hole, fillet, chamfer, slot, boss, counterbore, countersink. null = all.
    """
    r = _executor.call("get_features", {"feature_type": feature_type})
    return _format_result(r)


@mcp.tool()
def find_faces(
    surface_type: str | None = None,
    min_radius: float | None = None,
    max_radius: float | None = None,
    normal_x: float | None = None,
    normal_y: float | None = None,
    normal_z: float | None = None,
) -> str:
    """Find faces matching criteria.

    Args:
        surface_type: plane, cylinder, cone, sphere, torus, bspline
        min_radius: Minimum radius filter (for curved faces)
        max_radius: Maximum radius filter
        normal_x: Face normal X component (for planes, e.g. 0,0,1 = top face)
        normal_y: Face normal Y component
        normal_z: Face normal Z component
    """
    r = _executor.call("find_faces", {
        "surface_type": surface_type,
        "min_radius": min_radius, "max_radius": max_radius,
        "normal_x": normal_x, "normal_y": normal_y, "normal_z": normal_z,
    })
    return _format_result(r)


@mcp.tool()
def query_geometry(query: str) -> str:
    """Query the semantic graph using the DSL.

    Args:
        query: DSL string. Examples: 'faces(surface_type="cylinder")', 'features(feature_type="through_hole")', 'faces(radius>5)'
    """
    r = _executor.call("query_geometry", {"query": query})
    return _format_result(r)


# ---------------------------------------------------------------------------
# SESSION tools
# ---------------------------------------------------------------------------

@mcp.tool()
def load_step(path: str) -> str:
    """Load geometry from a STEP file.

    Args:
        path: Absolute path to the STEP file
    """
    r = _executor.call("load_step", {"path": path})
    return _format_result(r)


@mcp.tool()
def export_step(output_path: str) -> str:
    """Export current geometry to a STEP file.

    Args:
        output_path: Output file path (e.g. /tmp/part.step)
    """
    r = _executor.call("export_step", {"output_path": output_path})
    return _format_result(r)


@mcp.tool()
def export_stl(
    output_path: str,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
) -> str:
    """Export current geometry as STL for 3D printing.

    Args:
        output_path: Output STL file path
        linear_deflection: Max chord deviation in mm (lower = finer mesh)
        angular_deflection: Max angle deviation in radians
    """
    r = _executor.call("export_stl", {
        "output_path": output_path,
        "linear_deflection": linear_deflection,
        "angular_deflection": angular_deflection,
    })
    return _format_result(r)


@mcp.tool()
def export_3mf(
    output_path: str,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
) -> str:
    """Export current geometry as 3MF for 3D printing (modern format).

    Args:
        output_path: Output 3MF file path
        linear_deflection: Max chord deviation in mm
        angular_deflection: Max angle deviation in radians
    """
    r = _executor.call("export_3mf", {
        "output_path": output_path,
        "linear_deflection": linear_deflection,
        "angular_deflection": angular_deflection,
    })
    return _format_result(r)


@mcp.tool()
def render_png(
    output_path: str,
    width: int = 800,
    height: int = 600,
) -> str:
    """Render geometry to PNG/SVG image for visual feedback.

    Lets the AI 'see' what it built and self-correct.

    Args:
        output_path: Output path (.png requires cairosvg, .svg always works)
        width: Image width in pixels
        height: Image height in pixels
    """
    r = _executor.call("render_png", {
        "output_path": output_path, "width": width, "height": height,
    })
    return _format_result(r)


@mcp.tool()
def check_design_rules(process: str = "cnc_milling") -> str:
    """Check active body against manufacturing design rules.

    Validates hole diameters/spacing, draft angles, overhang limits, fillet radii.

    Args:
        process: cnc_milling, injection_molding, fdm_3d_print, sla_3d_print, sheet_metal, casting
    """
    r = _executor.call("check_design_rules", {"process": process})
    return _format_result(r)


@mcp.tool()
def set_parameter(
    name: str,
    value: float,
    description: str = "",
) -> str:
    """Define a named design parameter for parametric intent.

    Args:
        name: Parameter name (e.g. 'wall_thickness')
        value: Value in mm or degrees
        description: What this parameter controls
    """
    r = _executor.call("set_parameter", {
        "name": name, "value": value, "description": description,
    })
    return _format_result(r)


@mcp.tool()
def get_parameters() -> str:
    """Get all named design parameters."""
    r = _executor.call("get_parameters", {})
    return _format_result(r)


@mcp.tool()
def export_assembly(output_path: str) -> str:
    """Export the full assembly (all bodies with placements) as a STEP file.

    Args:
        output_path: Output STEP file path
    """
    r = _executor.call("export_assembly", {"output_path": output_path})
    return _format_result(r)


@mcp.tool()
def export_script() -> str:
    """Export the operation history as a standalone CadQuery Python script.

    Returns the full Python code that reproduces the current geometry.
    """
    r = _executor.call("export_script", {})
    return _format_result(r)


# ---------------------------------------------------------------------------
# MULTI-BODY tools
# ---------------------------------------------------------------------------

@mcp.tool()
def create_named_body(
    name: str,
    shape_type: str,
    material: str = "steel",
    length: float | None = None,
    width: float | None = None,
    height: float | None = None,
    radius: float | None = None,
) -> str:
    """Create a new named body. Enables multi-part assemblies.

    Args:
        name: Unique body name (e.g. 'bracket', 'shaft')
        shape_type: "box", "cylinder", "sphere", or "extrusion"
        material: Material: steel, aluminum, titanium, brass, copper, nylon, abs, pla
        length: X dimension (box)
        width: Y dimension (box)
        height: Z dimension (box/cylinder/extrusion)
        radius: Radius (cylinder/sphere)
    """
    r = _executor.call("create_named_body", {
        "name": name, "shape_type": shape_type, "material": material,
        "length": length, "width": width, "height": height, "radius": radius,
    })
    return _format_result(r)


@mcp.tool()
def set_active_body(name: str) -> str:
    """Switch which body subsequent operations (add_hole, add_fillet, etc.) target.

    Args:
        name: Body name to make active
    """
    r = _executor.call("set_active_body", {"name": name})
    return _format_result(r)


@mcp.tool()
def list_bodies() -> str:
    """List all bodies with faces, material, mass, and placement info."""
    r = _executor.call("list_bodies", {})
    return _format_result(r)


@mcp.tool()
def delete_body(name: str) -> str:
    """Delete a named body from the session.

    Args:
        name: Body to delete
    """
    r = _executor.call("delete_body", {"name": name})
    return _format_result(r)


@mcp.tool()
def place_body(
    name: str,
    x: float = 0,
    y: float = 0,
    z: float = 0,
    axis_x: float = 0,
    axis_y: float = 0,
    axis_z: float = 1,
    angle_degrees: float = 0,
) -> str:
    """Position a body in assembly space.

    Args:
        name: Body to place
        x: X translation in mm
        y: Y translation in mm
        z: Z translation in mm
        axis_x: Rotation axis X
        axis_y: Rotation axis Y
        axis_z: Rotation axis Z (default 1 = around Z)
        angle_degrees: Rotation angle
    """
    r = _executor.call("place_body", {
        "name": name, "x": x, "y": y, "z": z,
        "axis_x": axis_x, "axis_y": axis_y, "axis_z": axis_z,
        "angle_degrees": angle_degrees,
    })
    return _format_result(r)


@mcp.tool()
def check_interference(body_a: str, body_b: str) -> str:
    """Check if two bodies collide/interfere.

    Args:
        body_a: First body name
        body_b: Second body name
    """
    r = _executor.call("check_interference", {"body_a": body_a, "body_b": body_b})
    return _format_result(r)


@mcp.tool()
def get_bom() -> str:
    """Get bill of materials — part list with materials, volumes, and masses."""
    r = _executor.call("get_bom", {})
    return _format_result(r)


@mcp.tool()
def add_standard_part(
    name: str,
    part_type: str,
    size: str,
    length: float | None = None,
) -> str:
    """Insert a standard ISO metric fastener.

    Args:
        name: Body name for the part (e.g. 'bolt_1')
        part_type: hex_bolt, hex_nut, flat_washer, socket_head_cap_screw
        size: ISO size: M3, M4, M5, M6, M8, M10, M12
        length: Shank length in mm (for bolts/screws)
    """
    r = _executor.call("add_standard_part", {
        "name": name, "part_type": part_type, "size": size, "length": length,
    })
    return _format_result(r)


@mcp.tool()
def add_mate_constraint(
    mate_type: str,
    body_a: str,
    entity_a: str,
    body_b: str,
    entity_b: str,
    distance: float | None = None,
    angle: float | None = None,
) -> str:
    """Declare a mate constraint between two bodies.

    Args:
        mate_type: coincident, concentric, flush, distance, angle
        body_a: First body name
        entity_a: Persistent ID of face/edge on body A
        body_b: Second body name
        entity_b: Persistent ID of face/edge on body B
        distance: Distance value (for distance mate)
        angle: Angle value (for angle mate)
    """
    r = _executor.call("add_mate_constraint", {
        "mate_type": mate_type, "body_a": body_a, "entity_a": entity_a,
        "body_b": body_b, "entity_b": entity_b,
        "distance": distance, "angle": angle,
    })
    return _format_result(r)


@mcp.tool()
def undo() -> str:
    """Undo the last modeling operation."""
    r = _executor.call("undo", {})
    return _format_result(r)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_server(transport: str = "stdio") -> None:
    """Start the MCP server.

    Args:
        transport: "stdio" (default, for Claude Code) or "sse" (for web clients).
    """
    mcp.run(transport=transport)


if __name__ == "__main__":
    run_server()
