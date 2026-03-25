"""Modeling kernel — CadQuery wrapper for parametric 3D operations.

This is the bridge between AI tool calls and the OpenCascade geometry kernel.
Every operation takes explicit parameters and returns a TopoDS_Shape.

Design principle: AI calls functions with numbers, gets geometry back.
No interactive selection, no GUI concepts, no visual feedback needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cadquery as cq
from OCP.TopoDS import TopoDS_Shape


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_shape(wp: cq.Workplane) -> TopoDS_Shape:
    """Extract the underlying TopoDS_Shape from a CadQuery Workplane."""
    return wp.val().wrapped


def _wp_from_shape(shape: TopoDS_Shape) -> cq.Workplane:
    """Wrap an existing TopoDS_Shape in a CadQuery Workplane with solid context."""
    solid = cq.Solid(shape)
    return cq.Workplane(obj=solid)


# ---------------------------------------------------------------------------
# CREATE operations — make new geometry from scratch
# ---------------------------------------------------------------------------

def create_box(
    length: float,
    width: float,
    height: float,
    center: tuple[float, float, float] = (0, 0, 0),
    centered: bool = True,
) -> TopoDS_Shape:
    """Create a rectangular box.

    Args:
        length: X dimension (mm).
        width: Y dimension (mm).
        height: Z dimension (mm).
        center: Center point (x, y, z).
        centered: If True, center the box on the point. If False, corner at point.
    """
    wp = cq.Workplane("XY").transformed(offset=center).box(length, width, height, centered=(centered, centered, centered))
    return _to_shape(wp)


def create_cylinder(
    radius: float,
    height: float,
    center: tuple[float, float, float] = (0, 0, 0),
    axis: str = "Z",
) -> TopoDS_Shape:
    """Create a cylinder.

    Args:
        radius: Cylinder radius (mm).
        height: Cylinder height (mm).
        center: Center of the base circle.
        axis: Axis direction — "X", "Y", or "Z".
    """
    plane = {"X": "YZ", "Y": "XZ", "Z": "XY"}.get(axis.upper(), "XY")
    wp = cq.Workplane(plane).transformed(offset=center).cylinder(height, radius)
    return _to_shape(wp)


def create_sphere(
    radius: float,
    center: tuple[float, float, float] = (0, 0, 0),
) -> TopoDS_Shape:
    """Create a sphere."""
    wp = cq.Workplane("XY").transformed(offset=center).sphere(radius)
    return _to_shape(wp)


def create_cone(
    base_radius: float,
    top_radius: float,
    height: float,
    center: tuple[float, float, float] = (0, 0, 0),
) -> TopoDS_Shape:
    """Create a cone or truncated cone."""
    wp = (
        cq.Workplane("XY")
        .transformed(offset=center)
        .circle(base_radius)
        .workplane(offset=height)
        .circle(top_radius)
        .loft()
    )
    return _to_shape(wp)


def create_extrusion(
    points: Sequence[tuple[float, float]],
    height: float,
    center: tuple[float, float, float] = (0, 0, 0),
) -> TopoDS_Shape:
    """Create a solid by extruding a 2D polygon profile.

    Args:
        points: 2D polygon vertices [(x,y), ...]. Will be closed automatically.
        height: Extrusion height along Z.
        center: Offset applied to the sketch plane.
    """
    wp = cq.Workplane("XY").transformed(offset=center).polyline(list(points)).close().extrude(height)
    return _to_shape(wp)


# ---------------------------------------------------------------------------
# MODIFY operations — alter existing geometry
# ---------------------------------------------------------------------------

def _shape_has_solids(shape: TopoDS_Shape) -> bool:
    """Return True if the shape contains at least one solid."""
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID

    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    return explorer.More()


def _fix_shape(shape: TopoDS_Shape) -> TopoDS_Shape:
    """Run ShapeFix on a shape to repair orientation / normals.

    Operations like ``shell()`` can produce solids with inverted face
    normals (negative volume).  OCCT boolean operations treat such solids
    as empty, which causes downstream cuts (holes, pockets) to silently
    return an empty result.  ``ShapeFix_Shape`` corrects the orientation
    so that booleans succeed.
    """
    from OCP.ShapeFix import ShapeFix_Shape

    fixer = ShapeFix_Shape(shape)
    fixer.Perform()
    return fixer.Shape()


def _shape_needs_fix(shape: TopoDS_Shape) -> bool:
    """Return True if the shape has negative volume (inverted normals)."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass() < 0


def add_hole(
    shape: TopoDS_Shape,
    center_x: float,
    center_y: float,
    diameter: float,
    depth: float | None = None,
    face_selector: str = ">Z",
) -> TopoDS_Shape:
    """Add a hole to an existing shape.

    On shelled (hollow) bodies the ``shell()`` operation can produce a solid
    with inverted face normals (negative volume).  OCCT boolean operations
    interpret such shapes as empty, so a through-all hole silently returns
    zero geometry.  When we detect this situation we run ``ShapeFix_Shape``
    to repair the orientation before retrying the cut.

    Args:
        shape: The shape to modify.
        center_x: X position of hole center on the selected face.
        center_y: Y position of hole center on the selected face.
        diameter: Hole diameter (mm).
        depth: Hole depth. None = through-all.
        face_selector: CadQuery face selector string (default: top face ">Z").
    """

    def _do_hole(s: TopoDS_Shape) -> TopoDS_Shape:
        wp = _wp_from_shape(s)
        wp = wp.faces(face_selector).workplane().pushPoints([(center_x, center_y)])
        if depth is None:
            wp = wp.hole(diameter)
        else:
            wp = wp.hole(diameter, depth)
        return _to_shape(wp)

    # Fast path: try the hole directly.
    result = _do_hole(shape)
    if _shape_has_solids(result):
        return result

    # The result is empty — likely caused by inverted normals from a prior
    # shell operation.  Repair and retry.
    fixed = _fix_shape(shape)
    result = _do_hole(fixed)
    if _shape_has_solids(result):
        return result

    # Last resort: return whatever the first attempt produced so we don't
    # silently swallow errors for genuinely invalid operations.
    return _do_hole(shape)


def add_counterbore_hole(
    shape: TopoDS_Shape,
    center_x: float,
    center_y: float,
    hole_diameter: float,
    cb_diameter: float,
    cb_depth: float,
    depth: float | None = None,
    face_selector: str = ">Z",
) -> TopoDS_Shape:
    """Add a counterbore hole.

    Applies the same ShapeFix fallback as :func:`add_hole` to handle
    shelled bodies with inverted normals.
    """

    def _do(s: TopoDS_Shape) -> TopoDS_Shape:
        wp = _wp_from_shape(s)
        wp = wp.faces(face_selector).workplane().pushPoints([(center_x, center_y)])
        wp = wp.cboreHole(hole_diameter, cb_diameter, cb_depth, depth)
        return _to_shape(wp)

    result = _do(shape)
    if _shape_has_solids(result):
        return result
    fixed = _fix_shape(shape)
    result = _do(fixed)
    if _shape_has_solids(result):
        return result
    return _do(shape)


def add_countersink_hole(
    shape: TopoDS_Shape,
    center_x: float,
    center_y: float,
    hole_diameter: float,
    cs_diameter: float,
    cs_angle: float = 82.0,
    depth: float | None = None,
    face_selector: str = ">Z",
) -> TopoDS_Shape:
    """Add a countersink hole.

    Applies the same ShapeFix fallback as :func:`add_hole` to handle
    shelled bodies with inverted normals.
    """

    def _do(s: TopoDS_Shape) -> TopoDS_Shape:
        wp = _wp_from_shape(s)
        wp = wp.faces(face_selector).workplane().pushPoints([(center_x, center_y)])
        wp = wp.cskHole(hole_diameter, cs_diameter, cs_angle, depth)
        return _to_shape(wp)

    result = _do(shape)
    if _shape_has_solids(result):
        return result
    fixed = _fix_shape(shape)
    result = _do(fixed)
    if _shape_has_solids(result):
        return result
    return _do(shape)


def add_pocket(
    shape: TopoDS_Shape,
    center_x: float,
    center_y: float,
    length: float,
    width: float,
    depth: float,
    face_selector: str = ">Z",
) -> TopoDS_Shape:
    """Cut a rectangular pocket into a face."""
    wp = (
        _wp_from_shape(shape)
        .faces(face_selector)
        .workplane()
        .center(center_x, center_y)
        .rect(length, width)
        .cutBlind(-depth)
    )
    return _to_shape(wp)


def add_circular_pocket(
    shape: TopoDS_Shape,
    center_x: float,
    center_y: float,
    diameter: float,
    depth: float,
    face_selector: str = ">Z",
) -> TopoDS_Shape:
    """Cut a circular pocket into a face."""
    wp = (
        _wp_from_shape(shape)
        .faces(face_selector)
        .workplane()
        .center(center_x, center_y)
        .circle(diameter / 2)
        .cutBlind(-depth)
    )
    return _to_shape(wp)


def add_boss(
    shape: TopoDS_Shape,
    center_x: float,
    center_y: float,
    diameter: float,
    height: float,
    face_selector: str = ">Z",
) -> TopoDS_Shape:
    """Add a cylindrical boss (protrusion) on a face."""
    wp = (
        _wp_from_shape(shape)
        .faces(face_selector)
        .workplane()
        .center(center_x, center_y)
        .circle(diameter / 2)
        .extrude(height)
    )
    return _to_shape(wp)


def add_slot(
    shape: TopoDS_Shape,
    center_x: float,
    center_y: float,
    length: float,
    width: float,
    depth: float,
    angle: float = 0.0,
    face_selector: str = ">Z",
) -> TopoDS_Shape:
    """Cut a slot (rounded rectangle) into a face."""
    wp = (
        _wp_from_shape(shape)
        .faces(face_selector)
        .workplane()
        .center(center_x, center_y)
        .transformed(rotate=(0, 0, angle))
        .slot2D(length, width)
        .cutBlind(-depth)
    )
    return _to_shape(wp)


def add_fillet(
    shape: TopoDS_Shape,
    radius: float,
    edge_selector: str | None = None,
) -> TopoDS_Shape:
    """Add fillets (rounded edges).

    Args:
        shape: The shape to modify.
        radius: Fillet radius (mm).
        edge_selector: CadQuery edge selector. None = all edges.
    """
    wp = _wp_from_shape(shape)
    if edge_selector:
        wp = wp.edges(edge_selector).fillet(radius)
    else:
        wp = wp.edges().fillet(radius)
    return _to_shape(wp)


def add_chamfer(
    shape: TopoDS_Shape,
    distance: float,
    edge_selector: str | None = None,
) -> TopoDS_Shape:
    """Add chamfers (beveled edges).

    Args:
        shape: The shape to modify.
        distance: Chamfer distance (mm).
        edge_selector: CadQuery edge selector. None = all edges.
    """
    wp = _wp_from_shape(shape)
    try:
        if edge_selector:
            selected = wp.edges(edge_selector)
            n_edges = len(selected.vals())
            wp = selected.chamfer(distance)
        else:
            n_edges = len(wp.edges().vals())
            wp = wp.edges().chamfer(distance)
    except Exception as exc:
        msg = str(exc)
        # "BRep_API: command not done" is the typical OCCT error when chamfer
        # fails on complex edge topologies (e.g. shelled body edges shared
        # between inner and outer walls).
        if "command not done" in msg.lower() or "not done" in msg.lower():
            hint = (
                f"Chamfer (d={distance}) failed on {n_edges} edge(s) "
                f"matching '{edge_selector or 'all'}'. "
                "This commonly happens on shelled bodies where edges are "
                "shared between inner and outer walls, creating geometry "
                "too complex for the chamfer algorithm. "
                "Try selecting specific outer edges (e.g. use a more specific "
                "selector like '|Z' for vertical edges only) or apply the "
                "chamfer before shelling."
            )
            raise RuntimeError(hint) from exc
        raise
    return _to_shape(wp)


# ---------------------------------------------------------------------------
# BOOLEAN operations
# ---------------------------------------------------------------------------

def boolean_union(shape_a: TopoDS_Shape, shape_b: TopoDS_Shape) -> TopoDS_Shape:
    """Boolean union (add) of two shapes."""
    a = cq.Shape(shape_a)
    b = cq.Shape(shape_b)
    return a.fuse(b).wrapped


def boolean_cut(shape: TopoDS_Shape, tool: TopoDS_Shape) -> TopoDS_Shape:
    """Boolean cut (subtract tool from shape)."""
    a = cq.Shape(shape)
    b = cq.Shape(tool)
    return a.cut(b).wrapped


def boolean_intersect(shape_a: TopoDS_Shape, shape_b: TopoDS_Shape) -> TopoDS_Shape:
    """Boolean intersection of two shapes."""
    a = cq.Shape(shape_a)
    b = cq.Shape(shape_b)
    return a.intersect(b).wrapped


# ---------------------------------------------------------------------------
# TRANSFORM operations
# ---------------------------------------------------------------------------

def translate(
    shape: TopoDS_Shape,
    dx: float = 0,
    dy: float = 0,
    dz: float = 0,
) -> TopoDS_Shape:
    """Translate a shape."""
    s = cq.Shape(shape)
    return s.moved(cq.Location((dx, dy, dz))).wrapped


def rotate(
    shape: TopoDS_Shape,
    axis: tuple[float, float, float] = (0, 0, 1),
    angle_degrees: float = 0,
    center: tuple[float, float, float] = (0, 0, 0),
) -> TopoDS_Shape:
    """Rotate a shape around an axis through a center point."""
    wp = _wp_from_shape(shape)
    wp = wp.rotate(center, tuple(a + b for a, b in zip(center, axis)), angle_degrees)
    return _to_shape(wp)


def mirror(
    shape: TopoDS_Shape,
    plane: str = "XY",
) -> TopoDS_Shape:
    """Mirror a shape across a plane. Returns the mirrored copy."""
    s = cq.Shape(shape)
    return s.mirror(plane).wrapped


# ---------------------------------------------------------------------------
# PATTERN operations
# ---------------------------------------------------------------------------

def linear_pattern(
    shape: TopoDS_Shape,
    tool: TopoDS_Shape,
    direction: tuple[float, float, float],
    count: int,
    spacing: float,
) -> TopoDS_Shape:
    """Apply a tool shape in a linear pattern (e.g. row of holes).

    The tool is boolean-cut at each position along the direction vector.
    """
    result = shape
    dx, dy, dz = direction
    mag = math.sqrt(dx * dx + dy * dy + dz * dz)
    ux, uy, uz = dx / mag, dy / mag, dz / mag
    for i in range(count):
        offset = spacing * (i + 1)
        moved_tool = translate(tool, ux * offset, uy * offset, uz * offset)
        result = boolean_cut(result, moved_tool)
    return result


def circular_pattern(
    shape: TopoDS_Shape,
    tool: TopoDS_Shape,
    count: int,
    axis: tuple[float, float, float] = (0, 0, 1),
    center: tuple[float, float, float] = (0, 0, 0),
) -> TopoDS_Shape:
    """Apply a tool shape in a circular pattern (e.g. bolt circle)."""
    result = shape
    angle_step = 360.0 / count
    for i in range(1, count):
        moved_tool = rotate(tool, axis=axis, angle_degrees=angle_step * i, center=center)
        result = boolean_cut(result, moved_tool)
    # First instance (i=0) is already there via the tool's original position
    return result


# ---------------------------------------------------------------------------
# REVOLVE / SWEEP / LOFT — profile-based solid creation
# ---------------------------------------------------------------------------

def create_revolve(
    points: Sequence[tuple[float, float]],
    angle_degrees: float = 360.0,
    axis_origin: tuple[float, float] = (0, 0),
    axis_direction: tuple[float, float] = (0, 1),
    center: tuple[float, float, float] = (0, 0, 0),
) -> TopoDS_Shape:
    """Create a solid by revolving a 2D profile around an axis.

    The profile is defined in the XZ plane and revolved around a line
    defined by axis_origin and axis_direction (both in the XZ plane).

    Args:
        points: 2D profile vertices [(x, z), ...] in the XZ plane.
                Must be on one side of the axis. Auto-closed.
        angle_degrees: Rotation angle (360 = full revolution).
        axis_origin: Point on the revolution axis (x, z) in sketch plane.
        axis_direction: Direction of the revolution axis (x, z).
        center: 3D offset for the sketch plane.
    """
    # Use CadQuery Workplane.revolve() which handles the profile→solid pipeline
    # The XZ workplane maps: sketch X→world X, sketch Y→world Z
    wp = cq.Workplane("XZ").transformed(offset=center)
    wp = wp.polyline(list(points)).close()

    # axis_origin and axis_direction are in the 2D sketch plane
    # CadQuery revolve on a 2D workplane expects 2D (x, y) coords for axis
    ax_start = (axis_origin[0], axis_origin[1])
    ax_end = (
        axis_origin[0] + axis_direction[0],
        axis_origin[1] + axis_direction[1],
    )

    wp = wp.revolve(angle_degrees, axisStart=ax_start, axisEnd=ax_end)
    return _to_shape(wp)


def create_sweep(
    profile_points: Sequence[tuple[float, float]],
    path_points: Sequence[tuple[float, float, float]],
    center: tuple[float, float, float] = (0, 0, 0),
) -> TopoDS_Shape:
    """Create a solid by sweeping a 2D profile along a 3D path.

    Args:
        profile_points: 2D cross-section vertices [(x, y), ...]. Auto-closed.
        path_points: 3D path vertices [(x, y, z), ...]. At least 2 points.
        center: Offset applied to the path.
    """
    # Build the sweep path as a wire
    offset_path = [
        (p[0] + center[0], p[1] + center[1], p[2] + center[2])
        for p in path_points
    ]

    if len(offset_path) == 2:
        # Straight line path — build a wire via Workplane.spline (2 points = line)
        path_wp = cq.Workplane("XY").spline(
            [cq.Vector(*p) for p in offset_path]
        )
    else:
        # Spline path through 3D points
        path_wp = cq.Workplane("XY").spline(
            [cq.Vector(*p) for p in offset_path]
        )

    # Build profile on XY plane, then sweep along path
    # Profile is centered at origin, sweep places it at path start
    wp = cq.Workplane("XY")
    wp = wp.polyline(list(profile_points)).close()
    wp = wp.sweep(path_wp)
    return _to_shape(wp)


def create_loft(
    sections: Sequence[Sequence[tuple[float, float]]],
    heights: Sequence[float],
    ruled: bool = False,
    center: tuple[float, float, float] = (0, 0, 0),
) -> TopoDS_Shape:
    """Create a solid by lofting between cross-sections at different heights.

    Args:
        sections: List of 2D polygon profiles [(x, y), ...] for each section.
        heights: Z-height for each section. Must be same length as sections.
        ruled: If True, use ruled surfaces (straight lines between sections).
        center: Offset for the base.
    """
    if len(sections) != len(heights):
        raise ValueError("sections and heights must have the same length")

    # Build each section as a wire at its height
    wires = []
    for section, h in zip(sections, heights):
        wp = cq.Workplane("XY").transformed(
            offset=(center[0], center[1], center[2] + h)
        )
        wp = wp.polyline(list(section)).close()
        wires.append(wp.val())

    # Use OCC directly for reliable multi-wire loft
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    lofter = BRepOffsetAPI_ThruSections(True, ruled)
    for wire in wires:
        lofter.AddWire(wire.wrapped)
    lofter.Build()

    if not lofter.IsDone():
        raise RuntimeError("Loft operation failed")

    return lofter.Shape()


# ---------------------------------------------------------------------------
# SHELL / DRAFT — modify existing geometry
# ---------------------------------------------------------------------------

def add_shell(
    shape: TopoDS_Shape,
    thickness: float,
    face_selector: str = ">Z",
) -> TopoDS_Shape:
    """Hollow out a solid to a uniform wall thickness.

    The selected face(s) are removed (open), and the remaining walls
    are offset inward by the specified thickness.

    Args:
        shape: The solid to shell.
        thickness: Wall thickness in mm.
        face_selector: CadQuery face selector for face(s) to remove.
    """
    wp = _wp_from_shape(shape)
    wp = wp.faces(face_selector).shell(thickness)
    return _to_shape(wp)


def add_draft(
    shape: TopoDS_Shape,
    angle_degrees: float,
    face_selector: str = "#Z",
    pull_direction: tuple[float, float, float] = (0, 0, 1),
    plane_selector: str = "<Z",
) -> TopoDS_Shape:
    """Add draft (taper) to faces for mold release.

    Args:
        shape: The solid to draft.
        angle_degrees: Draft angle in degrees (typically 1-5° for molding).
        face_selector: Selector for faces to draft.  Use "#Z" for vertical
            faces (normal perpendicular to Z).  Note: "|Z" selects faces
            whose normal is *parallel* to Z (top/bottom) — not vertical faces.
        pull_direction: Mold pull direction as (dx, dy, dz).
        plane_selector: Selector for the neutral plane (parting surface).
    """
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
    from OCP.BRepOffsetAPI import BRepOffsetAPI_DraftAngle
    from OCP.Draft import (
        Draft_ErrorStatus,
        Draft_FaceRecomputation,
        Draft_EdgeRecomputation,
        Draft_VertexRecomputation,
    )
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS

    _DRAFT_STATUS_MSG = {
        Draft_FaceRecomputation: "face recomputation error",
        Draft_EdgeRecomputation: "edge recomputation error",
        Draft_VertexRecomputation: "vertex recomputation error",
    }

    # Use CadQuery to select faces
    wp = _wp_from_shape(shape)
    draft_faces = wp.faces(face_selector).vals()
    if not draft_faces:
        raise RuntimeError(
            f"No faces matched selector '{face_selector}'. "
            "Check that the selector is correct for the current shape."
        )

    neutral_plane_face = wp.faces(plane_selector).val()

    # Get the plane from the neutral face
    from OCP.BRep import BRep_Tool
    from OCP.GeomAdaptor import GeomAdaptor_Surface

    surf = BRep_Tool.Surface_s(neutral_plane_face.wrapped)
    adaptor = GeomAdaptor_Surface(surf)
    try:
        pln = adaptor.Plane()
    except Exception:
        raise RuntimeError(
            f"Neutral plane face (selector '{plane_selector}') is not planar. "
            "The draft neutral plane must be a flat face."
        )

    pull_dir = gp_Dir(*pull_direction)
    angle_rad = math.radians(angle_degrees)

    drafter = BRepOffsetAPI_DraftAngle(shape)

    # Add faces one at a time, checking for per-face failures
    failed_faces = []
    added_faces = 0
    for i, face in enumerate(draft_faces):
        try:
            drafter.Add(face.wrapped, pull_dir, angle_rad, pln)
            if not drafter.AddDone():
                status = drafter.Status()
                reason = _DRAFT_STATUS_MSG.get(status, "unknown error")
                failed_faces.append((i, reason))
                drafter.Remove(face.wrapped)
            else:
                added_faces += 1
        except Exception as exc:
            failed_faces.append((i, str(exc) or "OCCT exception"))

    if added_faces == 0:
        reasons = "; ".join(f"face {i}: {r}" for i, r in failed_faces)
        raise RuntimeError(
            f"Draft ({angle_degrees}°) failed on all {len(draft_faces)} face(s) "
            f"matching '{face_selector}'. Errors: {reasons}. "
            "This commonly happens on shelled or hollow bodies where the "
            "thin-walled faces cannot be tapered — the offset geometry "
            "self-intersects. Apply draft before shelling, or draft only "
            "the outer faces of a solid body."
        )

    try:
        drafter.Build()
    except Exception as exc:
        n_total = len(draft_faces)
        raise RuntimeError(
            f"Draft ({angle_degrees}°) build failed after adding "
            f"{added_faces}/{n_total} face(s): {type(exc).__name__}. "
            + (
                f"{len(failed_faces)} face(s) were skipped due to errors. "
                if failed_faces else ""
            )
            + "This commonly happens on shelled or hollow bodies where "
            "the thin-walled faces cannot be tapered. "
            "Apply draft before shelling, or use a smaller angle."
        ) from exc

    if not drafter.IsDone():
        status = drafter.Status()
        reason = _DRAFT_STATUS_MSG.get(status, "unknown error")
        n_total = len(draft_faces)
        raise RuntimeError(
            f"Draft ({angle_degrees}°) build failed after adding "
            f"{added_faces}/{n_total} face(s): {reason}. "
            + (
                f"{len(failed_faces)} face(s) were skipped due to errors. "
                if failed_faces else ""
            )
            + "This commonly happens on shelled or hollow bodies. "
            "Apply draft before shelling, or use a smaller angle."
        )

    return drafter.Shape()


# ---------------------------------------------------------------------------
# EXPORT — STL and 3MF mesh formats
# ---------------------------------------------------------------------------

def check_interference(
    shape_a: TopoDS_Shape,
    shape_b: TopoDS_Shape,
) -> dict:
    """Check if two shapes interfere (collide).

    Returns dict with interferes (bool), interference_volume, and min_clearance.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape

    result: dict = {"interferes": False, "interference_volume_mm3": 0.0, "min_clearance_mm": 0.0}

    # Check intersection volume
    common = BRepAlgoAPI_Common(shape_a, shape_b)
    if common.IsDone():
        common_shape = common.Shape()
        if not common_shape.IsNull():
            props = GProp_GProps()
            BRepGProp.VolumeProperties_s(common_shape, props)
            vol = props.Mass()
            if vol > 1e-6:  # tolerance
                result["interferes"] = True
                result["interference_volume_mm3"] = round(vol, 4)
                return result

    # No interference — compute min clearance
    dist_calc = BRepExtrema_DistShapeShape(shape_a, shape_b)
    if dist_calc.IsDone() and dist_calc.NbSolution() > 0:
        result["min_clearance_mm"] = round(dist_calc.Value(), 4)

    return result


def export_stl(
    shape: TopoDS_Shape,
    path: str,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
) -> None:
    """Export shape as STL (tessellated mesh) for 3D printing.

    Args:
        shape: The shape to export.
        path: Output STL file path.
        linear_deflection: Max chord deviation in mm (lower = finer mesh).
        angular_deflection: Max angle deviation in radians.
    """
    from OCP.StlAPI import StlAPI_Writer
    from OCP.BRepMesh import BRepMesh_IncrementalMesh

    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection)
    mesh.Perform()

    writer = StlAPI_Writer()
    writer.Write(shape, path)


def export_3mf(
    shape: TopoDS_Shape,
    path: str,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
) -> None:
    """Export shape as 3MF for 3D printing (via STL intermediary + lib3mf).

    Falls back to STL export if lib3mf is not available.

    Args:
        shape: The shape to export.
        path: Output 3MF file path.
        linear_deflection: Max chord deviation in mm.
        angular_deflection: Max angle deviation in radians.
    """
    import tempfile
    import os

    # First tessellate and export as STL
    stl_path = tempfile.mktemp(suffix=".stl")
    try:
        export_stl(shape, stl_path, linear_deflection, angular_deflection)

        try:
            import lib3mf
            wrapper = lib3mf.Wrapper()
            model = wrapper.CreateModel()
            reader = model.QueryReader("stl")
            reader.ReadFromFile(stl_path)
            writer = model.QueryWriter("3mf")
            writer.WriteToFile(path)
        except ImportError:
            # Fallback: use CadQuery's built-in if available
            import cadquery as cq
            solid = cq.Solid(shape)
            cq.exporters.export(cq.Workplane(obj=solid), path, exportType="3MF")
    finally:
        if os.path.exists(stl_path):
            os.unlink(stl_path)


def render_png(
    shape: TopoDS_Shape,
    path: str,
    width: int = 800,
    height: int = 600,
) -> None:
    """Render shape to PNG image for visual feedback.

    Uses CadQuery's SVG export as base, then converts to PNG if
    cairosvg is available, otherwise saves as SVG.

    Args:
        shape: The shape to render.
        path: Output image file path (.png or .svg).
        width: Image width in pixels.
        height: Image height in pixels.
    """
    import tempfile
    import cadquery as cq

    solid = cq.Solid(shape)
    wp = cq.Workplane(obj=solid)

    if path.endswith(".svg"):
        cq.exporters.export(wp, path, exportType="SVG")
        return

    # For PNG: export SVG to temp file, then convert
    with tempfile.NamedTemporaryFile(suffix=".svg", mode="w", delete=False) as tmp:
        svg_tmp = tmp.name
    cq.exporters.export(wp, svg_tmp, exportType="SVG")

    try:
        with open(svg_tmp, "r") as f:
            svg_content = f.read()
    finally:
        import os
        os.unlink(svg_tmp)

    try:
        import cairosvg
        cairosvg.svg2png(
            bytestring=svg_content.encode("utf-8"),
            write_to=path,
            output_width=width,
            output_height=height,
        )
    except ImportError:
        # Fallback: save as SVG with .png extension note
        svg_path = path.rsplit(".", 1)[0] + ".svg"
        with open(svg_path, "w") as f:
            f.write(svg_content)
        raise ImportError(
            f"cairosvg not installed — saved SVG to {svg_path}. "
            "Install cairosvg for PNG: pip install cairosvg"
        )
