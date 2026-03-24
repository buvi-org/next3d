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

def add_hole(
    shape: TopoDS_Shape,
    center_x: float,
    center_y: float,
    diameter: float,
    depth: float | None = None,
    face_selector: str = ">Z",
) -> TopoDS_Shape:
    """Add a hole to an existing shape.

    Args:
        shape: The shape to modify.
        center_x: X position of hole center on the selected face.
        center_y: Y position of hole center on the selected face.
        diameter: Hole diameter (mm).
        depth: Hole depth. None = through-all.
        face_selector: CadQuery face selector string (default: top face ">Z").
    """
    wp = _wp_from_shape(shape)
    wp = wp.faces(face_selector).workplane().pushPoints([(center_x, center_y)])
    if depth is None:
        wp = wp.hole(diameter)
    else:
        wp = wp.hole(diameter, depth)
    return _to_shape(wp)


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
    """Add a counterbore hole."""
    wp = _wp_from_shape(shape)
    wp = wp.faces(face_selector).workplane().pushPoints([(center_x, center_y)])
    wp = wp.cboreHole(hole_diameter, cb_diameter, cb_depth, depth)
    return _to_shape(wp)


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
    """Add a countersink hole."""
    wp = _wp_from_shape(shape)
    wp = wp.faces(face_selector).workplane().pushPoints([(center_x, center_y)])
    wp = wp.cskHole(hole_diameter, cs_diameter, cs_angle, depth)
    return _to_shape(wp)


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
    if edge_selector:
        wp = wp.edges(edge_selector).chamfer(distance)
    else:
        wp = wp.edges().chamfer(distance)
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
