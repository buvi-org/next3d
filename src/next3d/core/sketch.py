"""2D sketch + constraints — the foundation for profile-based 3D modeling.

A Sketch holds 2D entities (lines, arcs, circles, rectangles, polygons,
splines) and constraints (geometric + dimensional). Entities are stored as
simple data structures until extrude/revolve is called, at which point they
are converted to CadQuery wires for solid creation.

Constraints are declarative — they document design intent for AI agents
but do not solve (no constraint solver). They are stored for inspection,
serialization, and future solving.

Usage:
    sketch = Sketch(plane="XY")
    l1 = sketch.add_line(0, 0, 10, 0)
    l2 = sketch.add_line(10, 0, 10, 10)
    l3 = sketch.add_line(10, 10, 0, 10)
    l4 = sketch.add_line(0, 10, 0, 0)
    sketch.add_constraint("horizontal", l1)
    sketch.add_constraint("vertical", l2)
    shape = sketch.extrude(5.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence
from uuid import uuid4

import cadquery as cq
from OCP.TopoDS import TopoDS_Shape


# ---------------------------------------------------------------------------
# Entity types
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    """Supported 2D sketch entity types."""
    LINE = "line"
    ARC = "arc"
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    POLYGON = "polygon"
    SPLINE = "spline"


@dataclass
class SketchEntity:
    """A single 2D sketch entity."""

    entity_id: str
    entity_type: EntityType
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "params": self.params,
        }


# ---------------------------------------------------------------------------
# Constraint types
# ---------------------------------------------------------------------------

class ConstraintType(str, Enum):
    """Supported sketch constraints."""
    # Geometric
    COINCIDENT = "coincident"
    TANGENT = "tangent"
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"
    EQUAL_LENGTH = "equal_length"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    FIXED = "fixed"
    CONCENTRIC = "concentric"
    # Dimensional
    DISTANCE = "distance"
    ANGLE = "angle"
    RADIUS = "radius"
    DIAMETER = "diameter"


@dataclass
class SketchConstraint:
    """A constraint between one or two sketch entities."""

    constraint_id: str
    constraint_type: ConstraintType
    entity_a: str
    entity_b: str | None = None
    value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "constraint_id": self.constraint_id,
            "constraint_type": self.constraint_type.value,
            "entity_a": self.entity_a,
        }
        if self.entity_b is not None:
            result["entity_b"] = self.entity_b
        if self.value is not None:
            result["value"] = self.value
        return result


# ---------------------------------------------------------------------------
# Sketch
# ---------------------------------------------------------------------------

class Sketch:
    """2D sketch with entities and constraints.

    Entities are stored as data until extrude/revolve converts them to
    CadQuery geometry.

    Args:
        plane: Sketch plane — "XY", "XZ", or "YZ".
    """

    def __init__(self, plane: str = "XY") -> None:
        if plane not in ("XY", "XZ", "YZ"):
            raise ValueError(f"Invalid plane: {plane}. Use XY, XZ, or YZ.")
        self.plane = plane
        self.entities: list[SketchEntity] = []
        self.constraints: list[SketchConstraint] = []

    def _new_id(self) -> str:
        return uuid4().hex[:8]

    # ------------------------------------------------------------------
    # Add entities
    # ------------------------------------------------------------------

    def add_line(self, x1: float, y1: float, x2: float, y2: float) -> str:
        """Add a line segment. Returns the entity ID."""
        eid = self._new_id()
        self.entities.append(SketchEntity(
            entity_id=eid,
            entity_type=EntityType.LINE,
            params={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        ))
        return eid

    def add_arc(
        self,
        cx: float,
        cy: float,
        radius: float,
        start_angle: float,
        end_angle: float,
    ) -> str:
        """Add a circular arc. Angles in degrees. Returns the entity ID."""
        eid = self._new_id()
        self.entities.append(SketchEntity(
            entity_id=eid,
            entity_type=EntityType.ARC,
            params={
                "cx": cx, "cy": cy, "radius": radius,
                "start_angle": start_angle, "end_angle": end_angle,
            },
        ))
        return eid

    def add_circle(self, cx: float, cy: float, radius: float) -> str:
        """Add a full circle. Returns the entity ID."""
        eid = self._new_id()
        self.entities.append(SketchEntity(
            entity_id=eid,
            entity_type=EntityType.CIRCLE,
            params={"cx": cx, "cy": cy, "radius": radius},
        ))
        return eid

    def add_rect(self, cx: float, cy: float, width: float, height: float) -> str:
        """Add a rectangle centered at (cx, cy). Returns the entity ID."""
        eid = self._new_id()
        self.entities.append(SketchEntity(
            entity_id=eid,
            entity_type=EntityType.RECTANGLE,
            params={"cx": cx, "cy": cy, "width": width, "height": height},
        ))
        return eid

    def add_polygon(self, points: Sequence[tuple[float, float]]) -> str:
        """Add a closed polygon from a list of (x, y) vertices. Returns the entity ID."""
        if len(points) < 3:
            raise ValueError("Polygon needs at least 3 points.")
        eid = self._new_id()
        self.entities.append(SketchEntity(
            entity_id=eid,
            entity_type=EntityType.POLYGON,
            params={"points": [list(p) for p in points]},
        ))
        return eid

    def add_spline(self, points: Sequence[tuple[float, float]]) -> str:
        """Add a spline through points. Returns the entity ID."""
        if len(points) < 2:
            raise ValueError("Spline needs at least 2 points.")
        eid = self._new_id()
        self.entities.append(SketchEntity(
            entity_id=eid,
            entity_type=EntityType.SPLINE,
            params={"points": [list(p) for p in points]},
        ))
        return eid

    # ------------------------------------------------------------------
    # Add constraints
    # ------------------------------------------------------------------

    def add_constraint(
        self,
        constraint_type: str,
        entity_a: str,
        entity_b: str | None = None,
        value: float | None = None,
    ) -> str:
        """Add a constraint. Returns the constraint ID.

        Args:
            constraint_type: One of the ConstraintType values.
            entity_a: ID of the first entity.
            entity_b: ID of the second entity (optional, depends on type).
            value: Dimensional value (for distance, angle, radius, diameter).
        """
        # Validate constraint type
        try:
            ct = ConstraintType(constraint_type)
        except ValueError:
            valid = ", ".join(c.value for c in ConstraintType)
            raise ValueError(f"Unknown constraint type: {constraint_type}. Valid: {valid}")

        # Validate entity references
        entity_ids = {e.entity_id for e in self.entities}
        if entity_a not in entity_ids:
            raise ValueError(f"Entity '{entity_a}' not found in sketch.")
        if entity_b is not None and entity_b not in entity_ids:
            raise ValueError(f"Entity '{entity_b}' not found in sketch.")

        cid = self._new_id()
        self.constraints.append(SketchConstraint(
            constraint_id=cid,
            constraint_type=ct,
            entity_a=entity_a,
            entity_b=entity_b,
            value=value,
        ))
        return cid

    # ------------------------------------------------------------------
    # Build CadQuery wire from entities
    # ------------------------------------------------------------------

    def _entity_bbox(self, entity: SketchEntity) -> tuple[float, float, float, float]:
        """Return (min_x, min_y, max_x, max_y) bounding box for an entity."""
        p = entity.params
        if entity.entity_type == EntityType.CIRCLE:
            return (p["cx"] - p["radius"], p["cy"] - p["radius"],
                    p["cx"] + p["radius"], p["cy"] + p["radius"])
        if entity.entity_type == EntityType.RECTANGLE:
            hw, hh = p["width"] / 2, p["height"] / 2
            return (p["cx"] - hw, p["cy"] - hh, p["cx"] + hw, p["cy"] + hh)
        if entity.entity_type == EntityType.POLYGON:
            pts = p["points"]
            xs = [pt[0] for pt in pts]
            ys = [pt[1] for pt in pts]
            return (min(xs), min(ys), max(xs), max(ys))
        if entity.entity_type == EntityType.LINE:
            return (min(p["x1"], p["x2"]), min(p["y1"], p["y2"]),
                    max(p["x1"], p["x2"]), max(p["y1"], p["y2"]))
        if entity.entity_type == EntityType.ARC:
            cx, cy, r = p["cx"], p["cy"], p["radius"]
            return (cx - r, cy - r, cx + r, cy + r)
        # Spline / fallback
        if "points" in p:
            pts = p["points"]
            xs = [pt[0] for pt in pts]
            ys = [pt[1] for pt in pts]
            return (min(xs), min(ys), max(xs), max(ys))
        return (0, 0, 0, 0)

    def _entity_area(self, entity: SketchEntity) -> float:
        """Approximate area enclosed by an entity (for outer/inner classification)."""
        p = entity.params
        if entity.entity_type == EntityType.CIRCLE:
            return math.pi * p["radius"] ** 2
        if entity.entity_type == EntityType.RECTANGLE:
            return p["width"] * p["height"]
        if entity.entity_type == EntityType.POLYGON:
            # Shoelace formula
            pts = p["points"]
            n = len(pts)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += pts[i][0] * pts[j][1]
                area -= pts[j][0] * pts[i][1]
            return abs(area) / 2
        # Bounding-box area as rough fallback
        bb = self._entity_bbox(entity)
        return (bb[2] - bb[0]) * (bb[3] - bb[1])

    @staticmethod
    def _bbox_contains(outer: tuple[float, float, float, float],
                       inner: tuple[float, float, float, float],
                       tol: float = 1e-9) -> bool:
        """Return True if *outer* bounding box fully contains *inner*."""
        return (outer[0] - tol <= inner[0] and outer[1] - tol <= inner[1] and
                outer[2] + tol >= inner[2] and outer[3] + tol >= inner[3])

    def _classify_profiles(self) -> tuple[list[SketchEntity], list[SketchEntity]]:
        """Split entities into (outer_entities, inner_entities).

        An entity is considered *inner* if:
        - It is a closed profile (CIRCLE, RECTANGLE, POLYGON) and
        - Its bounding box is fully inside another closed profile's bounding box.

        Lines and arcs that form the outer contour are always outer.
        """
        closed_types = {EntityType.CIRCLE, EntityType.RECTANGLE, EntityType.POLYGON}
        open_types = {EntityType.LINE, EntityType.ARC, EntityType.SPLINE}

        # Separate closed standalone profiles from open/wire entities
        closed_entities = [e for e in self.entities if e.entity_type in closed_types]
        open_entities = [e for e in self.entities if e.entity_type in open_types]

        # If 0 or 1 closed entities, no inner profiles possible
        if len(closed_entities) <= 1 and not open_entities:
            return (self.entities, [])
        if len(closed_entities) == 0:
            return (self.entities, [])

        # When we have open entities (lines) plus closed entities, the lines
        # form the outer wire and closed entities inside are inner profiles.
        if open_entities:
            # Lines form the outer contour — compute its bounding box
            outer_bb = self._entity_bbox(open_entities[0])
            for e in open_entities[1:]:
                bb = self._entity_bbox(e)
                outer_bb = (min(outer_bb[0], bb[0]), min(outer_bb[1], bb[1]),
                            max(outer_bb[2], bb[2]), max(outer_bb[3], bb[3]))

            inner = []
            outer_closed = []
            for e in closed_entities:
                if self._bbox_contains(outer_bb, self._entity_bbox(e)):
                    inner.append(e)
                else:
                    outer_closed.append(e)
            return (open_entities + outer_closed, inner)

        # All closed entities — the largest area is the outer profile
        sorted_by_area = sorted(closed_entities, key=self._entity_area, reverse=True)
        outer_entity = sorted_by_area[0]
        outer_bb = self._entity_bbox(outer_entity)

        inner = []
        extra_outer = []
        for e in sorted_by_area[1:]:
            if self._bbox_contains(outer_bb, self._entity_bbox(e)):
                inner.append(e)
            else:
                extra_outer.append(e)

        return ([outer_entity] + extra_outer, inner)

    def _build_workplane(self, entities: list[SketchEntity] | None = None) -> cq.Workplane:
        """Convert sketch entities into a CadQuery Workplane with a closed wire/face.

        Strategy:
        - If there is a single circle, use .circle()
        - If there is a single rectangle, use .rect()
        - If there is a single polygon, use .polyline().close()
        - For lines, chain them as a polyline (collect endpoints)
        - Mixed entity types: build wire from individual segments

        Args:
            entities: Subset of entities to build. Defaults to all entities.
        """
        ents = entities if entities is not None else self.entities
        if not ents:
            raise ValueError("Sketch has no entities. Add lines, circles, etc. first.")

        wp = cq.Workplane(self.plane)

        # Single-entity fast paths
        if len(ents) == 1:
            e = ents[0]
            p = e.params

            if e.entity_type == EntityType.CIRCLE:
                return wp.center(p["cx"], p["cy"]).circle(p["radius"])

            if e.entity_type == EntityType.RECTANGLE:
                return wp.center(p["cx"], p["cy"]).rect(p["width"], p["height"])

            if e.entity_type == EntityType.POLYGON:
                pts = [tuple(pt) for pt in p["points"]]
                return wp.polyline(pts).close()

        # Multiple entities — check if all lines (common case)
        all_lines = all(e.entity_type == EntityType.LINE for e in ents)
        if all_lines:
            # Build polyline from line endpoints
            first = ents[0].params
            wp = wp.moveTo(first["x1"], first["y1"])
            for e in ents:
                p = e.params
                wp = wp.lineTo(p["x2"], p["y2"])
            wp = wp.close()
            return wp

        # Mixed entities — build wire segment by segment
        first_entity = ents[0]
        if first_entity.entity_type == EntityType.LINE:
            p = first_entity.params
            wp = wp.moveTo(p["x1"], p["y1"])
        else:
            wp = wp.moveTo(0, 0)

        for e in ents:
            p = e.params
            if e.entity_type == EntityType.LINE:
                wp = wp.lineTo(p["x2"], p["y2"])
            elif e.entity_type == EntityType.ARC:
                # Convert arc to three-point arc
                cx, cy = p["cx"], p["cy"]
                r = p["radius"]
                sa = math.radians(p["start_angle"])
                ea = math.radians(p["end_angle"])
                ma = (sa + ea) / 2
                mid_x = cx + r * math.cos(ma)
                mid_y = cy + r * math.sin(ma)
                end_x = cx + r * math.cos(ea)
                end_y = cy + r * math.sin(ea)
                wp = wp.threePointArc((mid_x, mid_y), (end_x, end_y))
            elif e.entity_type == EntityType.CIRCLE:
                return cq.Workplane(self.plane).center(p["cx"], p["cy"]).circle(p["radius"])
            elif e.entity_type == EntityType.RECTANGLE:
                return cq.Workplane(self.plane).center(p["cx"], p["cy"]).rect(p["width"], p["height"])
            elif e.entity_type == EntityType.POLYGON:
                pts = [tuple(pt) for pt in p["points"]]
                return cq.Workplane(self.plane).polyline(pts).close()
            elif e.entity_type == EntityType.SPLINE:
                pts = [tuple(pt) for pt in p["points"]]
                wp = wp.spline(pts)

        wp = wp.close()
        return wp

    # ------------------------------------------------------------------
    # Solid creation
    # ------------------------------------------------------------------

    def extrude(self, height: float) -> TopoDS_Shape:
        """Extrude the sketch profile to create a solid.

        If the sketch contains inner profiles (e.g. circles inside a rectangle),
        they are subtracted from the extrusion to create holes/cutouts.

        Args:
            height: Extrusion height along the sketch plane normal.

        Returns:
            TopoDS_Shape of the extruded solid.
        """
        outer_entities, inner_entities = self._classify_profiles()

        # Extrude the outer profile
        wp = self._build_workplane(outer_entities)
        wp = wp.extrude(height)
        result_shape = wp.val().wrapped

        # Subtract each inner profile extrusion to create holes
        if inner_entities:
            from next3d.modeling.kernel import boolean_cut
            for inner_e in inner_entities:
                inner_wp = self._build_workplane([inner_e])
                inner_wp = inner_wp.extrude(height)
                inner_shape = inner_wp.val().wrapped
                result_shape = boolean_cut(result_shape, inner_shape)

        return result_shape

    def revolve(
        self,
        angle_degrees: float = 360.0,
        axis_origin: tuple[float, float] = (0, 0),
        axis_direction: tuple[float, float] = (0, 1),
    ) -> TopoDS_Shape:
        """Revolve the sketch profile around an axis to create a solid.

        Args:
            angle_degrees: Revolution angle (360 = full revolution).
            axis_origin: Point on the revolution axis (in sketch plane coords).
            axis_direction: Direction of the revolution axis (in sketch plane coords).

        Returns:
            TopoDS_Shape of the revolved solid.
        """
        wp = self._build_workplane()
        ax_start = (axis_origin[0], axis_origin[1])
        ax_end = (
            axis_origin[0] + axis_direction[0],
            axis_origin[1] + axis_direction[1],
        )
        wp = wp.revolve(angle_degrees, axisStart=ax_start, axisEnd=ax_end)
        return wp.val().wrapped

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize sketch state for inspection/export."""
        return {
            "plane": self.plane,
            "entity_count": len(self.entities),
            "constraint_count": len(self.constraints),
            "entities": [e.to_dict() for e in self.entities],
            "constraints": [c.to_dict() for c in self.constraints],
        }

    def get_entity(self, entity_id: str) -> SketchEntity:
        """Look up an entity by ID."""
        for e in self.entities:
            if e.entity_id == entity_id:
                return e
        raise ValueError(f"Entity '{entity_id}' not found in sketch.")
