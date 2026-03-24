"""Spatial reasoning engine.

Provides geometric queries: inside/outside classification,
clearance/distance computation, bounding box, and intersection checks.
"""

from __future__ import annotations

from dataclasses import dataclass

from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.gp import gp_Pnt
from OCP.TopoDS import TopoDS_Shape

from next3d.core.schema import Vec3


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box."""

    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float

    @property
    def size(self) -> Vec3:
        return Vec3(
            x=self.x_max - self.x_min,
            y=self.y_max - self.y_min,
            z=self.z_max - self.z_min,
        )

    @property
    def center(self) -> Vec3:
        return Vec3(
            x=(self.x_min + self.x_max) / 2,
            y=(self.y_min + self.y_max) / 2,
            z=(self.z_min + self.z_max) / 2,
        )

    @property
    def diagonal(self) -> float:
        s = self.size
        return (s.x ** 2 + s.y ** 2 + s.z ** 2) ** 0.5

    def contains_point(self, p: Vec3) -> bool:
        return (
            self.x_min <= p.x <= self.x_max
            and self.y_min <= p.y <= self.y_max
            and self.z_min <= p.z <= self.z_max
        )

    def overlaps(self, other: BoundingBox) -> bool:
        return not (
            self.x_max < other.x_min
            or self.x_min > other.x_max
            or self.y_max < other.y_min
            or self.y_min > other.y_max
            or self.z_max < other.z_min
            or self.z_min > other.z_max
        )

    def to_dict(self) -> dict:
        return {
            "min": {"x": self.x_min, "y": self.y_min, "z": self.z_min},
            "max": {"x": self.x_max, "y": self.y_max, "z": self.z_max},
            "size": {"x": self.size.x, "y": self.size.y, "z": self.size.z},
        }


def bounding_box(shape: TopoDS_Shape) -> BoundingBox:
    """Compute the axis-aligned bounding box of a shape."""
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return BoundingBox(
        x_min=xmin, y_min=ymin, z_min=zmin,
        x_max=xmax, y_max=ymax, z_max=zmax,
    )


def point_in_solid(shape: TopoDS_Shape, point: Vec3, tolerance: float = 1e-6) -> str:
    """Classify a point relative to a solid.

    Returns:
        'inside', 'outside', or 'on_boundary'
    """
    classifier = BRepClass3d_SolidClassifier(shape, gp_Pnt(point.x, point.y, point.z), tolerance)
    state = classifier.State()
    # TopAbs_IN = 0, TopAbs_OUT = 1, TopAbs_ON = 2
    if state == 0:
        return "inside"
    elif state == 1:
        return "outside"
    else:
        return "on_boundary"


def minimum_distance(shape1: TopoDS_Shape, shape2: TopoDS_Shape) -> float:
    """Compute the minimum distance between two shapes (clearance)."""
    dist_calc = BRepExtrema_DistShapeShape(shape1, shape2)
    if dist_calc.IsDone():
        return dist_calc.Value()
    return float("inf")
