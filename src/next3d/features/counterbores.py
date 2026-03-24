"""Counterbore and countersink feature recognition.

Detection strategy:
- Counterbore: Two concentric cylindrical faces with different radii,
  sharing a planar step face between them. Both are through/blind holes.
- Countersink: A conical face between a cylindrical hole and a planar top face.
"""

from __future__ import annotations

import math

from next3d.core.identity import feature_id
from next3d.core.schema import (
    AdjacencyEdge,
    EdgeData,
    EdgeRelationType,
    FaceData,
    FeatureData,
    FeatureType,
    SurfaceType,
    Vec3,
)


def _dot(a: Vec3, b: Vec3) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def _magnitude(v: Vec3) -> float:
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(
        x=a.y * b.z - a.z * b.y,
        y=a.z * b.x - a.x * b.z,
        z=a.x * b.y - a.y * b.x,
    )


def _axes_colinear(a1: Vec3, a2: Vec3, c1: Vec3, c2: Vec3, angle_tol: float = 5.0, dist_tol: float = 0.5) -> bool:
    """Check if two axis-bearing faces are colinear (same axis line)."""
    m1 = _magnitude(a1)
    m2 = _magnitude(a2)
    if m1 < 1e-12 or m2 < 1e-12:
        return False
    n1 = Vec3(x=a1.x / m1, y=a1.y / m1, z=a1.z / m1)
    n2 = Vec3(x=a2.x / m2, y=a2.y / m2, z=a2.z / m2)

    # Axes must be parallel
    if abs(_dot(n1, n2)) < math.cos(math.radians(angle_tol)):
        return False

    # Centroids must be close to the same axis line
    diff = Vec3(x=c2.x - c1.x, y=c2.y - c1.y, z=c2.z - c1.z)
    cross = _cross(n1, diff)
    lateral = _magnitude(cross)
    return lateral < dist_tol


class CounterboreRecognizer:
    """Recognizes counterbore features (concentric stepped holes).

    Pattern: Two coaxial cylindrical faces with different radii,
    connected by a planar step (annular) face.
    """

    def recognize(
        self,
        faces: list[FaceData],
        edges: list[EdgeData],
        adjacency: list[AdjacencyEdge],
    ) -> list[FeatureData]:
        features: list[FeatureData] = []
        face_lookup = {f.persistent_id: f for f in faces}

        adj_lookup: dict[str, list[str]] = {}
        for adj in adjacency:
            if adj.edge_type != EdgeRelationType.ADJACENT:
                continue
            adj_lookup.setdefault(adj.source_id, []).append(adj.target_id)
            adj_lookup.setdefault(adj.target_id, []).append(adj.source_id)

        cylinders = [f for f in faces if f.surface_type == SurfaceType.CYLINDER and f.radius is not None and f.axis is not None]
        used: set[str] = set()

        for i, c1 in enumerate(cylinders):
            if c1.persistent_id in used:
                continue
            for j, c2 in enumerate(cylinders):
                if j <= i or c2.persistent_id in used:
                    continue
                if c1.radius == c2.radius:
                    continue  # need different radii

                if not _axes_colinear(c1.axis, c2.axis, c1.centroid, c2.centroid):
                    continue

                # Check for a shared planar step face between them
                c1_adj = set(adj_lookup.get(c1.persistent_id, []))
                c2_adj = set(adj_lookup.get(c2.persistent_id, []))
                shared_adj = c1_adj & c2_adj

                step_faces = [
                    face_lookup[fid] for fid in shared_adj
                    if fid in face_lookup and face_lookup[fid].surface_type == SurfaceType.PLANE
                ]

                if not step_faces:
                    continue

                # Found a counterbore
                small = c1 if c1.radius < c2.radius else c2
                large = c2 if c1.radius < c2.radius else c1
                step = step_faces[0]

                cb_faces = [small.persistent_id, large.persistent_id, step.persistent_id]
                fid = feature_id(FeatureType.COUNTERBORE.value, cb_faces)
                features.append(
                    FeatureData(
                        persistent_id=fid,
                        feature_type=FeatureType.COUNTERBORE,
                        face_ids=cb_faces,
                        edge_ids=small.edge_ids + large.edge_ids,
                        parameters={
                            "hole_diameter": small.radius * 2,
                            "bore_diameter": large.radius * 2,
                        },
                        axis=small.axis,
                        description=(
                            f"Counterbore, hole d={small.radius * 2:.2f}mm, "
                            f"bore d={large.radius * 2:.2f}mm"
                        ),
                    )
                )
                used.update(cb_faces)

        return features


class CountersinkRecognizer:
    """Recognizes countersink features (conical transitions at holes).

    Pattern: A conical face adjacent to a cylindrical face (hole)
    and a planar face (top surface).
    """

    def recognize(
        self,
        faces: list[FaceData],
        edges: list[EdgeData],
        adjacency: list[AdjacencyEdge],
    ) -> list[FeatureData]:
        features: list[FeatureData] = []
        face_lookup = {f.persistent_id: f for f in faces}

        adj_lookup: dict[str, list[str]] = {}
        for adj in adjacency:
            if adj.edge_type != EdgeRelationType.ADJACENT:
                continue
            adj_lookup.setdefault(adj.source_id, []).append(adj.target_id)
            adj_lookup.setdefault(adj.target_id, []).append(adj.source_id)

        used: set[str] = set()

        for face in faces:
            if face.persistent_id in used:
                continue
            if face.surface_type != SurfaceType.CONE:
                continue

            adjacent_ids = adj_lookup.get(face.persistent_id, [])
            adjacent_faces = [face_lookup[fid] for fid in adjacent_ids if fid in face_lookup]

            # Find an adjacent cylinder (the hole)
            cyl_neighbors = [af for af in adjacent_faces if af.surface_type == SurfaceType.CYLINDER and af.radius is not None]
            # Find an adjacent plane (the top surface)
            plane_neighbors = [af for af in adjacent_faces if af.surface_type == SurfaceType.PLANE]

            if not cyl_neighbors or not plane_neighbors:
                continue

            cyl = cyl_neighbors[0]
            cs_faces = [face.persistent_id, cyl.persistent_id]

            fid = feature_id(FeatureType.COUNTERSINK.value, cs_faces)
            features.append(
                FeatureData(
                    persistent_id=fid,
                    feature_type=FeatureType.COUNTERSINK,
                    face_ids=cs_faces,
                    edge_ids=face.edge_ids + cyl.edge_ids,
                    parameters={
                        "hole_diameter": cyl.radius * 2,
                    },
                    axis=cyl.axis,
                    description=f"Countersink, hole d={cyl.radius * 2:.2f}mm",
                )
            )
            used.update(cs_faces)

        return features
