"""Geometric relationship detection between topological entities.

Detects: parallel, perpendicular, concentric, coaxial, coplanar, tangent
relationships between faces and edges.
"""

from __future__ import annotations

import math

from next3d.core.schema import (
    AdjacencyEdge,
    EdgeRelationType,
    FaceData,
    Relationship,
    RelationshipType,
    SurfaceType,
    Vec3,
)


def _dot(a: Vec3, b: Vec3) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def _magnitude(v: Vec3) -> float:
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def _normalize(v: Vec3) -> Vec3:
    m = _magnitude(v)
    if m < 1e-12:
        return v
    return Vec3(x=v.x / m, y=v.y / m, z=v.z / m)


def _distance(a: Vec3, b: Vec3) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(
        x=a.y * b.z - a.z * b.y,
        y=a.z * b.x - a.x * b.z,
        z=a.x * b.y - a.y * b.x,
    )


# ---------------------------------------------------------------------------
# Relationship detectors
# ---------------------------------------------------------------------------

_ANGLE_TOL = 3.0  # degrees


def detect_parallel_faces(faces: list[FaceData]) -> list[Relationship]:
    """Detect parallel planar faces (normals aligned or anti-aligned)."""
    results: list[Relationship] = []
    planes = [f for f in faces if f.surface_type == SurfaceType.PLANE and f.normal is not None]

    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            n1 = _normalize(planes[i].normal)
            n2 = _normalize(planes[j].normal)
            cos_angle = abs(_dot(n1, n2))
            if cos_angle > math.cos(math.radians(_ANGLE_TOL)):
                # Compute offset distance (projection of centroid difference onto normal)
                diff = Vec3(
                    x=planes[j].centroid.x - planes[i].centroid.x,
                    y=planes[j].centroid.y - planes[i].centroid.y,
                    z=planes[j].centroid.z - planes[i].centroid.z,
                )
                offset = abs(_dot(diff, n1))
                results.append(
                    Relationship(
                        source_id=planes[i].persistent_id,
                        target_id=planes[j].persistent_id,
                        relationship_type=RelationshipType.PARALLEL,
                        parameters={"offset": round(offset, 6)},
                    )
                )
    return results


def detect_perpendicular_faces(faces: list[FaceData]) -> list[Relationship]:
    """Detect perpendicular planar faces."""
    results: list[Relationship] = []
    planes = [f for f in faces if f.surface_type == SurfaceType.PLANE and f.normal is not None]

    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            n1 = _normalize(planes[i].normal)
            n2 = _normalize(planes[j].normal)
            cos_angle = abs(_dot(n1, n2))
            if cos_angle < math.sin(math.radians(_ANGLE_TOL)):
                results.append(
                    Relationship(
                        source_id=planes[i].persistent_id,
                        target_id=planes[j].persistent_id,
                        relationship_type=RelationshipType.PERPENDICULAR,
                    )
                )
    return results


def detect_concentric_faces(faces: list[FaceData]) -> list[Relationship]:
    """Detect concentric cylindrical/conical faces (same axis, different radii)."""
    results: list[Relationship] = []
    revolute = [
        f for f in faces
        if f.surface_type in (SurfaceType.CYLINDER, SurfaceType.CONE)
        and f.axis is not None
        and f.radius is not None
    ]

    for i in range(len(revolute)):
        for j in range(i + 1, len(revolute)):
            f1, f2 = revolute[i], revolute[j]
            # Check axes are colinear: parallel + small distance between axes
            a1 = _normalize(f1.axis)
            a2 = _normalize(f2.axis)
            if abs(_dot(a1, a2)) < math.cos(math.radians(_ANGLE_TOL)):
                continue

            # Check centroids project close to same axis line
            diff = Vec3(
                x=f2.centroid.x - f1.centroid.x,
                y=f2.centroid.y - f1.centroid.y,
                z=f2.centroid.z - f1.centroid.z,
            )
            cross = _cross(a1, diff)
            lateral_dist = _magnitude(cross)

            if lateral_dist < 0.1:  # tolerance in mm
                results.append(
                    Relationship(
                        source_id=f1.persistent_id,
                        target_id=f2.persistent_id,
                        relationship_type=RelationshipType.CONCENTRIC,
                        parameters={
                            "radius_1": f1.radius,
                            "radius_2": f2.radius,
                        },
                    )
                )
    return results


def detect_coaxial_faces(faces: list[FaceData]) -> list[Relationship]:
    """Detect coaxial faces (same axis and same radius)."""
    results: list[Relationship] = []
    revolute = [
        f for f in faces
        if f.surface_type in (SurfaceType.CYLINDER, SurfaceType.CONE)
        and f.axis is not None
        and f.radius is not None
    ]

    for i in range(len(revolute)):
        for j in range(i + 1, len(revolute)):
            f1, f2 = revolute[i], revolute[j]
            a1 = _normalize(f1.axis)
            a2 = _normalize(f2.axis)
            if abs(_dot(a1, a2)) < math.cos(math.radians(_ANGLE_TOL)):
                continue

            diff = Vec3(
                x=f2.centroid.x - f1.centroid.x,
                y=f2.centroid.y - f1.centroid.y,
                z=f2.centroid.z - f1.centroid.z,
            )
            cross = _cross(a1, diff)
            lateral_dist = _magnitude(cross)

            if lateral_dist < 0.1 and abs(f1.radius - f2.radius) < 0.01:
                results.append(
                    Relationship(
                        source_id=f1.persistent_id,
                        target_id=f2.persistent_id,
                        relationship_type=RelationshipType.COAXIAL,
                        parameters={"radius": f1.radius},
                    )
                )
    return results


def detect_coplanar_faces(faces: list[FaceData]) -> list[Relationship]:
    """Detect coplanar planar faces (parallel + zero offset)."""
    results: list[Relationship] = []
    planes = [f for f in faces if f.surface_type == SurfaceType.PLANE and f.normal is not None]

    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            n1 = _normalize(planes[i].normal)
            n2 = _normalize(planes[j].normal)
            if abs(_dot(n1, n2)) < math.cos(math.radians(_ANGLE_TOL)):
                continue

            diff = Vec3(
                x=planes[j].centroid.x - planes[i].centroid.x,
                y=planes[j].centroid.y - planes[i].centroid.y,
                z=planes[j].centroid.z - planes[i].centroid.z,
            )
            offset = abs(_dot(diff, n1))
            if offset < 0.01:  # within tolerance
                results.append(
                    Relationship(
                        source_id=planes[i].persistent_id,
                        target_id=planes[j].persistent_id,
                        relationship_type=RelationshipType.COPLANAR,
                    )
                )
    return results


def detect_tangent_faces(
    faces: list[FaceData], adjacency: list[AdjacencyEdge]
) -> list[Relationship]:
    """Detect tangent adjacent faces (smooth G1 continuity at shared edge).

    Heuristic: a cylindrical/toroidal face adjacent to a planar face where
    the cylinder axis is perpendicular to the plane normal → tangent transition.
    """
    results: list[Relationship] = []
    face_lookup = {f.persistent_id: f for f in faces}

    for adj in adjacency:
        if adj.edge_type != EdgeRelationType.ADJACENT:
            continue
        f1 = face_lookup.get(adj.source_id)
        f2 = face_lookup.get(adj.target_id)
        if f1 is None or f2 is None:
            continue

        # Check plane-cylinder tangency
        plane, cyl = None, None
        if f1.surface_type == SurfaceType.PLANE and f2.surface_type in (
            SurfaceType.CYLINDER, SurfaceType.TORUS
        ):
            plane, cyl = f1, f2
        elif f2.surface_type == SurfaceType.PLANE and f1.surface_type in (
            SurfaceType.CYLINDER, SurfaceType.TORUS
        ):
            plane, cyl = f2, f1
        else:
            continue

        if plane.normal is None or cyl.axis is None:
            continue

        # Tangent if cylinder axis is perpendicular to plane normal
        n = _normalize(plane.normal)
        a = _normalize(cyl.axis)
        if abs(_dot(n, a)) < math.sin(math.radians(10)):
            results.append(
                Relationship(
                    source_id=plane.persistent_id,
                    target_id=cyl.persistent_id,
                    relationship_type=RelationshipType.TANGENT,
                )
            )

    return results


def detect_all_relationships(
    faces: list[FaceData], adjacency: list[AdjacencyEdge]
) -> list[Relationship]:
    """Run all relationship detectors and return combined results."""
    results: list[Relationship] = []
    results.extend(detect_parallel_faces(faces))
    results.extend(detect_perpendicular_faces(faces))
    results.extend(detect_concentric_faces(faces))
    results.extend(detect_coaxial_faces(faces))
    results.extend(detect_coplanar_faces(faces))
    results.extend(detect_tangent_faces(faces, adjacency))
    return results
