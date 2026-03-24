"""Slot feature recognition.

Detection strategy:
- U-shaped profile: planar bottom + two planar side walls + cylindrical ends
- The bottom face is adjacent to two parallel planar walls
- Walls are perpendicular (or near-perpendicular) to the bottom
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


def _are_parallel(n1: Vec3, n2: Vec3, tol_deg: float = 5.0) -> bool:
    """Check if two normals are parallel (same or opposite direction)."""
    d = abs(_dot(n1, n2))
    return d > math.cos(math.radians(tol_deg))


def _are_perpendicular(n1: Vec3, n2: Vec3, tol_deg: float = 10.0) -> bool:
    """Check if two normals are perpendicular."""
    d = abs(_dot(n1, n2))
    return d < math.sin(math.radians(tol_deg))


class SlotRecognizer:
    """Recognizes slot features (U-shaped profiles).

    Pattern: a planar bottom face adjacent to two parallel planar side walls
    that are perpendicular to the bottom. Optionally bounded by cylindrical ends.
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

        # Track faces already assigned to a slot to avoid duplicates
        used_faces: set[str] = set()

        for face in faces:
            if face.persistent_id in used_faces:
                continue
            if face.surface_type != SurfaceType.PLANE:
                continue
            if face.normal is None:
                continue

            adjacent_ids = adj_lookup.get(face.persistent_id, [])
            adjacent_faces = [face_lookup[fid] for fid in adjacent_ids if fid in face_lookup]

            # Find planar neighbors perpendicular to this face (candidate walls)
            perp_walls = [
                af
                for af in adjacent_faces
                if af.surface_type == SurfaceType.PLANE
                and af.normal is not None
                and _are_perpendicular(face.normal, af.normal)
            ]

            # Need at least two walls that are parallel to each other
            for i in range(len(perp_walls)):
                for j in range(i + 1, len(perp_walls)):
                    w1, w2 = perp_walls[i], perp_walls[j]
                    if w1.normal is None or w2.normal is None:
                        continue
                    if not _are_parallel(w1.normal, w2.normal):
                        continue

                    # Found a slot: bottom + two parallel walls
                    slot_faces = [face.persistent_id, w1.persistent_id, w2.persistent_id]

                    # Check for cylindrical end faces (optional)
                    end_faces = []
                    for af in adjacent_faces:
                        if af.surface_type == SurfaceType.CYLINDER and af.persistent_id not in slot_faces:
                            end_faces.append(af)

                    all_faces = slot_faces + [ef.persistent_id for ef in end_faces]

                    if any(fid in used_faces for fid in all_faces):
                        continue

                    # Estimate slot width from wall distance (using centroids as approximation)
                    dx = w1.centroid.x - w2.centroid.x
                    dy = w1.centroid.y - w2.centroid.y
                    dz = w1.centroid.z - w2.centroid.z
                    width = math.sqrt(dx * dx + dy * dy + dz * dz)

                    all_edge_ids = face.edge_ids + w1.edge_ids + w2.edge_ids
                    fid = feature_id(FeatureType.SLOT.value, all_faces)
                    features.append(
                        FeatureData(
                            persistent_id=fid,
                            feature_type=FeatureType.SLOT,
                            face_ids=all_faces,
                            edge_ids=list(set(all_edge_ids)),
                            parameters={"width": round(width, 3)},
                            axis=face.normal,
                            description=f"Slot, width ~{width:.1f}mm",
                        )
                    )
                    used_faces.update(all_faces)

        return features
