"""Chamfer feature recognition.

Detection strategy:
- Planar face at an angle between two other faces
- Narrow strip geometry (small area relative to neighbors)
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


def _angle_between_normals(n1: Vec3, n2: Vec3) -> float:
    """Compute angle in degrees between two normal vectors."""
    dot = n1.x * n2.x + n1.y * n2.y + n1.z * n2.z
    dot = max(-1.0, min(1.0, dot))  # clamp for numerical safety
    return math.degrees(math.acos(abs(dot)))


class ChamferRecognizer:
    """Recognizes chamfer features (angled planar cuts between faces)."""

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

        for face in faces:
            if face.surface_type != SurfaceType.PLANE:
                continue
            if face.normal is None:
                continue

            adjacent_ids = adj_lookup.get(face.persistent_id, [])
            adjacent_faces = [face_lookup[fid] for fid in adjacent_ids if fid in face_lookup]

            # Chamfer: small planar face at ~45° between two larger faces
            larger_neighbors_with_normals = [
                af
                for af in adjacent_faces
                if af.area > face.area and af.normal is not None
            ]

            if len(larger_neighbors_with_normals) >= 2:
                # Check that this face is angled (not parallel/perpendicular) to neighbors
                angles = [
                    _angle_between_normals(face.normal, n.normal)
                    for n in larger_neighbors_with_normals
                    if n.normal is not None
                ]
                # Chamfers are typically at 15-75 degrees to their neighbors
                angled = [a for a in angles if 15.0 < a < 75.0]

                if len(angled) >= 2:
                    fid = feature_id(FeatureType.CHAMFER.value, [face.persistent_id])
                    features.append(
                        FeatureData(
                            persistent_id=fid,
                            feature_type=FeatureType.CHAMFER,
                            face_ids=[face.persistent_id],
                            edge_ids=face.edge_ids,
                            parameters={"angle": angled[0]},
                            normal=face.normal,
                            description=f"Chamfer, ~{angled[0]:.0f}° cut",
                        )
                    )

        return features
