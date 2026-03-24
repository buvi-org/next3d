"""Fillet feature recognition.

Detection strategy:
- Toroidal or cylindrical face that is tangent to two adjacent faces.
- The face is a narrow blend surface connecting two larger faces.
"""

from __future__ import annotations

from next3d.core.identity import feature_id
from next3d.core.schema import (
    AdjacencyEdge,
    EdgeData,
    EdgeRelationType,
    FaceData,
    FeatureData,
    FeatureType,
    SurfaceType,
)


class FilletRecognizer:
    """Recognizes fillet features (constant-radius blends)."""

    def recognize(
        self,
        faces: list[FaceData],
        edges: list[EdgeData],
        adjacency: list[AdjacencyEdge],
    ) -> list[FeatureData]:
        features: list[FeatureData] = []
        face_lookup = {f.persistent_id: f for f in faces}

        # Build adjacency lookup
        adj_lookup: dict[str, list[str]] = {}
        for adj in adjacency:
            if adj.edge_type != EdgeRelationType.ADJACENT:
                continue
            adj_lookup.setdefault(adj.source_id, []).append(adj.target_id)
            adj_lookup.setdefault(adj.target_id, []).append(adj.source_id)

        for face in faces:
            # Fillets are typically toroidal or cylindrical blends
            if face.surface_type not in (SurfaceType.TORUS, SurfaceType.CYLINDER):
                continue
            if face.radius is None:
                continue

            adjacent_ids = adj_lookup.get(face.persistent_id, [])
            adjacent_faces = [face_lookup[fid] for fid in adjacent_ids if fid in face_lookup]

            # A fillet connects exactly two larger faces
            # and its area is typically smaller than both neighbors
            larger_neighbors = [af for af in adjacent_faces if af.area > face.area]

            if len(larger_neighbors) >= 2 and face.surface_type == SurfaceType.TORUS:
                fid = feature_id(FeatureType.FILLET.value, [face.persistent_id])
                features.append(
                    FeatureData(
                        persistent_id=fid,
                        feature_type=FeatureType.FILLET,
                        face_ids=[face.persistent_id],
                        edge_ids=face.edge_ids,
                        parameters={"radius": face.radius},
                        description=f"Fillet, radius {face.radius:.2f}mm",
                    )
                )

        return features
