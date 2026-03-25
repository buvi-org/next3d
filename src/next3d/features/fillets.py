"""Fillet feature recognition.

Detection strategy:
- Toroidal or cylindrical face that is tangent to two adjacent faces.
- The face is a narrow blend surface connecting two larger faces.
- Cylindrical fillets have angular extent < 180 degrees (partial arcs).
"""

from __future__ import annotations

import math

from next3d.core.identity import feature_id
from next3d.core.schema import (
    AdjacencyEdge,
    CurveType,
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

            if len(larger_neighbors) >= 2:
                # Both toroidal faces (curved-edge blends) and cylindrical
                # faces (straight-edge blends) can be fillets when they are
                # sandwiched between two larger neighbors.
                # For cylindrical faces, check angular extent: fillets are
                # partial arcs (< 180 deg), while holes span >= 180 deg.
                if face.surface_type == SurfaceType.CYLINDER:
                    edge_lookup = {e.persistent_id: e for e in edges}
                    line_edges = [
                        edge_lookup[eid]
                        for eid in face.edge_ids
                        if eid in edge_lookup
                        and edge_lookup[eid].curve_type == CurveType.LINE
                    ]
                    if line_edges:
                        height = max(e.length for e in line_edges)
                        if height > 0 and face.radius and face.radius > 0:
                            theta = face.area / (face.radius * height)
                            if theta >= math.pi * 0.9:
                                continue  # >= ~180 deg → hole, not fillet
                    else:
                        # No line edges and it's a cylinder — likely a hole
                        continue

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
