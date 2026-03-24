"""Hole feature recognition: through holes, blind holes, counterbores.

Detection strategy (rule-based):
- A cylindrical face bounded by circular edges
- Through hole: both bounding edges are shared with other faces
- Blind hole: one bounding edge is shared, one connects to a planar bottom
"""

from __future__ import annotations

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


def _build_adjacency_lookup(adjacency: list[AdjacencyEdge]) -> dict[str, list[str]]:
    """Build face_id → [adjacent_face_ids] mapping."""
    lookup: dict[str, list[str]] = {}
    for adj in adjacency:
        if adj.edge_type != EdgeRelationType.ADJACENT:
            continue
        lookup.setdefault(adj.source_id, []).append(adj.target_id)
        lookup.setdefault(adj.target_id, []).append(adj.source_id)
    return lookup


def _build_face_lookup(faces: list[FaceData]) -> dict[str, FaceData]:
    return {f.persistent_id: f for f in faces}


def _build_edge_lookup(edges: list[EdgeData]) -> dict[str, EdgeData]:
    return {e.persistent_id: e for e in edges}


class HoleRecognizer:
    """Recognizes through holes and blind holes."""

    def recognize(
        self,
        faces: list[FaceData],
        edges: list[EdgeData],
        adjacency: list[AdjacencyEdge],
    ) -> list[FeatureData]:
        features: list[FeatureData] = []
        adj_lookup = _build_adjacency_lookup(adjacency)
        face_lookup = _build_face_lookup(faces)
        edge_lookup = _build_edge_lookup(edges)

        for face in faces:
            if face.surface_type != SurfaceType.CYLINDER:
                continue
            if face.radius is None:
                continue

            # Get circular bounding edges
            circular_edges = [
                edge_lookup[eid]
                for eid in face.edge_ids
                if eid in edge_lookup and edge_lookup[eid].curve_type == CurveType.CIRCLE
            ]

            if len(circular_edges) < 1:
                continue

            # Check adjacent faces
            adjacent_ids = adj_lookup.get(face.persistent_id, [])
            adjacent_faces = [face_lookup[fid] for fid in adjacent_ids if fid in face_lookup]

            # Skip fillet-like cylinders: small cylinder sandwiched between
            # two larger faces (characteristic of edge blends, not holes)
            larger_neighbors = [af for af in adjacent_faces if af.area > face.area * 2]
            if len(larger_neighbors) >= 2 and len(circular_edges) == 0:
                continue  # likely a fillet, not a hole

            # Through hole: cylindrical face with 2 circular edges, all neighbors are non-cylindrical
            if len(circular_edges) >= 2:
                fid = feature_id(FeatureType.THROUGH_HOLE.value, [face.persistent_id])
                features.append(
                    FeatureData(
                        persistent_id=fid,
                        feature_type=FeatureType.THROUGH_HOLE,
                        face_ids=[face.persistent_id],
                        edge_ids=[e.persistent_id for e in circular_edges],
                        parameters={
                            "diameter": face.radius * 2,
                            "radius": face.radius,
                        },
                        axis=face.axis,
                        description=f"Through hole, diameter {face.radius * 2:.2f}mm",
                    )
                )
            # Blind hole: cylindrical face + adjacent planar bottom
            elif len(circular_edges) == 1:
                planar_bottoms = [
                    af for af in adjacent_faces if af.surface_type == SurfaceType.PLANE
                ]
                if planar_bottoms:
                    bottom = planar_bottoms[0]
                    fid = feature_id(
                        FeatureType.BLIND_HOLE.value,
                        [face.persistent_id, bottom.persistent_id],
                    )
                    features.append(
                        FeatureData(
                            persistent_id=fid,
                            feature_type=FeatureType.BLIND_HOLE,
                            face_ids=[face.persistent_id, bottom.persistent_id],
                            edge_ids=[e.persistent_id for e in circular_edges],
                            parameters={
                                "diameter": face.radius * 2,
                                "radius": face.radius,
                            },
                            axis=face.axis,
                            description=f"Blind hole, diameter {face.radius * 2:.2f}mm",
                        )
                    )

        return features
