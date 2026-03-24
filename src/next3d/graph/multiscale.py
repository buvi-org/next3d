"""Multi-scale representation: navigate between abstraction levels.

Levels:
    L0 — Solid (top-level body)
    L1 — Features (holes, fillets, chamfers, etc.)
    L2 — Faces (topological faces with surface geometry)
    L3 — Edges (topological edges with curve geometry)
    L4 — Vertices (3D points)

Allows AI to zoom in/out across levels, e.g.:
    "Show me the features of solid_001"
    "What edges bound face_abc?"
    "Which feature contains edge_xyz?"
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

from next3d.core.schema import (
    EdgeData,
    FaceData,
    FeatureData,
    SemanticGraph,
    SolidData,
    VertexData,
)


class Level(IntEnum):
    SOLID = 0
    FEATURE = 1
    FACE = 2
    EDGE = 3
    VERTEX = 4


class MultiScaleView:
    """Navigate a SemanticGraph across abstraction levels."""

    def __init__(self, graph: SemanticGraph):
        self._graph = graph
        self._face_to_feature: dict[str, list[str]] = {}
        self._edge_to_face: dict[str, list[str]] = {}

        # Build reverse indexes
        for feat in graph.features:
            for fid in feat.face_ids:
                self._face_to_feature.setdefault(fid, []).append(feat.persistent_id)
        for face in graph.faces:
            for eid in face.edge_ids:
                self._edge_to_face.setdefault(eid, []).append(face.persistent_id)

    def at_level(self, level: Level) -> list[Any]:
        """Get all entities at a given level."""
        if level == Level.SOLID:
            return list(self._graph.solids)
        elif level == Level.FEATURE:
            return list(self._graph.features)
        elif level == Level.FACE:
            return list(self._graph.faces)
        elif level == Level.EDGE:
            return list(self._graph.edges)
        elif level == Level.VERTEX:
            return list(self._graph.vertices)
        return []

    def children(self, entity_id: str) -> list[Any]:
        """Get children one level down.

        Solid → Features (or Faces if no features)
        Feature → Faces
        Face → Edges
        Edge → Vertices
        """
        # Check if it's a solid
        for solid in self._graph.solids:
            if solid.persistent_id == entity_id:
                # Return features that belong to this solid's faces
                solid_face_ids = set(solid.face_ids)
                features = [
                    f for f in self._graph.features
                    if any(fid in solid_face_ids for fid in f.face_ids)
                ]
                if features:
                    return features
                # Fallback: return faces directly
                return [f for f in self._graph.faces if f.persistent_id in solid_face_ids]

        # Check if it's a feature
        for feat in self._graph.features:
            if feat.persistent_id == entity_id:
                return [
                    f for f in self._graph.faces
                    if f.persistent_id in feat.face_ids
                ]

        # Check if it's a face
        for face in self._graph.faces:
            if face.persistent_id == entity_id:
                return [
                    e for e in self._graph.edges
                    if e.persistent_id in face.edge_ids
                ]

        # Check if it's an edge
        for edge in self._graph.edges:
            if edge.persistent_id == entity_id:
                vertex_ids = {edge.start_vertex, edge.end_vertex}
                return [
                    v for v in self._graph.vertices
                    if v.persistent_id in vertex_ids
                ]

        return []

    def parent(self, entity_id: str) -> list[Any]:
        """Get parents one level up.

        Vertex → Edges
        Edge → Faces
        Face → Features (or Solid)
        Feature → Solid
        """
        # Vertex → edges containing it
        for v in self._graph.vertices:
            if v.persistent_id == entity_id:
                return [
                    e for e in self._graph.edges
                    if entity_id in (e.start_vertex, e.end_vertex)
                ]

        # Edge → faces containing it
        for e in self._graph.edges:
            if e.persistent_id == entity_id:
                face_ids = self._edge_to_face.get(entity_id, [])
                return [f for f in self._graph.faces if f.persistent_id in face_ids]

        # Face → features containing it
        for f in self._graph.faces:
            if f.persistent_id == entity_id:
                feat_ids = self._face_to_feature.get(entity_id, [])
                if feat_ids:
                    return [
                        ft for ft in self._graph.features
                        if ft.persistent_id in feat_ids
                    ]
                # Fallback: return containing solid
                return [
                    s for s in self._graph.solids
                    if entity_id in s.face_ids
                ]

        # Feature → solid
        for feat in self._graph.features:
            if feat.persistent_id == entity_id:
                feat_face_ids = set(feat.face_ids)
                return [
                    s for s in self._graph.solids
                    if feat_face_ids & set(s.face_ids)
                ]

        return []

    def level_of(self, entity_id: str) -> Level | None:
        """Determine which level an entity belongs to."""
        for s in self._graph.solids:
            if s.persistent_id == entity_id:
                return Level.SOLID
        for f in self._graph.features:
            if f.persistent_id == entity_id:
                return Level.FEATURE
        for f in self._graph.faces:
            if f.persistent_id == entity_id:
                return Level.FACE
        for e in self._graph.edges:
            if e.persistent_id == entity_id:
                return Level.EDGE
        for v in self._graph.vertices:
            if v.persistent_id == entity_id:
                return Level.VERTEX
        return None

    def summary_at_level(self, level: Level) -> dict:
        """Get a count summary at a given level."""
        entities = self.at_level(level)
        return {
            "level": level.name,
            "count": len(entities),
            "ids": [getattr(e, "persistent_id", None) for e in entities],
        }
