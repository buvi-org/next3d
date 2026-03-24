"""Boss/protrusion feature recognition.

Detection strategy:
- Cylindrical or prismatic face protruding from a planar base face
"""

from __future__ import annotations

from next3d.core.schema import AdjacencyEdge, EdgeData, FaceData, FeatureData


class BossRecognizer:
    """Recognizes boss features (protrusions from a base face)."""

    def recognize(
        self,
        faces: list[FaceData],
        edges: list[EdgeData],
        adjacency: list[AdjacencyEdge],
    ) -> list[FeatureData]:
        # TODO: Implement boss recognition
        # Strategy: cylindrical face with top planar cap, adjacent to a larger base plane
        return []
