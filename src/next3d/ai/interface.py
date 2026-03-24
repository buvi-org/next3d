"""AI interface: structured export of semantic graphs for LLM and ML consumption.

Provides summary and detail modes for JSON export, optimized for
injection into LLM context windows.
"""

from __future__ import annotations

import json
from typing import Any

from next3d.core.schema import SemanticGraph


def to_summary(graph: SemanticGraph) -> dict[str, Any]:
    """Export a compact summary for LLM context injection.

    Includes: solid overview, feature list with key dimensions,
    face/edge/vertex counts. Omits raw topology details.
    """
    return {
        "type": "semantic_3d_summary",
        "version": "0.1.0",
        "statistics": {
            "solids": len(graph.solids),
            "faces": len(graph.faces),
            "edges": len(graph.edges),
            "vertices": len(graph.vertices),
            "features": len(graph.features),
        },
        "solids": [
            {
                "id": s.persistent_id,
                "volume": s.volume,
                "centroid": {"x": s.centroid.x, "y": s.centroid.y, "z": s.centroid.z}
                if s.centroid
                else None,
                "face_count": len(s.face_ids),
            }
            for s in graph.solids
        ],
        "features": [
            {
                "id": f.persistent_id,
                "type": f.feature_type.value,
                "parameters": f.parameters,
                "description": f.description,
                "axis": {"x": f.axis.x, "y": f.axis.y, "z": f.axis.z} if f.axis else None,
            }
            for f in graph.features
        ],
        "face_type_distribution": _face_type_distribution(graph),
        "relationships": _relationship_summary(graph),
    }


def _relationship_summary(graph: SemanticGraph) -> dict[str, int]:
    """Count relationships by type."""
    dist: dict[str, int] = {}
    for rel in graph.relationships:
        key = rel.relationship_type.value
        dist[key] = dist.get(key, 0) + 1
    return dist


def to_detail(graph: SemanticGraph) -> dict[str, Any]:
    """Export the full semantic graph as JSON.

    Includes complete topology, all entity metadata, adjacency,
    features, and relationships. Self-contained — an LLM reading
    this should understand the part without seeing the geometry.
    """
    return {
        "type": "semantic_3d_graph",
        "version": "0.1.0",
        **json.loads(graph.model_dump_json()),
    }


def to_json(graph: SemanticGraph, mode: str = "summary") -> str:
    """Export semantic graph as a JSON string.

    Args:
        graph: The semantic graph to export.
        mode: 'summary' for compact output, 'detail' for full graph.

    Returns:
        JSON string.
    """
    data = to_summary(graph) if mode == "summary" else to_detail(graph)
    return json.dumps(data, indent=2)


def _face_type_distribution(graph: SemanticGraph) -> dict[str, int]:
    """Count faces by surface type."""
    dist: dict[str, int] = {}
    for face in graph.faces:
        key = face.surface_type.value
        dist[key] = dist.get(key, 0) + 1
    return dist
