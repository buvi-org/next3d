"""Streaming export for large models.

Yields semantic graph data in chunks for incremental consumption,
avoiding the need to serialize the entire graph at once.

Chunks are newline-delimited JSON (NDJSON) — each line is a valid
JSON object that can be parsed independently.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from next3d.core.schema import SemanticGraph


def stream_semantic_graph(graph: SemanticGraph) -> Iterator[str]:
    """Stream the semantic graph as newline-delimited JSON chunks.

    Yields chunks in order:
    1. Header (type, version, statistics)
    2. Solids (one per line)
    3. Features (one per line)
    4. Faces (one per line)
    5. Edges (one per line)
    6. Vertices (one per line)
    7. Adjacency edges (one per line)
    8. Relationships (one per line)
    9. Footer (end marker)

    Each line is a valid JSON object with a "chunk_type" field.
    """
    # Header
    yield _line({
        "chunk_type": "header",
        "type": "semantic_3d_graph_stream",
        "version": "0.1.0",
        "statistics": {
            "solids": len(graph.solids),
            "faces": len(graph.faces),
            "edges": len(graph.edges),
            "vertices": len(graph.vertices),
            "features": len(graph.features),
            "adjacency": len(graph.adjacency),
            "relationships": len(graph.relationships),
        },
    })

    # Solids
    for solid in graph.solids:
        yield _line({
            "chunk_type": "solid",
            **json.loads(solid.model_dump_json()),
        })

    # Features
    for feat in graph.features:
        yield _line({
            "chunk_type": "feature",
            **json.loads(feat.model_dump_json()),
        })

    # Faces
    for face in graph.faces:
        yield _line({
            "chunk_type": "face",
            **json.loads(face.model_dump_json()),
        })

    # Edges
    for edge in graph.edges:
        yield _line({
            "chunk_type": "edge",
            **json.loads(edge.model_dump_json()),
        })

    # Vertices
    for vertex in graph.vertices:
        yield _line({
            "chunk_type": "vertex",
            **json.loads(vertex.model_dump_json()),
        })

    # Adjacency
    for adj in graph.adjacency:
        yield _line({
            "chunk_type": "adjacency",
            **json.loads(adj.model_dump_json()),
        })

    # Relationships
    for rel in graph.relationships:
        yield _line({
            "chunk_type": "relationship",
            **json.loads(rel.model_dump_json()),
        })

    # Footer
    yield _line({"chunk_type": "footer", "status": "complete"})


def _line(obj: dict[str, Any]) -> str:
    return json.dumps(obj, separators=(",", ":"))
