"""Graph embedding export for GNN / ML model consumption.

Exports the semantic graph as numeric tensors suitable for graph neural networks:
- Node feature matrix  (N × D)
- Edge index           (2 × E) — COO format
- Edge attribute matrix (E × K)
- Node type vector     (N,)

Output is plain Python lists (JSON-serializable). Can be loaded directly
into PyTorch Geometric, DGL, or any GNN framework.
"""

from __future__ import annotations

import math
from typing import Any

from next3d.core.schema import (
    CurveType,
    EdgeRelationType,
    FaceData,
    SemanticGraph,
    SurfaceType,
)


# ---------------------------------------------------------------------------
# Surface type encoding (one-hot)
# ---------------------------------------------------------------------------
_SURFACE_TYPES = [s.value for s in SurfaceType]
_CURVE_TYPES = [c.value for c in CurveType]


def _one_hot(value: str, vocab: list[str]) -> list[float]:
    return [1.0 if v == value else 0.0 for v in vocab]


def _safe_vec(vec, default=(0.0, 0.0, 0.0)):
    if vec is None:
        return list(default)
    return [vec.x, vec.y, vec.z]


# ---------------------------------------------------------------------------
# Node features
# ---------------------------------------------------------------------------


def _face_features(face: FaceData) -> list[float]:
    """Encode a face as a fixed-length feature vector.

    Features (17-dim):
        [0:7]  surface type one-hot (7)
        [7:10] normal xyz (3)
        [10]   area (1)
        [11]   radius or 0 (1)
        [12:15] centroid xyz (3)
        [15:17] log_area, log_radius (2)  — log-scaled for better ML training
    """
    features: list[float] = []
    features.extend(_one_hot(face.surface_type.value, _SURFACE_TYPES))
    features.extend(_safe_vec(face.normal))
    features.append(face.area)
    features.append(face.radius or 0.0)
    features.extend([face.centroid.x, face.centroid.y, face.centroid.z])
    features.append(math.log1p(face.area))
    features.append(math.log1p(face.radius or 0.0))
    return features


FACE_FEATURE_DIM = 17  # keep in sync with _face_features


# ---------------------------------------------------------------------------
# Graph export
# ---------------------------------------------------------------------------


def to_graph_tensors(graph: SemanticGraph) -> dict[str, Any]:
    """Export semantic graph as numeric tensors for GNN consumption.

    Returns:
        Dictionary with:
        - node_features: list[list[float]] shape (N, D)
        - edge_index: list[list[int]] shape (2, E) — COO format
        - edge_attr: list[list[float]] shape (E, K)
        - node_ids: list[str] — persistent IDs for each node index
        - node_type: list[str] — 'face' for now (extensible)
        - num_nodes: int
        - num_edges: int
        - feature_dim: int
        - metadata: dict with feature/relationship counts
    """
    # Build node index from faces
    face_to_idx: dict[str, int] = {}
    node_features: list[list[float]] = []
    node_ids: list[str] = []

    for i, face in enumerate(graph.faces):
        face_to_idx[face.persistent_id] = i
        node_features.append(_face_features(face))
        node_ids.append(face.persistent_id)

    # Build edge index from adjacency (undirected → both directions)
    src_list: list[int] = []
    tgt_list: list[int] = []
    edge_attr: list[list[float]] = []

    _EDGE_TYPE_MAP = {
        EdgeRelationType.ADJACENT: [1.0, 0.0, 0.0],
        EdgeRelationType.CONTAINS: [0.0, 1.0, 0.0],
        EdgeRelationType.SHARES_VERTEX: [0.0, 0.0, 1.0],
    }

    for adj in graph.adjacency:
        si = face_to_idx.get(adj.source_id)
        ti = face_to_idx.get(adj.target_id)
        if si is None or ti is None:
            continue
        attr = _EDGE_TYPE_MAP.get(adj.edge_type, [0.0, 0.0, 0.0])
        # Forward
        src_list.append(si)
        tgt_list.append(ti)
        edge_attr.append(attr)
        # Backward (undirected)
        src_list.append(ti)
        tgt_list.append(si)
        edge_attr.append(attr)

    return {
        "node_features": node_features,
        "edge_index": [src_list, tgt_list],
        "edge_attr": edge_attr,
        "node_ids": node_ids,
        "node_type": ["face"] * len(node_ids),
        "num_nodes": len(node_ids),
        "num_edges": len(src_list),
        "feature_dim": FACE_FEATURE_DIM,
        "metadata": {
            "features_count": len(graph.features),
            "relationships_count": len(graph.relationships),
            "feature_types": list({f.feature_type.value for f in graph.features}),
        },
    }
