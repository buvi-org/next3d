"""Shared test fixtures for next3d tests."""

from __future__ import annotations

import pytest

from next3d.core.schema import (
    AdjacencyEdge,
    CurveType,
    EdgeData,
    EdgeRelationType,
    FaceData,
    FeatureData,
    FeatureType,
    SemanticGraph,
    SolidData,
    SurfaceType,
    Vec3,
    VertexData,
)


@pytest.fixture
def sample_faces() -> list[FaceData]:
    """A minimal set of faces: one cylinder + two planes (hole-like)."""
    return [
        FaceData(
            persistent_id="face_cyl_001",
            surface_type=SurfaceType.CYLINDER,
            area=314.159,
            centroid=Vec3(x=0, y=0, z=5),
            radius=5.0,
            axis=Vec3(x=0, y=0, z=1),
            edge_ids=["edge_circ_top", "edge_circ_bot"],
        ),
        FaceData(
            persistent_id="face_plane_top",
            surface_type=SurfaceType.PLANE,
            area=1000.0,
            centroid=Vec3(x=0, y=0, z=10),
            normal=Vec3(x=0, y=0, z=1),
            edge_ids=["edge_circ_top", "edge_line_1"],
        ),
        FaceData(
            persistent_id="face_plane_bot",
            surface_type=SurfaceType.PLANE,
            area=1000.0,
            centroid=Vec3(x=0, y=0, z=0),
            normal=Vec3(x=0, y=0, z=-1),
            edge_ids=["edge_circ_bot", "edge_line_2"],
        ),
    ]


@pytest.fixture
def sample_edges() -> list[EdgeData]:
    """Edges for the sample faces."""
    return [
        EdgeData(
            persistent_id="edge_circ_top",
            curve_type=CurveType.CIRCLE,
            length=31.416,
            start_vertex="vtx_1",
            end_vertex="vtx_1",
            radius=5.0,
            center=Vec3(x=0, y=0, z=10),
        ),
        EdgeData(
            persistent_id="edge_circ_bot",
            curve_type=CurveType.CIRCLE,
            length=31.416,
            start_vertex="vtx_2",
            end_vertex="vtx_2",
            radius=5.0,
            center=Vec3(x=0, y=0, z=0),
        ),
        EdgeData(
            persistent_id="edge_line_1",
            curve_type=CurveType.LINE,
            length=50.0,
            start_vertex="vtx_3",
            end_vertex="vtx_4",
        ),
        EdgeData(
            persistent_id="edge_line_2",
            curve_type=CurveType.LINE,
            length=50.0,
            start_vertex="vtx_5",
            end_vertex="vtx_6",
        ),
    ]


@pytest.fixture
def sample_adjacency() -> list[AdjacencyEdge]:
    """Adjacency edges: cylinder is adjacent to both planes."""
    return [
        AdjacencyEdge(
            source_id="face_cyl_001",
            target_id="face_plane_top",
            edge_type=EdgeRelationType.ADJACENT,
            shared_edge_id="edge_circ_top",
        ),
        AdjacencyEdge(
            source_id="face_cyl_001",
            target_id="face_plane_bot",
            edge_type=EdgeRelationType.ADJACENT,
            shared_edge_id="edge_circ_bot",
        ),
    ]


@pytest.fixture
def sample_graph(sample_faces, sample_edges, sample_adjacency) -> SemanticGraph:
    """A minimal semantic graph with a through-hole-like geometry."""
    return SemanticGraph(
        solids=[
            SolidData(
                persistent_id="solid_001",
                face_ids=[f.persistent_id for f in sample_faces],
                volume=5000.0,
                centroid=Vec3(x=0, y=0, z=5),
            )
        ],
        faces=sample_faces,
        edges=sample_edges,
        vertices=[
            VertexData(persistent_id="vtx_1", position=Vec3(x=5, y=0, z=10)),
            VertexData(persistent_id="vtx_2", position=Vec3(x=5, y=0, z=0)),
        ],
        features=[],
        adjacency=sample_adjacency,
        relationships=[],
    )
