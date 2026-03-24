"""Tests for core schema models."""

from next3d.core.schema import (
    FaceData,
    SemanticGraph,
    SurfaceType,
    Vec3,
)


class TestSemanticGraph:
    def test_face_count(self, sample_graph):
        assert sample_graph.face_count == 3

    def test_get_face(self, sample_graph):
        face = sample_graph.get_face("face_cyl_001")
        assert face is not None
        assert face.surface_type == SurfaceType.CYLINDER

    def test_get_face_missing(self, sample_graph):
        assert sample_graph.get_face("nonexistent") is None

    def test_faces_adjacent_to(self, sample_graph):
        adj = sample_graph.faces_adjacent_to("face_cyl_001")
        assert set(adj) == {"face_plane_top", "face_plane_bot"}

    def test_json_roundtrip(self, sample_graph):
        json_str = sample_graph.model_dump_json()
        restored = SemanticGraph.model_validate_json(json_str)
        assert restored.face_count == sample_graph.face_count
        assert len(restored.adjacency) == len(sample_graph.adjacency)
