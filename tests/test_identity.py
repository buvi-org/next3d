"""Tests for the persistent identity system."""

from next3d.core.identity import vertex_id, edge_id, face_id, solid_id, feature_id


class TestVertexId:
    def test_deterministic(self):
        assert vertex_id(1.0, 2.0, 3.0) == vertex_id(1.0, 2.0, 3.0)

    def test_different_coords_differ(self):
        assert vertex_id(1.0, 2.0, 3.0) != vertex_id(1.0, 2.0, 3.1)

    def test_prefix(self):
        assert vertex_id(0, 0, 0).startswith("vertex_")


class TestEdgeId:
    def test_direction_independent(self):
        """Same edge traversed in opposite directions → same ID."""
        id1 = edge_id(0, 0, 0, 1, 1, 1, "line")
        id2 = edge_id(1, 1, 1, 0, 0, 0, "line")
        assert id1 == id2

    def test_different_curve_type_differs(self):
        id1 = edge_id(0, 0, 0, 1, 1, 1, "line")
        id2 = edge_id(0, 0, 0, 1, 1, 1, "circle")
        assert id1 != id2

    def test_prefix(self):
        assert edge_id(0, 0, 0, 1, 0, 0, "line").startswith("edge_")


class TestFaceId:
    def test_deterministic(self):
        id1 = face_id("plane", 0, 0, 0, 100.0)
        id2 = face_id("plane", 0, 0, 0, 100.0)
        assert id1 == id2

    def test_different_type_differs(self):
        id1 = face_id("plane", 0, 0, 0, 100.0)
        id2 = face_id("cylinder", 0, 0, 0, 100.0)
        assert id1 != id2

    def test_prefix(self):
        assert face_id("plane", 0, 0, 0, 100).startswith("face_")


class TestSolidId:
    def test_deterministic(self):
        assert solid_id(0, 0, 0, 1000) == solid_id(0, 0, 0, 1000)

    def test_prefix(self):
        assert solid_id(0, 0, 0, 1000).startswith("solid_")


class TestFeatureId:
    def test_deterministic(self):
        id1 = feature_id("hole", ["face_a", "face_b"])
        id2 = feature_id("hole", ["face_a", "face_b"])
        assert id1 == id2

    def test_order_independent(self):
        """Face order shouldn't matter — IDs are sorted internally."""
        id1 = feature_id("hole", ["face_b", "face_a"])
        id2 = feature_id("hole", ["face_a", "face_b"])
        assert id1 == id2

    def test_prefix(self):
        assert feature_id("hole", ["face_a"]).startswith("feature_")
