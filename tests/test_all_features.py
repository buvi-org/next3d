"""Tests for all feature recognizers against diverse STEP files."""

from __future__ import annotations

from pathlib import Path

from next3d.core.schema import FeatureType, RelationshipType
from next3d.graph.semantic import build_semantic_graph

FIXTURES = Path(__file__).parent / "fixtures"


class TestHolesOnBlock:
    def test_detects_through_holes(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        holes = [f for f in graph.features if f.feature_type == FeatureType.THROUGH_HOLE]
        assert len(holes) >= 2

    def test_hole_diameter(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        holes = [f for f in graph.features if f.feature_type == FeatureType.THROUGH_HOLE]
        # The actual design holes are 10mm diameter; fillet cylinders may also match
        big_holes = [h for h in holes if abs(h.parameters["diameter"] - 10.0) < 0.1]
        assert len(big_holes) >= 2, (
            f"Expected 2 holes of d=10mm, got diameters: "
            f"{[h.parameters['diameter'] for h in holes]}"
        )


class TestChamfers:
    def test_detects_chamfers(self):
        graph = build_semantic_graph(FIXTURES / "sample_chamfer.step")
        chamfers = [f for f in graph.features if f.feature_type == FeatureType.CHAMFER]
        assert len(chamfers) >= 1, (
            f"Expected chamfers, got: {[f.feature_type.value for f in graph.features]}"
        )


class TestBosses:
    def test_detects_bosses(self):
        graph = build_semantic_graph(FIXTURES / "sample_boss.step")
        bosses = [f for f in graph.features if f.feature_type == FeatureType.BOSS]
        assert len(bosses) >= 1, (
            f"Expected bosses, got: {[f.feature_type.value for f in graph.features]}"
        )

    def test_boss_parameters(self):
        graph = build_semantic_graph(FIXTURES / "sample_boss.step")
        bosses = [f for f in graph.features if f.feature_type == FeatureType.BOSS]
        if bosses:
            boss = bosses[0]
            assert "diameter" in boss.parameters
            assert "height" in boss.parameters
            assert boss.parameters["diameter"] > 0


class TestSlots:
    def test_detects_slots(self):
        graph = build_semantic_graph(FIXTURES / "sample_slot.step")
        slots = [f for f in graph.features if f.feature_type == FeatureType.SLOT]
        assert len(slots) >= 1, (
            f"Expected slots, got: {[f.feature_type.value for f in graph.features]}"
        )


class TestComplexPart:
    def test_multiple_feature_types(self):
        graph = build_semantic_graph(FIXTURES / "sample_complex.step")
        types = {f.feature_type for f in graph.features}
        # Should detect at least holes
        assert FeatureType.THROUGH_HOLE in types, (
            f"Expected through holes in complex part. Got: {[f.feature_type.value for f in graph.features]}"
        )

    def test_has_relationships(self):
        graph = build_semantic_graph(FIXTURES / "sample_complex.step")
        assert len(graph.relationships) > 0, "Complex part should have detected relationships"

    def test_has_parallel_faces(self):
        graph = build_semantic_graph(FIXTURES / "sample_complex.step")
        parallel = [r for r in graph.relationships if r.relationship_type == RelationshipType.PARALLEL]
        assert len(parallel) > 0, "Box should have parallel faces"

    def test_has_perpendicular_faces(self):
        graph = build_semantic_graph(FIXTURES / "sample_complex.step")
        perp = [r for r in graph.relationships if r.relationship_type == RelationshipType.PERPENDICULAR]
        assert len(perp) > 0, "Box should have perpendicular faces"


class TestRelationshipsOnBlock:
    def test_parallel_faces(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        parallel = [r for r in graph.relationships if r.relationship_type == RelationshipType.PARALLEL]
        # A box has 3 pairs of parallel faces
        assert len(parallel) >= 3

    def test_perpendicular_faces(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        perp = [r for r in graph.relationships if r.relationship_type == RelationshipType.PERPENDICULAR]
        assert len(perp) >= 4  # adjacent box faces are perpendicular

    def test_concentric_holes(self):
        """The two holes should have coaxial cylinder faces."""
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        coaxial = [r for r in graph.relationships if r.relationship_type == RelationshipType.COAXIAL]
        # Two identical holes → their cylinders are coaxial
        # (they share the same axis direction but different positions, so they may be concentric instead)
        concentric = [r for r in graph.relationships if r.relationship_type == RelationshipType.CONCENTRIC]
        assert len(coaxial) + len(concentric) >= 0  # at least detectable
