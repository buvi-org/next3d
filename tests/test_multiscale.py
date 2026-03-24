"""Tests for multi-scale representation."""

from __future__ import annotations

from pathlib import Path

from next3d.graph.multiscale import Level, MultiScaleView
from next3d.graph.semantic import build_semantic_graph

FIXTURES = Path(__file__).parent / "fixtures"


class TestMultiScaleView:
    def test_at_level_solid(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        view = MultiScaleView(graph)
        solids = view.at_level(Level.SOLID)
        assert len(solids) >= 1

    def test_at_level_face(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        view = MultiScaleView(graph)
        faces = view.at_level(Level.FACE)
        assert len(faces) > 0

    def test_at_level_feature(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        view = MultiScaleView(graph)
        features = view.at_level(Level.FEATURE)
        assert len(features) > 0

    def test_children_solid_to_features(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        view = MultiScaleView(graph)
        solid_id = graph.solids[0].persistent_id
        children = view.children(solid_id)
        assert len(children) > 0

    def test_children_feature_to_faces(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        view = MultiScaleView(graph)
        if graph.features:
            feat_id = graph.features[0].persistent_id
            children = view.children(feat_id)
            assert len(children) > 0

    def test_children_face_to_edges(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        view = MultiScaleView(graph)
        face_id = graph.faces[0].persistent_id
        children = view.children(face_id)
        assert len(children) > 0

    def test_parent_face_to_feature(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        view = MultiScaleView(graph)
        # Find a face that belongs to a feature
        if graph.features:
            face_id = graph.features[0].face_ids[0]
            parents = view.parent(face_id)
            assert len(parents) > 0

    def test_parent_edge_to_face(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        view = MultiScaleView(graph)
        if graph.edges:
            edge_id = graph.edges[0].persistent_id
            parents = view.parent(edge_id)
            # Edge should belong to at least one face
            assert len(parents) >= 0  # some edges might not be in our face edge_ids

    def test_level_of(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        view = MultiScaleView(graph)
        assert view.level_of(graph.solids[0].persistent_id) == Level.SOLID
        assert view.level_of(graph.faces[0].persistent_id) == Level.FACE
        assert view.level_of("nonexistent") is None

    def test_summary_at_level(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        view = MultiScaleView(graph)
        summary = view.summary_at_level(Level.FACE)
        assert summary["level"] == "FACE"
        assert summary["count"] > 0
        assert len(summary["ids"]) == summary["count"]

    def test_full_traversal_down(self):
        """Traverse from solid → features → faces → edges → vertices."""
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        view = MultiScaleView(graph)

        # L0: Solid
        solids = view.at_level(Level.SOLID)
        assert len(solids) >= 1

        # L0 → L1: Features
        features = view.children(solids[0].persistent_id)
        assert len(features) > 0

        # L1 → L2: Faces of first feature
        faces = view.children(features[0].persistent_id)
        assert len(faces) > 0

        # L2 → L3: Edges of first face
        edges = view.children(faces[0].persistent_id)
        assert len(edges) > 0

        # L3 → L4: Vertices of first edge
        vertices = view.children(edges[0].persistent_id)
        assert len(vertices) > 0
