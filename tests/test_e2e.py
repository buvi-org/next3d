"""End-to-end test: STEP file → semantic graph → query → JSON export.

Uses tests/fixtures/sample_block.step (a block with 2 through holes and fillets).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from next3d.core.brep import load_step, STEPLoadError
from next3d.core.topology import build_topology_graph
from next3d.graph.semantic import build_semantic_graph
from next3d.graph.query import Query, execute_query
from next3d.ai.interface import to_json, to_summary
from next3d.core.schema import SurfaceType, FeatureType

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_STEP = FIXTURES / "sample_block.step"


class TestBRepLoading:
    def test_load_step(self):
        model = load_step(SAMPLE_STEP)
        assert not model.is_null

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_step("/nonexistent/file.step")

    def test_source_path(self):
        model = load_step(SAMPLE_STEP)
        assert model.source_path == SAMPLE_STEP


class TestTopologyGraph:
    def test_build_graph(self):
        model = load_step(SAMPLE_STEP)
        graph, faces, edges, vertices, adjacency = build_topology_graph(model.shape)

        assert len(faces) > 0, "Should have faces"
        assert len(edges) > 0, "Should have edges"
        assert len(vertices) > 0, "Should have vertices"
        assert len(adjacency) > 0, "Should have adjacency relations"

    def test_face_types_present(self):
        model = load_step(SAMPLE_STEP)
        _, faces, _, _, _ = build_topology_graph(model.shape)

        surface_types = {f.surface_type for f in faces}
        # Block with holes should have planes and cylinders
        assert SurfaceType.PLANE in surface_types
        assert SurfaceType.CYLINDER in surface_types

    def test_persistent_ids_unique(self):
        model = load_step(SAMPLE_STEP)
        _, faces, edges, vertices, _ = build_topology_graph(model.shape)

        face_ids = [f.persistent_id for f in faces]
        edge_ids = [e.persistent_id for e in edges]
        vertex_ids = [v.persistent_id for v in vertices]

        # IDs within each category should be unique
        assert len(face_ids) == len(set(face_ids)), "Face IDs not unique"
        assert len(edge_ids) == len(set(edge_ids)), "Edge IDs not unique"
        assert len(vertex_ids) == len(set(vertex_ids)), "Vertex IDs not unique"

    def test_persistent_ids_deterministic(self):
        """Loading the same file twice should produce identical IDs."""
        model1 = load_step(SAMPLE_STEP)
        model2 = load_step(SAMPLE_STEP)
        _, faces1, _, _, _ = build_topology_graph(model1.shape)
        _, faces2, _, _, _ = build_topology_graph(model2.shape)

        ids1 = sorted(f.persistent_id for f in faces1)
        ids2 = sorted(f.persistent_id for f in faces2)
        assert ids1 == ids2


class TestSemanticGraph:
    def test_build_semantic_graph(self):
        graph = build_semantic_graph(SAMPLE_STEP)

        assert len(graph.solids) >= 1
        assert graph.face_count > 0
        assert len(graph.edges) > 0
        assert len(graph.vertices) > 0
        assert len(graph.adjacency) > 0

    def test_solid_has_volume(self):
        graph = build_semantic_graph(SAMPLE_STEP)
        for solid in graph.solids:
            assert solid.volume is not None
            assert solid.volume > 0

    def test_features_detected(self):
        graph = build_semantic_graph(SAMPLE_STEP)
        # Should detect through holes (the 2 holes we created)
        through_holes = [
            f for f in graph.features if f.feature_type == FeatureType.THROUGH_HOLE
        ]
        assert len(through_holes) >= 2, (
            f"Expected at least 2 through holes, got {len(through_holes)}. "
            f"All features: {[f.feature_type.value for f in graph.features]}"
        )

    def test_hole_parameters(self):
        graph = build_semantic_graph(SAMPLE_STEP)
        through_holes = [
            f for f in graph.features if f.feature_type == FeatureType.THROUGH_HOLE
        ]
        for hole in through_holes:
            assert "diameter" in hole.parameters
            assert hole.parameters["diameter"] > 0


class TestQueryOnRealData:
    def test_query_cylindrical_faces(self):
        graph = build_semantic_graph(SAMPLE_STEP)
        q = Query(graph)
        cylinders = q.faces(surface_type="cylinder")
        assert len(cylinders) >= 2  # at least the 2 hole cylinders

    def test_query_planes(self):
        graph = build_semantic_graph(SAMPLE_STEP)
        q = Query(graph)
        planes = q.faces(surface_type="plane")
        assert len(planes) >= 6  # block has 6 main faces minus holes + extra

    def test_query_features(self):
        graph = build_semantic_graph(SAMPLE_STEP)
        q = Query(graph)
        holes = q.features(feature_type="through_hole")
        assert len(holes) >= 2

    def test_query_adjacent(self):
        graph = build_semantic_graph(SAMPLE_STEP)
        q = Query(graph)
        # Cylinders (holes) should be adjacent to planes
        result = q.faces(surface_type="cylinder").adjacent(surface_type="plane")
        assert len(result) > 0

    def test_execute_dsl_query(self):
        graph = build_semantic_graph(SAMPLE_STEP)
        result = execute_query(graph, 'faces(surface_type="cylinder")')
        assert len(result) >= 2


class TestAIExportOnRealData:
    def test_summary_export(self):
        graph = build_semantic_graph(SAMPLE_STEP)
        summary = to_summary(graph)

        assert summary["statistics"]["faces"] > 0
        assert summary["statistics"]["features"] >= 2
        assert "cylinder" in summary["face_type_distribution"]
        assert "plane" in summary["face_type_distribution"]

    def test_detail_export(self):
        graph = build_semantic_graph(SAMPLE_STEP)
        result = to_json(graph, mode="detail")
        data = json.loads(result)

        assert data["type"] == "semantic_3d_graph"
        assert len(data["faces"]) > 0
        assert len(data["adjacency"]) > 0

    def test_summary_has_features(self):
        graph = build_semantic_graph(SAMPLE_STEP)
        summary = to_summary(graph)

        assert len(summary["features"]) >= 2
        for feat in summary["features"]:
            assert "type" in feat
            assert "description" in feat
            assert "parameters" in feat

    def test_json_is_self_contained(self):
        """JSON output should describe the part without needing the STEP file."""
        graph = build_semantic_graph(SAMPLE_STEP)
        result = to_json(graph, mode="summary")
        data = json.loads(result)

        # Must have enough info for an LLM to reason about the part
        assert data["statistics"]["solids"] >= 1
        assert data["statistics"]["faces"] > 0
        assert len(data["features"]) > 0
        assert any(f["description"] for f in data["features"])
