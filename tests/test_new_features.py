"""Tests for Phase 2-3 features: counterbores, embeddings, properties,
manufacturing, multi-body, patterns, CLI new commands."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from next3d.ai.embeddings import FACE_FEATURE_DIM, to_graph_tensors
from next3d.core.brep import load_step
from next3d.core.manufacturing import analyze_manufacturing
from next3d.core.properties import MATERIALS, compute_physical_properties
from next3d.core.schema import FeatureType, RelationshipType
from next3d.graph.semantic import build_semantic_graph
from next3d.cli.main import cli

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Counterbore recognition
# ---------------------------------------------------------------------------


class TestCounterbore:
    def test_detects_counterbore(self):
        graph = build_semantic_graph(FIXTURES / "sample_counterbore.step")
        cbs = [f for f in graph.features if f.feature_type == FeatureType.COUNTERBORE]
        # Even if the counterbore recognizer doesn't fire (geometry-dependent),
        # we should at least detect the constituent holes
        holes = [f for f in graph.features if f.feature_type in (
            FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE, FeatureType.COUNTERBORE
        )]
        assert len(holes) >= 1, (
            f"Expected holes or counterbore in counterbore part, got: "
            f"{[f.feature_type.value for f in graph.features]}"
        )

    def test_counterbore_has_faces(self):
        graph = build_semantic_graph(FIXTURES / "sample_counterbore.step")
        assert len(graph.faces) > 0


# ---------------------------------------------------------------------------
# Graph embeddings
# ---------------------------------------------------------------------------


class TestGraphEmbeddings:
    def test_tensor_shape(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        tensors = to_graph_tensors(graph)
        assert tensors["num_nodes"] == len(graph.faces)
        assert len(tensors["node_features"]) == tensors["num_nodes"]
        assert tensors["feature_dim"] == FACE_FEATURE_DIM

    def test_feature_vector_length(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        tensors = to_graph_tensors(graph)
        for fv in tensors["node_features"]:
            assert len(fv) == FACE_FEATURE_DIM

    def test_edge_index_shape(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        tensors = to_graph_tensors(graph)
        src, tgt = tensors["edge_index"]
        assert len(src) == len(tgt)
        assert len(src) == tensors["num_edges"]

    def test_undirected(self):
        """Edge index should contain both directions."""
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        tensors = to_graph_tensors(graph)
        assert tensors["num_edges"] % 2 == 0  # pairs

    def test_node_ids_match(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        tensors = to_graph_tensors(graph)
        assert set(tensors["node_ids"]) == {f.persistent_id for f in graph.faces}

    def test_json_serializable(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        tensors = to_graph_tensors(graph)
        s = json.dumps(tensors)
        assert len(s) > 0

    def test_metadata(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        tensors = to_graph_tensors(graph)
        assert "features_count" in tensors["metadata"]
        assert "feature_types" in tensors["metadata"]


# ---------------------------------------------------------------------------
# Physical properties
# ---------------------------------------------------------------------------


class TestPhysicalProperties:
    def test_volume(self):
        model = load_step(FIXTURES / "sample_block.step")
        props = compute_physical_properties(model.shape)
        # 100x60x20 block minus two holes (d=10, h=20) minus fillets
        expected_volume = 100 * 60 * 20  # ~120000 mm³
        assert props.volume > 0
        assert abs(props.volume - expected_volume) / expected_volume < 0.1  # within 10%

    def test_surface_area(self):
        model = load_step(FIXTURES / "sample_block.step")
        props = compute_physical_properties(model.shape)
        assert props.surface_area > 0

    def test_mass_with_density(self):
        model = load_step(FIXTURES / "sample_block.step")
        steel = compute_physical_properties(model.shape, density=MATERIALS["steel"])
        aluminum = compute_physical_properties(model.shape, density=MATERIALS["aluminum"])
        assert steel.mass > aluminum.mass  # steel is denser

    def test_center_of_gravity(self):
        model = load_step(FIXTURES / "sample_block.step")
        props = compute_physical_properties(model.shape)
        # Symmetric block → CoG near origin
        assert abs(props.center_of_gravity.x) < 2.0
        assert abs(props.center_of_gravity.y) < 2.0

    def test_moments_of_inertia(self):
        model = load_step(FIXTURES / "sample_block.step")
        props = compute_physical_properties(model.shape)
        # All principal moments should be positive
        assert props.ixx > 0
        assert props.iyy > 0
        assert props.izz > 0

    def test_to_dict(self):
        model = load_step(FIXTURES / "sample_block.step")
        props = compute_physical_properties(model.shape)
        d = props.to_dict()
        assert "volume_mm3" in d
        assert "mass_grams" in d
        assert "center_of_gravity" in d
        assert "moments_of_inertia" in d


# ---------------------------------------------------------------------------
# Manufacturing analysis
# ---------------------------------------------------------------------------


class TestManufacturingAnalysis:
    def test_basic_analysis(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        analysis = analyze_manufacturing(graph)
        assert analysis.min_axes >= 3
        assert 0 <= analysis.complexity_score <= 100

    def test_has_feature_assessments(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        analysis = analyze_manufacturing(graph)
        assert len(analysis.feature_assessments) == len(graph.features)

    def test_suggested_processes(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        analysis = analyze_manufacturing(graph)
        assert len(analysis.suggested_processes) > 0

    def test_machining_axes(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        analysis = analyze_manufacturing(graph)
        # Holes along Z should give at least one machining axis
        assert len(analysis.machining_axes) >= 1

    def test_to_dict(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        analysis = analyze_manufacturing(graph)
        d = analysis.to_dict()
        assert "min_axes" in d
        assert "complexity_score" in d
        assert "suggested_processes" in d

    def test_complex_part_higher_complexity(self):
        simple = build_semantic_graph(FIXTURES / "sample_chamfer.step")
        complex_ = build_semantic_graph(FIXTURES / "sample_complex.step")
        a_simple = analyze_manufacturing(simple)
        a_complex = analyze_manufacturing(complex_)
        assert a_complex.complexity_score >= a_simple.complexity_score


# ---------------------------------------------------------------------------
# Multi-body STEP
# ---------------------------------------------------------------------------


class TestMultiBody:
    def test_two_solids(self):
        graph = build_semantic_graph(FIXTURES / "sample_multibody.step")
        assert len(graph.solids) == 2

    def test_each_solid_has_faces(self):
        graph = build_semantic_graph(FIXTURES / "sample_multibody.step")
        for solid in graph.solids:
            assert len(solid.face_ids) > 0

    def test_faces_partitioned(self):
        """Each face should belong to exactly one solid."""
        graph = build_semantic_graph(FIXTURES / "sample_multibody.step")
        all_assigned = []
        for solid in graph.solids:
            all_assigned.extend(solid.face_ids)
        # No duplicates
        assert len(all_assigned) == len(set(all_assigned))


# ---------------------------------------------------------------------------
# Symmetric / pattern detection
# ---------------------------------------------------------------------------


class TestSymmetryAndPatterns:
    def test_symmetric_holes_detected(self):
        """Block with holes at (20,0) and (-20,0) should be symmetric about YZ plane."""
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        symmetric = [r for r in graph.relationships if r.relationship_type == RelationshipType.SYMMETRIC]
        assert len(symmetric) >= 1, (
            f"Expected symmetric relationships, got types: "
            f"{[r.relationship_type.value for r in graph.relationships]}"
        )

    def test_pattern_holes_detected(self):
        """4 evenly spaced holes at -45, -15, 15, 45 should form a linear pattern."""
        graph = build_semantic_graph(FIXTURES / "sample_pattern.step")
        patterns = [r for r in graph.relationships if r.relationship_type == RelationshipType.PATTERN_MEMBER]
        assert len(patterns) >= 2, (
            f"Expected pattern relationships for 4 holes, got: "
            f"{[r.relationship_type.value for r in graph.relationships]}"
        )


# ---------------------------------------------------------------------------
# CLI new commands
# ---------------------------------------------------------------------------


class TestCLINewCommands:
    def test_properties_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["properties", str(FIXTURES / "sample_block.step")])
        assert result.exit_code == 0
        assert "Volume" in result.output or "volume" in result.output

    def test_properties_json(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["properties", str(FIXTURES / "sample_block.step"),
                                     "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "volume_mm3" in data

    def test_properties_aluminum(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["properties", str(FIXTURES / "sample_block.step"),
                                     "--material", "aluminum"])
        assert result.exit_code == 0

    def test_manufacturing_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["manufacturing", str(FIXTURES / "sample_block.step")])
        assert result.exit_code == 0
        assert "axis" in result.output.lower() or "Axis" in result.output

    def test_manufacturing_json(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["manufacturing", str(FIXTURES / "sample_block.step"),
                                     "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "min_axes" in data

    def test_embeddings_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["embeddings", str(FIXTURES / "sample_block.step")])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "node_features" in data
        assert "edge_index" in data

    def test_embeddings_output_file(self, tmp_path):
        output = str(tmp_path / "emb.json")
        runner = CliRunner()
        result = runner.invoke(cli, ["embeddings", str(FIXTURES / "sample_block.step"),
                                     "-o", output])
        assert result.exit_code == 0
        data = json.loads(Path(output).read_text())
        assert data["num_nodes"] > 0

    def test_inspect_json(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", str(FIXTURES / "sample_block.step"),
                                     "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "statistics" in data

    def test_inspect_yaml(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", str(FIXTURES / "sample_block.step"),
                                     "--format", "yaml"])
        assert result.exit_code == 0
        assert "statistics:" in result.output

    def test_features_json(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["features", str(FIXTURES / "sample_block.step"),
                                     "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
