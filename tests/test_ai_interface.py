"""Tests for the AI interface export."""

import json

from next3d.ai.interface import to_summary, to_detail, to_json


class TestAIInterface:
    def test_summary_structure(self, sample_graph):
        summary = to_summary(sample_graph)
        assert summary["type"] == "semantic_3d_summary"
        assert summary["statistics"]["faces"] == 3
        assert "cylinder" in summary["face_type_distribution"]

    def test_detail_structure(self, sample_graph):
        detail = to_detail(sample_graph)
        assert detail["type"] == "semantic_3d_graph"
        assert len(detail["faces"]) == 3

    def test_to_json_summary(self, sample_graph):
        result = to_json(sample_graph, mode="summary")
        data = json.loads(result)
        assert data["type"] == "semantic_3d_summary"

    def test_to_json_detail(self, sample_graph):
        result = to_json(sample_graph, mode="detail")
        data = json.loads(result)
        assert data["type"] == "semantic_3d_graph"
        assert len(data["faces"]) == 3

    def test_json_valid(self, sample_graph):
        """Output must be valid JSON."""
        for mode in ("summary", "detail"):
            result = to_json(sample_graph, mode=mode)
            json.loads(result)  # should not raise
