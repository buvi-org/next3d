"""Tests for streaming export and error handling."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from next3d.ai.streaming import stream_semantic_graph
from next3d.cli.main import cli
from next3d.graph.semantic import build_semantic_graph

FIXTURES = Path(__file__).parent / "fixtures"


class TestStreaming:
    def test_yields_ndjson(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        lines = list(stream_semantic_graph(graph))
        assert len(lines) > 0
        for line in lines:
            obj = json.loads(line)
            assert "chunk_type" in obj

    def test_first_chunk_is_header(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        lines = list(stream_semantic_graph(graph))
        header = json.loads(lines[0])
        assert header["chunk_type"] == "header"
        assert "statistics" in header

    def test_last_chunk_is_footer(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        lines = list(stream_semantic_graph(graph))
        footer = json.loads(lines[-1])
        assert footer["chunk_type"] == "footer"
        assert footer["status"] == "complete"

    def test_chunk_count_matches(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        lines = list(stream_semantic_graph(graph))
        header = json.loads(lines[0])
        stats = header["statistics"]
        expected = (
            1  # header
            + stats["solids"]
            + stats["features"]
            + stats["faces"]
            + stats["edges"]
            + stats["vertices"]
            + stats["adjacency"]
            + stats["relationships"]
            + 1  # footer
        )
        assert len(lines) == expected

    def test_all_chunk_types_present(self):
        graph = build_semantic_graph(FIXTURES / "sample_block.step")
        lines = list(stream_semantic_graph(graph))
        types = {json.loads(l)["chunk_type"] for l in lines}
        assert "header" in types
        assert "footer" in types
        assert "face" in types
        assert "edge" in types


class TestStreamingCLI:
    def test_stream_mode(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["graph", str(FIXTURES / "sample_block.step"), "--mode", "stream"])
        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        assert len(lines) > 2
        header = json.loads(lines[0])
        assert header["chunk_type"] == "header"

    def test_stream_to_file(self, tmp_path):
        output = str(tmp_path / "stream.ndjson")
        runner = CliRunner()
        result = runner.invoke(cli, ["graph", str(FIXTURES / "sample_block.step"),
                                     "--mode", "stream", "-o", output])
        assert result.exit_code == 0
        lines = Path(output).read_text().strip().split("\n")
        assert len(lines) > 2


class TestErrorHandling:
    def test_invalid_step_file(self, tmp_path):
        """Malformed STEP file should fail gracefully."""
        bad_file = tmp_path / "bad.step"
        bad_file.write_text("not a step file")
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", str(bad_file)])
        assert result.exit_code == 1
        assert "Error" in result.output or "error" in result.output

    def test_empty_file(self, tmp_path):
        """Empty file should fail gracefully."""
        empty = tmp_path / "empty.step"
        empty.write_text("")
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(empty)])
        assert result.exit_code == 1

    def test_nonexistent_file(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", "/does/not/exist.step"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_metadata_on_bad_file(self, tmp_path):
        """Metadata extraction on junk should not crash."""
        from next3d.core.step_metadata import extract_metadata
        junk = tmp_path / "junk.step"
        junk.write_text("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;")
        meta = extract_metadata(junk)
        assert meta.total_entities == 0
