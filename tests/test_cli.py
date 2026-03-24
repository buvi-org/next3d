"""Tests for the CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from next3d.cli.main import cli

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_STEP = str(FIXTURES / "sample_block.step")


class TestCLI:
    def test_inspect(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", SAMPLE_STEP])
        assert result.exit_code == 0
        assert "Statistics" in result.output

    def test_graph_detail(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["graph", SAMPLE_STEP, "--mode", "detail"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["type"] == "semantic_3d_graph"
        assert len(data["faces"]) > 0

    def test_graph_summary(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["graph", SAMPLE_STEP, "--mode", "summary"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["type"] == "semantic_3d_summary"

    def test_features(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["features", SAMPLE_STEP])
        assert result.exit_code == 0
        assert "through_hole" in result.output

    def test_query(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["query", SAMPLE_STEP, 'faces(surface_type="cylinder")'])
        assert result.exit_code == 0
        assert "Results" in result.output

    def test_validate(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", SAMPLE_STEP])
        assert result.exit_code == 0

    def test_inspect_nonexistent(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", "/nonexistent.step"])
        assert result.exit_code == 1

    def test_graph_output_file(self, tmp_path):
        output = str(tmp_path / "out.json")
        runner = CliRunner()
        result = runner.invoke(cli, ["graph", SAMPLE_STEP, "-o", output])
        assert result.exit_code == 0
        data = json.loads(Path(output).read_text())
        assert data["type"] == "semantic_3d_graph"
