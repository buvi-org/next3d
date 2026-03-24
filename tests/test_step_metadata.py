"""Tests for STEP metadata extraction."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from next3d.cli.main import cli
from next3d.core.step_metadata import extract_metadata

FIXTURES = Path(__file__).parent / "fixtures"


class TestSTEPMetadata:
    def test_header_parsed(self):
        meta = extract_metadata(FIXTURES / "sample_complex.step")
        assert meta.header.schema != ""
        assert meta.header.ap_version in ("AP203", "AP214", "AP242")

    def test_originating_system(self):
        meta = extract_metadata(FIXTURES / "sample_complex.step")
        assert "Open CASCADE" in meta.header.originating_system

    def test_products_found(self):
        meta = extract_metadata(FIXTURES / "sample_complex.step")
        assert len(meta.products) >= 1

    def test_entity_counts(self):
        meta = extract_metadata(FIXTURES / "sample_complex.step")
        assert meta.total_entities > 0
        assert "ADVANCED_FACE" in meta.entity_counts
        assert "CARTESIAN_POINT" in meta.entity_counts

    def test_to_dict(self):
        meta = extract_metadata(FIXTURES / "sample_complex.step")
        d = meta.to_dict()
        assert "header" in d
        assert "products" in d
        assert "entity_statistics" in d
        assert "capabilities" in d

    def test_multibody_products(self):
        meta = extract_metadata(FIXTURES / "sample_multibody.step")
        assert meta.total_entities > 0

    def test_no_false_pmi(self):
        """CadQuery-generated files should not report PMI."""
        meta = extract_metadata(FIXTURES / "sample_block.step")
        assert not meta.has_pmi

    def test_no_false_assembly(self):
        """Single-body file should not report assembly."""
        meta = extract_metadata(FIXTURES / "sample_block.step")
        assert not meta.has_assembly


class TestMetadataCLI:
    def test_metadata_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["metadata", str(FIXTURES / "sample_block.step")])
        assert result.exit_code == 0
        assert "Schema" in result.output or "schema" in result.output

    def test_metadata_json(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["metadata", str(FIXTURES / "sample_block.step"),
                                     "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "header" in data


class TestMatingCLI:
    def test_mating_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["mating", str(FIXTURES / "sample_complex.step")])
        assert result.exit_code == 0

    def test_mating_json(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["mating", str(FIXTURES / "sample_complex.step"),
                                     "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
