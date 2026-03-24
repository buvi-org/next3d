"""Tests for spatial reasoning engine."""

from __future__ import annotations

from pathlib import Path

from next3d.core.brep import load_step
from next3d.core.schema import Vec3
from next3d.core.spatial import bounding_box, point_in_solid

FIXTURES = Path(__file__).parent / "fixtures"


class TestBoundingBox:
    def test_box_dimensions(self):
        """Sample block is 100x60x20."""
        model = load_step(FIXTURES / "sample_block.step")
        bb = bounding_box(model.shape)
        size = bb.size
        # Allow tolerance for fillets
        assert abs(size.x - 100) < 1.0
        assert abs(size.y - 60) < 1.0
        assert abs(size.z - 20) < 1.0

    def test_center(self):
        model = load_step(FIXTURES / "sample_block.step")
        bb = bounding_box(model.shape)
        center = bb.center
        assert abs(center.x) < 1.0
        assert abs(center.y) < 1.0
        assert abs(center.z) < 1.0  # centered at origin

    def test_diagonal(self):
        model = load_step(FIXTURES / "sample_block.step")
        bb = bounding_box(model.shape)
        assert bb.diagonal > 0

    def test_contains_point(self):
        model = load_step(FIXTURES / "sample_block.step")
        bb = bounding_box(model.shape)
        assert bb.contains_point(bb.center)
        assert not bb.contains_point(Vec3(x=1000, y=1000, z=1000))

    def test_to_dict(self):
        model = load_step(FIXTURES / "sample_block.step")
        bb = bounding_box(model.shape)
        d = bb.to_dict()
        assert "min" in d
        assert "max" in d
        assert "size" in d


class TestPointInSolid:
    def test_inside(self):
        model = load_step(FIXTURES / "sample_block.step")
        result = point_in_solid(model.shape, Vec3(x=0, y=0, z=0))
        assert result == "inside"

    def test_outside(self):
        model = load_step(FIXTURES / "sample_block.step")
        result = point_in_solid(model.shape, Vec3(x=500, y=500, z=500))
        assert result == "outside"
