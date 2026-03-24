"""Tests for feature recognition."""

from next3d.core.schema import FeatureType
from next3d.features.holes import HoleRecognizer


class TestHoleRecognizer:
    def test_through_hole_detected(self, sample_faces, sample_edges, sample_adjacency):
        rec = HoleRecognizer()
        features = rec.recognize(sample_faces, sample_edges, sample_adjacency)

        through_holes = [f for f in features if f.feature_type == FeatureType.THROUGH_HOLE]
        assert len(through_holes) == 1

        hole = through_holes[0]
        assert hole.parameters["diameter"] == 10.0
        assert hole.parameters["radius"] == 5.0
        assert "face_cyl_001" in hole.face_ids

    def test_no_false_positives_on_planes(self, sample_faces, sample_edges, sample_adjacency):
        """Plane faces should not be recognized as holes."""
        rec = HoleRecognizer()
        features = rec.recognize(sample_faces, sample_edges, sample_adjacency)
        for f in features:
            assert "face_plane" not in f.face_ids[0]
