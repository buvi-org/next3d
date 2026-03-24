"""Tests for the query DSL."""

from next3d.core.schema import SurfaceType
from next3d.graph.query import Query, parse_query_string, execute_query


class TestQuery:
    def test_faces_by_type(self, sample_graph):
        q = Query(sample_graph)
        result = q.faces(surface_type="cylinder")
        assert len(result) == 1
        assert result.first().persistent_id == "face_cyl_001"

    def test_faces_by_radius(self, sample_graph):
        q = Query(sample_graph)
        result = q.faces(radius=5.0)
        assert len(result) == 1

    def test_faces_gt_filter(self, sample_graph):
        q = Query(sample_graph)
        result = q.faces(area__gt=500)
        assert len(result) == 2  # the two planes

    def test_adjacent_chain(self, sample_graph):
        q = Query(sample_graph)
        result = q.faces(surface_type="cylinder").adjacent(surface_type="plane")
        assert len(result) == 2

    def test_no_results(self, sample_graph):
        q = Query(sample_graph)
        result = q.faces(surface_type="sphere")
        assert len(result) == 0

    def test_edges_by_type(self, sample_graph):
        q = Query(sample_graph)
        result = q.edges(curve_type="circle")
        assert len(result) == 2


class TestParseQueryString:
    def test_simple(self):
        method, filters = parse_query_string('faces(surface_type="cylinder")')
        assert method == "faces"
        assert filters == {"surface_type": "cylinder"}

    def test_numeric(self):
        method, filters = parse_query_string("faces(radius=5.0)")
        assert method == "faces"
        assert filters == {"radius": 5.0}

    def test_comparison(self):
        method, filters = parse_query_string("faces(area__gt=100)")
        assert method == "faces"
        assert filters == {"area__gt": 100}

    def test_multiple_filters(self):
        method, filters = parse_query_string('faces(surface_type="cylinder", radius=5.0)')
        assert method == "faces"
        assert filters == {"surface_type": "cylinder", "radius": 5.0}


class TestExecuteQuery:
    def test_execute(self, sample_graph):
        result = execute_query(sample_graph, 'faces(surface_type="cylinder")')
        assert len(result) == 1
