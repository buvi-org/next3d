"""Query DSL for geometry selection.

Provides a fluent interface and string-based DSL for selecting
topological entities and features from a SemanticGraph.

Examples:
    q = Query(graph)
    q.faces(surface_type="cylinder", radius=5.0)
    q.features(feature_type="through_hole", diameter__gt=8.0)
    q.faces(surface_type="cylinder").adjacent(surface_type="plane")
"""

from __future__ import annotations

import operator
import re
from typing import Any

from next3d.core.schema import (
    EdgeData,
    EdgeRelationType,
    FaceData,
    FeatureData,
    SemanticGraph,
    VertexData,
)


# Comparison operators for filter suffixes
_OPS = {
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
    "ne": operator.ne,
}


def _match(entity: Any, filters: dict[str, Any]) -> bool:
    """Check if an entity matches all filters.

    Supports:
        field=value        → exact match
        field__gt=value    → greater than
        field__lt=value    → less than
        field__gte=value   → greater or equal
        field__lte=value   → less or equal
        field__ne=value    → not equal
    """
    for key, expected in filters.items():
        parts = key.split("__")
        field = parts[0]
        op_name = parts[1] if len(parts) > 1 else None

        # Get field value — support nested dict access for parameters
        if hasattr(entity, field):
            actual = getattr(entity, field)
        elif hasattr(entity, "parameters") and field in entity.parameters:
            actual = entity.parameters[field]
        else:
            return False

        # Handle enum comparison
        if hasattr(actual, "value"):
            actual = actual.value

        if actual is None:
            return False

        if op_name is None:
            if actual != expected:
                return False
        elif op_name in _OPS:
            if not _OPS[op_name](actual, expected):
                return False
        else:
            return False

    return True


class QueryResult:
    """A set of query results that supports chaining."""

    def __init__(self, graph: SemanticGraph, entities: list[Any]):
        self._graph = graph
        self._entities = entities

    @property
    def entities(self) -> list[Any]:
        return list(self._entities)

    def __len__(self) -> int:
        return len(self._entities)

    def __iter__(self):
        return iter(self._entities)

    def first(self) -> Any | None:
        return self._entities[0] if self._entities else None

    def adjacent(self, **filters: Any) -> QueryResult:
        """Find faces adjacent to the current result set, optionally filtered."""
        if not self._entities:
            return QueryResult(self._graph, [])

        # Collect IDs of current faces
        current_ids = set()
        for e in self._entities:
            if isinstance(e, FaceData):
                current_ids.add(e.persistent_id)

        # Find adjacent face IDs
        adj_ids: set[str] = set()
        for fid in current_ids:
            adj_ids.update(self._graph.faces_adjacent_to(fid))
        adj_ids -= current_ids  # exclude self

        # Filter adjacent faces
        results = []
        for face in self._graph.faces:
            if face.persistent_id in adj_ids and _match(face, filters):
                results.append(face)

        return QueryResult(self._graph, results)


class Query:
    """Query interface for a SemanticGraph."""

    def __init__(self, graph: SemanticGraph):
        self._graph = graph

    def faces(self, **filters: Any) -> QueryResult:
        """Select faces matching the given filters."""
        results = [f for f in self._graph.faces if _match(f, filters)]
        return QueryResult(self._graph, results)

    def edges(self, **filters: Any) -> QueryResult:
        """Select edges matching the given filters."""
        results = [e for e in self._graph.edges if _match(e, filters)]
        return QueryResult(self._graph, results)

    def vertices(self, **filters: Any) -> QueryResult:
        """Select vertices matching the given filters."""
        results = [v for v in self._graph.vertices if _match(v, filters)]
        return QueryResult(self._graph, results)

    def features(self, **filters: Any) -> QueryResult:
        """Select features matching the given filters."""
        results = [f for f in self._graph.features if _match(f, filters)]
        return QueryResult(self._graph, results)


def parse_query_string(query_str: str) -> tuple[str, dict[str, Any]]:
    """Parse a simple DSL query string into method name and filters.

    Format: 'entity_type(key=value, key__op=value, ...)'

    Examples:
        'faces(surface_type="cylinder", radius=5.0)'
        'features(feature_type="through_hole", diameter__gt=8.0)'

    Returns:
        Tuple of (method_name, filters_dict)
    """
    match = re.match(r"(\w+)\((.*)?\)", query_str.strip())
    if not match:
        raise ValueError(f"Invalid query: {query_str}")

    method = match.group(1)
    args_str = match.group(2) or ""

    filters: dict[str, Any] = {}
    if args_str.strip():
        for part in args_str.split(","):
            part = part.strip()
            key, _, value = part.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")

            # Try numeric conversion
            try:
                value = float(value)
                if value == int(value):
                    value = int(value)
            except ValueError:
                pass

            filters[key] = value

    return method, filters


def execute_query(graph: SemanticGraph, query_str: str) -> QueryResult:
    """Execute a DSL query string against a SemanticGraph."""
    method, filters = parse_query_string(query_str)
    q = Query(graph)

    if not hasattr(q, method):
        raise ValueError(f"Unknown query method: {method}")

    return getattr(q, method)(**filters)
