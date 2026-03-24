"""Feature recognition orchestrator.

Runs all registered feature recognizers against the semantic graph
and returns the discovered features.
"""

from __future__ import annotations

from typing import Protocol

from next3d.core.schema import FaceData, EdgeData, AdjacencyEdge, FeatureData


class FeatureRecognizer(Protocol):
    """Protocol for pluggable feature recognizers."""

    def recognize(
        self,
        faces: list[FaceData],
        edges: list[EdgeData],
        adjacency: list[AdjacencyEdge],
    ) -> list[FeatureData]: ...


# Registry of recognizers — new recognizers are appended here
_registry: list[FeatureRecognizer] = []


def register(recognizer: FeatureRecognizer) -> None:
    """Register a feature recognizer."""
    _registry.append(recognizer)


def recognize_all(
    faces: list[FaceData],
    edges: list[EdgeData],
    adjacency: list[AdjacencyEdge],
) -> list[FeatureData]:
    """Run all registered recognizers and collect features."""
    features: list[FeatureData] = []
    for rec in _registry:
        features.extend(rec.recognize(faces, edges, adjacency))
    return features
