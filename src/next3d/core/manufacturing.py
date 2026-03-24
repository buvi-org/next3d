"""Manufacturing semantics analysis.

Provides machinability assessment, machining axis requirements,
and manufacturing process suggestions based on feature analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from next3d.core.schema import (
    FeatureData,
    FeatureType,
    SemanticGraph,
    Vec3,
)
from next3d.core.spatial import BoundingBox, bounding_box


@dataclass
class MachiningAxis:
    """A required machining access direction."""

    direction: Vec3
    features: list[str]  # feature persistent_ids needing this axis
    description: str


@dataclass
class ManufacturingAnalysis:
    """Manufacturing semantics for a part."""

    # Axis count needed (2-axis, 3-axis, 5-axis, etc.)
    min_axes: int
    machining_axes: list[MachiningAxis]

    # Feature-level manufacturability
    feature_assessments: list[dict[str, Any]]

    # Overall complexity score (0-100)
    complexity_score: float

    # Suggested processes
    suggested_processes: list[str]

    # Potential issues
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "min_axes": self.min_axes,
            "complexity_score": round(self.complexity_score, 1),
            "suggested_processes": self.suggested_processes,
            "machining_axes": [
                {
                    "direction": {"x": a.direction.x, "y": a.direction.y, "z": a.direction.z},
                    "feature_count": len(a.features),
                    "description": a.description,
                }
                for a in self.machining_axes
            ],
            "feature_assessments": self.feature_assessments,
            "warnings": self.warnings,
        }


def _normalize(v: Vec3) -> Vec3:
    m = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
    if m < 1e-12:
        return v
    return Vec3(x=v.x / m, y=v.y / m, z=v.z / m)


def _axes_equivalent(a: Vec3, b: Vec3, tol: float = 0.05) -> bool:
    """Check if two axes are the same (or opposite)."""
    dot = abs(a.x * b.x + a.y * b.y + a.z * b.z)
    return dot > 1.0 - tol


def _cluster_axes(features: list[FeatureData]) -> list[MachiningAxis]:
    """Cluster features by their machining access axis."""
    axes: list[MachiningAxis] = []

    for feat in features:
        if feat.axis is None:
            continue
        norm = _normalize(feat.axis)

        # Try to merge with existing axis cluster
        merged = False
        for existing in axes:
            if _axes_equivalent(norm, existing.direction):
                existing.features.append(feat.persistent_id)
                merged = True
                break

        if not merged:
            label = _axis_label(norm)
            axes.append(
                MachiningAxis(
                    direction=norm,
                    features=[feat.persistent_id],
                    description=f"Machining access along {label}",
                )
            )

    return axes


def _axis_label(v: Vec3) -> str:
    """Human-readable label for an axis direction."""
    ax, ay, az = abs(v.x), abs(v.y), abs(v.z)
    if az > 0.95:
        return "+Z" if v.z > 0 else "-Z"
    if ay > 0.95:
        return "+Y" if v.y > 0 else "-Y"
    if ax > 0.95:
        return "+X" if v.x > 0 else "-X"
    return f"({v.x:.2f}, {v.y:.2f}, {v.z:.2f})"


def _assess_feature(feat: FeatureData) -> dict[str, Any]:
    """Assess manufacturability of a single feature."""
    assessment: dict[str, Any] = {
        "feature_id": feat.persistent_id,
        "feature_type": feat.feature_type.value,
        "description": feat.description,
    }

    if feat.feature_type == FeatureType.THROUGH_HOLE:
        diameter = feat.parameters.get("diameter", 0)
        assessment["process"] = "drilling"
        assessment["tool"] = f"drill bit d={diameter:.1f}mm"
        if diameter < 1.0:
            assessment["difficulty"] = "high"
            assessment["notes"] = "Very small hole — may need EDM or laser"
        elif diameter > 50:
            assessment["difficulty"] = "medium"
            assessment["notes"] = "Large hole — may need boring bar"
        else:
            assessment["difficulty"] = "low"

    elif feat.feature_type == FeatureType.BLIND_HOLE:
        diameter = feat.parameters.get("diameter", 0)
        assessment["process"] = "drilling"
        assessment["tool"] = f"drill bit d={diameter:.1f}mm"
        assessment["difficulty"] = "low"

    elif feat.feature_type == FeatureType.FILLET:
        radius = feat.parameters.get("radius", 0)
        assessment["process"] = "milling"
        assessment["tool"] = f"ball end mill r={radius:.1f}mm"
        assessment["difficulty"] = "low" if radius > 1.0 else "medium"

    elif feat.feature_type == FeatureType.CHAMFER:
        assessment["process"] = "chamfer mill or deburring"
        assessment["difficulty"] = "low"

    elif feat.feature_type == FeatureType.SLOT:
        width = feat.parameters.get("width", 0)
        assessment["process"] = "milling"
        assessment["tool"] = f"end mill d<={width:.1f}mm"
        assessment["difficulty"] = "low" if width > 3.0 else "medium"

    elif feat.feature_type == FeatureType.BOSS:
        assessment["process"] = "milling (pocket around boss)"
        assessment["difficulty"] = "medium"

    elif feat.feature_type == FeatureType.COUNTERBORE:
        assessment["process"] = "drilling + counterbore tool"
        assessment["difficulty"] = "low"

    elif feat.feature_type == FeatureType.COUNTERSINK:
        assessment["process"] = "drilling + countersink tool"
        assessment["difficulty"] = "low"

    return assessment


def analyze_manufacturing(graph: SemanticGraph) -> ManufacturingAnalysis:
    """Analyze manufacturing requirements for a part.

    Args:
        graph: The semantic graph of the part.

    Returns:
        ManufacturingAnalysis with machining axes, complexity, and process suggestions.
    """
    features = graph.features

    # Cluster by machining axis
    machining_axes = _cluster_axes(features)
    num_unique_axes = len(machining_axes)

    # Determine minimum axis count
    if num_unique_axes <= 1:
        min_axes = 3  # simple 3-axis milling
    elif num_unique_axes == 2:
        # Check if the two axes are orthogonal — still 3-axis capable
        if len(machining_axes) >= 2:
            a1 = machining_axes[0].direction
            a2 = machining_axes[1].direction
            dot = abs(a1.x * a2.x + a1.y * a2.y + a1.z * a2.z)
            min_axes = 3 if dot < 0.1 else 4
        else:
            min_axes = 3
    elif num_unique_axes <= 3:
        min_axes = 4
    else:
        min_axes = 5

    # Assess each feature
    assessments = [_assess_feature(f) for f in features]

    # Complexity score (0-100)
    complexity = 0.0
    complexity += min(len(features) * 5, 30)  # feature count
    complexity += num_unique_axes * 10  # axis diversity
    complexity += len(graph.faces) * 0.3  # face count contribution

    # Difficulty bonuses
    for a in assessments:
        if a.get("difficulty") == "high":
            complexity += 10
        elif a.get("difficulty") == "medium":
            complexity += 5
    complexity = min(complexity, 100)

    # Process suggestions
    processes = set()
    for a in assessments:
        p = a.get("process", "")
        if p:
            processes.add(p.split(" +")[0].split(" (")[0])  # base process
    if not processes:
        processes = {"milling"}

    # Warnings
    warnings: list[str] = []
    if min_axes >= 5:
        warnings.append("Part requires 5-axis machining — consider redesign for 3-axis")
    small_features = [a for a in assessments if a.get("difficulty") == "high"]
    if small_features:
        warnings.append(f"{len(small_features)} feature(s) with high manufacturing difficulty")

    return ManufacturingAnalysis(
        min_axes=min_axes,
        machining_axes=machining_axes,
        feature_assessments=assessments,
        complexity_score=complexity,
        suggested_processes=sorted(processes),
        warnings=warnings,
    )
