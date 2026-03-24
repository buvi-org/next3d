"""Assembly mating logic and fit analysis.

Analyzes potential mating conditions between features:
- Shaft-hole fits (clearance, transition, interference)
- Planar contact (coplanar faces across solids)
- Concentric alignment
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from next3d.core.schema import (
    FeatureData,
    FeatureType,
    Relationship,
    RelationshipType,
    SemanticGraph,
    Vec3,
)


@dataclass(frozen=True)
class MatingCondition:
    """A detected mating condition between two features or faces."""

    source_id: str
    target_id: str
    mate_type: str  # 'clearance_fit', 'transition_fit', 'interference_fit', 'planar_contact', 'axial_alignment'
    parameters: dict[str, Any]
    description: str


@dataclass(frozen=True)
class FitAnalysis:
    """Shaft-hole fit analysis per ISO 286."""

    shaft_diameter: float
    hole_diameter: float
    clearance: float  # positive = clearance, negative = interference
    fit_type: str  # 'clearance', 'transition', 'interference'
    description: str


def analyze_fit(hole_diameter: float, shaft_diameter: float) -> FitAnalysis:
    """Analyze fit between a hole and a shaft.

    Args:
        hole_diameter: Nominal hole diameter (mm).
        shaft_diameter: Nominal shaft diameter (mm).

    Returns:
        FitAnalysis with clearance and fit classification.
    """
    clearance = hole_diameter - shaft_diameter

    if clearance > 0.01:
        fit_type = "clearance"
        desc = f"Clearance fit, gap={clearance:.3f}mm"
    elif clearance < -0.01:
        fit_type = "interference"
        desc = f"Interference fit, overlap={abs(clearance):.3f}mm"
    else:
        fit_type = "transition"
        desc = f"Transition fit, clearance={clearance:.3f}mm"

    return FitAnalysis(
        shaft_diameter=shaft_diameter,
        hole_diameter=hole_diameter,
        clearance=clearance,
        fit_type=fit_type,
        description=desc,
    )


def _feature_diameter(feat: FeatureData) -> float | None:
    """Extract diameter from a feature's parameters."""
    return feat.parameters.get("diameter")


def _features_coaxial(f1: FeatureData, f2: FeatureData, tol_deg: float = 5.0) -> bool:
    """Check if two features share the same axis."""
    if f1.axis is None or f2.axis is None:
        return False
    dot = abs(f1.axis.x * f2.axis.x + f1.axis.y * f2.axis.y + f1.axis.z * f2.axis.z)
    mag1 = math.sqrt(f1.axis.x ** 2 + f1.axis.y ** 2 + f1.axis.z ** 2)
    mag2 = math.sqrt(f2.axis.x ** 2 + f2.axis.y ** 2 + f2.axis.z ** 2)
    if mag1 < 1e-12 or mag2 < 1e-12:
        return False
    cos_angle = dot / (mag1 * mag2)
    return cos_angle > math.cos(math.radians(tol_deg))


def detect_mating_conditions(graph: SemanticGraph) -> list[MatingCondition]:
    """Detect potential mating conditions between features.

    Analyzes:
    1. Hole-boss pairs that could form shaft-hole fits
    2. Coplanar face pairs that suggest planar mating surfaces
    3. Coaxial features for axial alignment

    Args:
        graph: The semantic graph.

    Returns:
        List of detected mating conditions.
    """
    conditions: list[MatingCondition] = []

    holes = [f for f in graph.features if f.feature_type in (
        FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE
    )]
    bosses = [f for f in graph.features if f.feature_type == FeatureType.BOSS]

    # 1. Hole-boss fit analysis
    for hole in holes:
        hole_d = _feature_diameter(hole)
        if hole_d is None:
            continue
        for boss in bosses:
            boss_d = _feature_diameter(boss)
            if boss_d is None:
                continue
            # Check if they could mate (coaxial or close diameters)
            if _features_coaxial(hole, boss) or abs(hole_d - boss_d) < max(hole_d, boss_d) * 0.5:
                fit = analyze_fit(hole_d, boss_d)
                conditions.append(
                    MatingCondition(
                        source_id=hole.persistent_id,
                        target_id=boss.persistent_id,
                        mate_type=f"{fit.fit_type}_fit",
                        parameters={
                            "hole_diameter": hole_d,
                            "shaft_diameter": boss_d,
                            "clearance": fit.clearance,
                        },
                        description=fit.description,
                    )
                )

    # 2. Coplanar relationships as potential planar contacts
    for rel in graph.relationships:
        if rel.relationship_type == RelationshipType.COPLANAR:
            conditions.append(
                MatingCondition(
                    source_id=rel.source_id,
                    target_id=rel.target_id,
                    mate_type="planar_contact",
                    parameters={},
                    description="Coplanar faces — potential planar mating surface",
                )
            )

    # 3. Coaxial relationships as alignment conditions
    for rel in graph.relationships:
        if rel.relationship_type == RelationshipType.COAXIAL:
            conditions.append(
                MatingCondition(
                    source_id=rel.source_id,
                    target_id=rel.target_id,
                    mate_type="axial_alignment",
                    parameters=rel.parameters,
                    description="Coaxial features — axial alignment mate",
                )
            )

    return conditions
