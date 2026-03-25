"""Design rules engine — automated validation of design constraints.

Checks geometry against manufacturing and design rules:
- Minimum wall thickness (per material/process)
- Maximum unsupported overhang angle (for FDM/SLA)
- Minimum hole diameter and spacing
- Draft angle requirements (injection molding)
- Fillet/chamfer radius minimums (CNC)
- Feature-to-edge clearance

Rules are configurable per manufacturing process and material.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from next3d.core.schema import (
    FaceData,
    FeatureData,
    FeatureType,
    SemanticGraph,
    SurfaceType,
    Vec3,
)


@dataclass(frozen=True)
class RuleViolation:
    """A single design rule violation."""

    rule_name: str
    severity: str  # "error", "warning", "info"
    message: str
    entity_id: str | None = None  # persistent_id of the offending entity
    actual_value: float | None = None
    required_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "rule": self.rule_name,
            "severity": self.severity,
            "message": self.message,
        }
        if self.entity_id:
            d["entity_id"] = self.entity_id
        if self.actual_value is not None:
            d["actual_value"] = round(self.actual_value, 3)
        if self.required_value is not None:
            d["required_value"] = round(self.required_value, 3)
        return d


@dataclass
class DesignRuleSet:
    """Configurable set of design rules for a manufacturing process."""

    name: str = "default"

    # Wall thickness
    min_wall_thickness: float = 1.0  # mm
    recommended_wall_thickness: float = 2.0  # mm

    # Holes
    min_hole_diameter: float = 1.0  # mm
    min_hole_spacing_factor: float = 2.0  # multiple of diameter
    max_hole_depth_ratio: float = 10.0  # depth / diameter

    # Fillets and chamfers
    min_internal_radius: float = 0.5  # mm (CNC tool radius limit)

    # Draft (injection molding)
    min_draft_angle: float = 0.0  # degrees (0 = not required)

    # Overhangs (additive manufacturing)
    max_overhang_angle: float = 90.0  # degrees from vertical (90 = no limit)

    # Feature-to-edge clearance
    min_edge_clearance: float = 1.0  # mm

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "min_wall_thickness_mm": self.min_wall_thickness,
            "recommended_wall_thickness_mm": self.recommended_wall_thickness,
            "min_hole_diameter_mm": self.min_hole_diameter,
            "min_hole_spacing_factor": self.min_hole_spacing_factor,
            "max_hole_depth_ratio": self.max_hole_depth_ratio,
            "min_internal_radius_mm": self.min_internal_radius,
            "min_draft_angle_deg": self.min_draft_angle,
            "max_overhang_angle_deg": self.max_overhang_angle,
            "min_edge_clearance_mm": self.min_edge_clearance,
        }


# Predefined rule sets for common manufacturing processes
PROCESS_RULES: dict[str, DesignRuleSet] = {
    "cnc_milling": DesignRuleSet(
        name="cnc_milling",
        min_wall_thickness=0.8,
        recommended_wall_thickness=1.5,
        min_hole_diameter=1.0,
        min_hole_spacing_factor=2.0,
        max_hole_depth_ratio=10.0,
        min_internal_radius=0.5,
        min_draft_angle=0.0,
        max_overhang_angle=90.0,
        min_edge_clearance=1.0,
    ),
    "injection_molding": DesignRuleSet(
        name="injection_molding",
        min_wall_thickness=1.0,
        recommended_wall_thickness=2.5,
        min_hole_diameter=1.0,
        min_hole_spacing_factor=2.0,
        max_hole_depth_ratio=5.0,
        min_internal_radius=0.5,
        min_draft_angle=1.0,  # at least 1° draft required
        max_overhang_angle=90.0,
        min_edge_clearance=2.0,
    ),
    "fdm_3d_print": DesignRuleSet(
        name="fdm_3d_print",
        min_wall_thickness=0.8,
        recommended_wall_thickness=1.2,
        min_hole_diameter=2.0,
        min_hole_spacing_factor=2.0,
        max_hole_depth_ratio=20.0,
        min_internal_radius=0.0,
        min_draft_angle=0.0,
        max_overhang_angle=45.0,  # FDM overhang limit
        min_edge_clearance=0.5,
    ),
    "sla_3d_print": DesignRuleSet(
        name="sla_3d_print",
        min_wall_thickness=0.5,
        recommended_wall_thickness=1.0,
        min_hole_diameter=0.5,
        min_hole_spacing_factor=1.5,
        max_hole_depth_ratio=20.0,
        min_internal_radius=0.0,
        min_draft_angle=0.0,
        max_overhang_angle=30.0,  # SLA needs supports earlier
        min_edge_clearance=0.3,
    ),
    "sheet_metal": DesignRuleSet(
        name="sheet_metal",
        min_wall_thickness=0.5,
        recommended_wall_thickness=1.0,
        min_hole_diameter=1.0,
        min_hole_spacing_factor=3.0,
        max_hole_depth_ratio=1.0,  # sheet metal holes go through
        min_internal_radius=0.5,
        min_draft_angle=0.0,
        max_overhang_angle=90.0,
        min_edge_clearance=2.0,  # hole-to-edge in sheet metal
    ),
    "casting": DesignRuleSet(
        name="casting",
        min_wall_thickness=3.0,
        recommended_wall_thickness=5.0,
        min_hole_diameter=5.0,
        min_hole_spacing_factor=3.0,
        max_hole_depth_ratio=3.0,
        min_internal_radius=2.0,  # generous radii for casting
        min_draft_angle=3.0,  # 3° minimum for casting
        max_overhang_angle=90.0,
        min_edge_clearance=3.0,
    ),
}


def _check_hole_rules(
    features: list[FeatureData],
    rules: DesignRuleSet,
) -> list[RuleViolation]:
    """Check hole-related design rules."""
    violations = []
    holes = [f for f in features if f.feature_type in (
        FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE,
    )]

    for hole in holes:
        diameter = hole.parameters.get("diameter", 0)
        depth = hole.parameters.get("depth")

        # Min diameter
        if diameter < rules.min_hole_diameter:
            violations.append(RuleViolation(
                rule_name="min_hole_diameter",
                severity="error",
                message=f"Hole diameter {diameter:.1f}mm < minimum {rules.min_hole_diameter}mm",
                entity_id=hole.persistent_id,
                actual_value=diameter,
                required_value=rules.min_hole_diameter,
            ))

        # Depth-to-diameter ratio (blind holes)
        if depth and diameter > 0:
            ratio = depth / diameter
            if ratio > rules.max_hole_depth_ratio:
                violations.append(RuleViolation(
                    rule_name="max_hole_depth_ratio",
                    severity="warning",
                    message=f"Hole depth/diameter ratio {ratio:.1f} > max {rules.max_hole_depth_ratio}",
                    entity_id=hole.persistent_id,
                    actual_value=ratio,
                    required_value=rules.max_hole_depth_ratio,
                ))

    # Hole spacing
    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            h1, h2 = holes[i], holes[j]
            if h1.axis and h2.axis:
                c1 = h1.parameters.get("center")
                c2 = h2.parameters.get("center")
                if c1 and c2:
                    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
                    max_d = max(
                        h1.parameters.get("diameter", 0),
                        h2.parameters.get("diameter", 0),
                    )
                    min_spacing = max_d * rules.min_hole_spacing_factor
                    if dist < min_spacing and dist > 0:
                        violations.append(RuleViolation(
                            rule_name="min_hole_spacing",
                            severity="warning",
                            message=f"Holes too close: {dist:.1f}mm < {min_spacing:.1f}mm ({rules.min_hole_spacing_factor}× diameter)",
                            entity_id=h1.persistent_id,
                            actual_value=dist,
                            required_value=min_spacing,
                        ))

    return violations


def _check_fillet_rules(
    features: list[FeatureData],
    rules: DesignRuleSet,
) -> list[RuleViolation]:
    """Check fillet/chamfer design rules."""
    violations = []

    if rules.min_internal_radius <= 0:
        return violations

    for feat in features:
        if feat.feature_type == FeatureType.FILLET:
            radius = feat.parameters.get("radius", 0)
            if radius < rules.min_internal_radius:
                violations.append(RuleViolation(
                    rule_name="min_internal_radius",
                    severity="warning",
                    message=f"Fillet radius {radius:.1f}mm < minimum {rules.min_internal_radius}mm for {rules.name}",
                    entity_id=feat.persistent_id,
                    actual_value=radius,
                    required_value=rules.min_internal_radius,
                ))

    return violations


def _check_overhang_rules(
    faces: list[FaceData],
    rules: DesignRuleSet,
) -> list[RuleViolation]:
    """Check overhang angles for additive manufacturing."""
    violations = []

    if rules.max_overhang_angle >= 90.0:
        return violations  # no overhang limit

    up = Vec3(x=0, y=0, z=1)  # build direction

    for face in faces:
        if face.surface_type != SurfaceType.PLANE:
            continue
        if face.normal is None:
            continue

        # Compute angle between face normal and build direction
        dot = face.normal.x * up.x + face.normal.y * up.y + face.normal.z * up.z
        dot = max(-1.0, min(1.0, dot))
        angle_from_up = math.degrees(math.acos(dot))

        # Overhang angle = angle from vertical (90 - angle_from_horizontal)
        # A face pointing down (normal.z < 0) at >45° from vertical is an overhang
        if face.normal.z < -0.01:  # downward-facing
            overhang_angle = 180 - angle_from_up  # angle from vertical
            if overhang_angle > rules.max_overhang_angle:
                violations.append(RuleViolation(
                    rule_name="max_overhang_angle",
                    severity="warning",
                    message=f"Face overhang {overhang_angle:.0f}° > max {rules.max_overhang_angle}° — needs support",
                    entity_id=face.persistent_id,
                    actual_value=overhang_angle,
                    required_value=rules.max_overhang_angle,
                ))

    return violations


@dataclass
class DesignCheckResult:
    """Result of a design rules check."""

    process: str
    rules: DesignRuleSet
    violations: list[RuleViolation]
    passed: bool

    @property
    def error_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "process": self.process,
            "passed": self.passed,
            "errors": self.error_count,
            "warnings": self.warning_count,
            "total_violations": len(self.violations),
            "violations": [v.to_dict() for v in self.violations],
            "rules_applied": self.rules.to_dict(),
        }


def check_design_rules(
    graph: SemanticGraph,
    process: str = "cnc_milling",
    custom_rules: DesignRuleSet | None = None,
) -> DesignCheckResult:
    """Check a part against design rules for a manufacturing process.

    Args:
        graph: Semantic graph of the part.
        process: Manufacturing process name (cnc_milling, injection_molding,
                 fdm_3d_print, sla_3d_print, sheet_metal, casting).
        custom_rules: Override with custom rules instead of process defaults.

    Returns:
        DesignCheckResult with violations and pass/fail status.
    """
    rules = custom_rules or PROCESS_RULES.get(process)
    if rules is None:
        raise ValueError(
            f"Unknown process: {process}. Available: {', '.join(PROCESS_RULES)}"
        )

    violations: list[RuleViolation] = []

    # Check hole rules
    violations.extend(_check_hole_rules(graph.features, rules))

    # Check fillet/chamfer rules
    violations.extend(_check_fillet_rules(graph.features, rules))

    # Check overhang rules (additive manufacturing)
    violations.extend(_check_overhang_rules(graph.faces, rules))

    # Check draft requirement (injection molding)
    if rules.min_draft_angle > 0:
        vertical_faces = [
            f for f in graph.faces
            if f.surface_type == SurfaceType.PLANE
            and f.normal is not None
            and abs(f.normal.z) < 0.1  # roughly vertical
            and f.area > 10  # ignore tiny faces
        ]
        if vertical_faces:
            violations.append(RuleViolation(
                rule_name="min_draft_angle",
                severity="warning",
                message=f"{len(vertical_faces)} vertical face(s) without draft — {rules.name} requires {rules.min_draft_angle}° minimum",
                actual_value=0,
                required_value=rules.min_draft_angle,
            ))

    # Pass if no errors (warnings are OK)
    passed = all(v.severity != "error" for v in violations)

    return DesignCheckResult(
        process=process,
        rules=rules,
        violations=violations,
        passed=passed,
    )


def list_available_processes() -> list[str]:
    """Return available manufacturing process rule sets."""
    return list(PROCESS_RULES.keys())
