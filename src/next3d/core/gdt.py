"""GD&T (Geometric Dimensioning and Tolerancing) annotation system.

Models GD&T annotations per ASME Y14.5 / ISO 1101, including:
- Datum references (A, B, C labels tied to geometric features)
- Tolerance zones (form, orientation, location, runout, profile)
- Validation against the semantic graph
- Auto-suggestion based on feature analysis

This is metadata — it does not modify geometry, but annotates it
for manufacturing, inspection, and downstream tooling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from next3d.core.schema import SemanticGraph


class ToleranceType(str, Enum):
    """GD&T tolerance types per ASME Y14.5 / ISO 1101."""

    # Form
    FLATNESS = "flatness"
    STRAIGHTNESS = "straightness"
    CIRCULARITY = "circularity"
    CYLINDRICITY = "cylindricity"

    # Orientation
    PARALLELISM = "parallelism"
    PERPENDICULARITY = "perpendicularity"
    ANGULARITY = "angularity"

    # Location
    POSITION = "position"
    CONCENTRICITY = "concentricity"
    SYMMETRY = "symmetry"

    # Runout
    CIRCULAR_RUNOUT = "circular_runout"
    TOTAL_RUNOUT = "total_runout"

    # Profile
    PROFILE_OF_LINE = "profile_of_line"
    PROFILE_OF_SURFACE = "profile_of_surface"


# Tolerance types that require at least one datum reference
_REQUIRES_DATUM = {
    ToleranceType.PARALLELISM,
    ToleranceType.PERPENDICULARITY,
    ToleranceType.ANGULARITY,
    ToleranceType.POSITION,
    ToleranceType.CONCENTRICITY,
    ToleranceType.SYMMETRY,
    ToleranceType.CIRCULAR_RUNOUT,
    ToleranceType.TOTAL_RUNOUT,
}

# Valid material condition modifiers
VALID_MATERIAL_CONDITIONS = {"MMC", "LMC", "RFS", ""}


@dataclass
class DatumReference:
    """A datum reference — a labeled feature used as a reference for tolerancing."""

    label: str  # "A", "B", "C"
    entity_id: str  # persistent_id of the datum feature
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "label": self.label,
            "entity_id": self.entity_id,
        }
        if self.description:
            result["description"] = self.description
        return result


@dataclass
class ToleranceZone:
    """A GD&T tolerance zone applied to a controlled feature."""

    tolerance_type: str  # from ToleranceType enum
    value: float  # tolerance value in mm
    entity_id: str  # persistent_id of the controlled feature
    datum_refs: list[str] = field(default_factory=list)  # datum labels, e.g. ["A", "B"]
    material_condition: str = ""  # "MMC", "LMC", "RFS", or ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "tolerance_type": self.tolerance_type,
            "value": self.value,
            "entity_id": self.entity_id,
        }
        if self.datum_refs:
            result["datum_refs"] = self.datum_refs
        if self.material_condition:
            result["material_condition"] = self.material_condition
        if self.description:
            result["description"] = self.description
        return result


@dataclass
class GDTAnnotationSet:
    """Complete set of GD&T annotations for a body."""

    datums: list[DatumReference] = field(default_factory=list)
    tolerances: list[ToleranceZone] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "datums": [d.to_dict() for d in self.datums],
            "tolerances": [t.to_dict() for t in self.tolerances],
            "datum_count": len(self.datums),
            "tolerance_count": len(self.tolerances),
        }


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_datum(
    label: str,
    entity_id: str,
    description: str = "",
) -> DatumReference:
    """Create a datum reference.

    Args:
        label: Datum label (e.g. "A", "B", "C"). Must be a single uppercase letter.
        entity_id: persistent_id of the datum feature.
        description: Optional human-readable description.

    Returns:
        DatumReference
    """
    if not label or not label.isalpha() or not label.isupper():
        raise ValueError(f"Datum label must be an uppercase letter, got: '{label}'")
    return DatumReference(label=label, entity_id=entity_id, description=description)


def create_tolerance(
    tolerance_type: str,
    value: float,
    entity_id: str,
    datum_refs: list[str] | None = None,
    material_condition: str = "",
    description: str = "",
) -> ToleranceZone:
    """Create a tolerance zone.

    Args:
        tolerance_type: One of the ToleranceType enum values.
        value: Tolerance value in mm (must be > 0).
        entity_id: persistent_id of the controlled feature.
        datum_refs: Datum labels this tolerance references.
        material_condition: "MMC", "LMC", "RFS", or "".
        description: Optional human-readable description.

    Returns:
        ToleranceZone
    """
    # Validate tolerance type
    try:
        ToleranceType(tolerance_type)
    except ValueError:
        valid = ", ".join(t.value for t in ToleranceType)
        raise ValueError(f"Unknown tolerance type: '{tolerance_type}'. Valid: {valid}")

    if value <= 0:
        raise ValueError(f"Tolerance value must be > 0, got: {value}")

    if material_condition not in VALID_MATERIAL_CONDITIONS:
        raise ValueError(
            f"Invalid material condition: '{material_condition}'. "
            f"Valid: {', '.join(repr(m) for m in VALID_MATERIAL_CONDITIONS if m)}, or empty"
        )

    return ToleranceZone(
        tolerance_type=tolerance_type,
        value=value,
        entity_id=entity_id,
        datum_refs=datum_refs or [],
        material_condition=material_condition,
        description=description,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_gdt(
    annotations: GDTAnnotationSet,
    graph: SemanticGraph,
) -> list[dict[str, str]]:
    """Validate GD&T annotations against the semantic graph.

    Checks:
    - All datum entity_ids exist in the graph
    - All tolerance entity_ids exist in the graph
    - All datum_refs in tolerances reference defined datums
    - Orientation/location/runout tolerances have required datums
    - No duplicate datum labels

    Args:
        annotations: The annotation set to validate.
        graph: Semantic graph of the body.

    Returns:
        List of issue dicts with "severity" and "message" keys.
        Empty list means valid.
    """
    issues: list[dict[str, str]] = []

    # Collect all persistent_ids from the graph
    all_ids: set[str] = set()
    for face in graph.faces:
        all_ids.add(face.persistent_id)
    for edge in graph.edges:
        all_ids.add(edge.persistent_id)
    for feature in graph.features:
        all_ids.add(feature.persistent_id)

    # Check for duplicate datum labels
    labels_seen: set[str] = set()
    for datum in annotations.datums:
        if datum.label in labels_seen:
            issues.append({
                "severity": "error",
                "message": f"Duplicate datum label: '{datum.label}'",
            })
        labels_seen.add(datum.label)

    # Check datum entity_ids exist
    for datum in annotations.datums:
        if datum.entity_id not in all_ids:
            issues.append({
                "severity": "warning",
                "message": f"Datum '{datum.label}' references entity '{datum.entity_id}' not found in graph",
            })

    # Check tolerance entity_ids and datum references
    defined_labels = {d.label for d in annotations.datums}
    for tol in annotations.tolerances:
        if tol.entity_id not in all_ids:
            issues.append({
                "severity": "warning",
                "message": (
                    f"Tolerance ({tol.tolerance_type}) references entity "
                    f"'{tol.entity_id}' not found in graph"
                ),
            })

        # Check datum refs are defined
        for ref in tol.datum_refs:
            if ref not in defined_labels:
                issues.append({
                    "severity": "error",
                    "message": (
                        f"Tolerance ({tol.tolerance_type}) references datum '{ref}' "
                        f"which is not defined"
                    ),
                })

        # Check that orientation/location/runout have datum refs
        try:
            tt = ToleranceType(tol.tolerance_type)
            if tt in _REQUIRES_DATUM and not tol.datum_refs:
                issues.append({
                    "severity": "error",
                    "message": (
                        f"Tolerance type '{tol.tolerance_type}' requires at least "
                        f"one datum reference"
                    ),
                })
        except ValueError:
            issues.append({
                "severity": "error",
                "message": f"Unknown tolerance type: '{tol.tolerance_type}'",
            })

    return issues


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def gdt_to_dict(annotations: GDTAnnotationSet) -> dict[str, Any]:
    """Convert annotations to a JSON-serializable dict.

    Args:
        annotations: The annotation set.

    Returns:
        Dict suitable for JSON serialization.
    """
    return annotations.to_dict()


# ---------------------------------------------------------------------------
# Auto-suggestion
# ---------------------------------------------------------------------------

def suggest_gdt(graph: SemanticGraph) -> GDTAnnotationSet:
    """Auto-suggest GD&T annotations based on feature analysis.

    Heuristics:
    - The largest planar face becomes datum A (primary datum)
    - Through holes get position tolerance (0.1mm) referencing datum A
    - Blind holes get position tolerance (0.1mm) referencing datum A
    - Planar faces get flatness tolerance (0.05mm)
    - Cylindrical features get circularity tolerance (0.025mm)

    Args:
        graph: Semantic graph of the body.

    Returns:
        Suggested GDTAnnotationSet.
    """
    datums: list[DatumReference] = []
    tolerances: list[ToleranceZone] = []

    # Find the largest planar face for datum A
    planar_faces = [
        f for f in graph.faces
        if f.surface_type.value == "plane"
    ]

    if planar_faces:
        # Sort by area descending
        planar_faces_sorted = sorted(planar_faces, key=lambda f: f.area, reverse=True)
        primary = planar_faces_sorted[0]
        datums.append(DatumReference(
            label="A",
            entity_id=primary.persistent_id,
            description="Primary datum (largest planar face)",
        ))

        # Second largest perpendicular plane for datum B (if available)
        if len(planar_faces_sorted) > 1:
            for candidate in planar_faces_sorted[1:]:
                # Check if perpendicular to datum A (normals should be ~orthogonal)
                if primary.normal and candidate.normal:
                    dot = abs(
                        primary.normal.x * candidate.normal.x
                        + primary.normal.y * candidate.normal.y
                        + primary.normal.z * candidate.normal.z
                    )
                    if dot < 0.1:  # approximately perpendicular
                        datums.append(DatumReference(
                            label="B",
                            entity_id=candidate.persistent_id,
                            description="Secondary datum (perpendicular planar face)",
                        ))
                        break

        # Suggest flatness for major planar faces
        for face in planar_faces_sorted[:3]:  # top 3 by area
            tolerances.append(ToleranceZone(
                tolerance_type=ToleranceType.FLATNESS.value,
                value=0.05,
                entity_id=face.persistent_id,
                description=f"Flatness for planar face (area={face.area:.1f}mm2)",
            ))

    # Analyze features
    datum_labels = [d.label for d in datums]

    for feature in graph.features:
        ft = feature.feature_type.value

        if ft in ("through_hole", "blind_hole", "counterbore", "countersink"):
            # Position tolerance for holes
            tolerances.append(ToleranceZone(
                tolerance_type=ToleranceType.POSITION.value,
                value=0.1,
                entity_id=feature.persistent_id,
                datum_refs=datum_labels[:1],  # reference primary datum
                material_condition="MMC",
                description=f"Position for {ft}",
            ))

        elif ft == "boss":
            # Position tolerance for bosses
            tolerances.append(ToleranceZone(
                tolerance_type=ToleranceType.POSITION.value,
                value=0.1,
                entity_id=feature.persistent_id,
                datum_refs=datum_labels[:1],
                description=f"Position for boss",
            ))

    # Cylindrical faces get circularity
    cylinder_faces = [
        f for f in graph.faces
        if f.surface_type.value == "cylinder"
    ]
    for face in cylinder_faces[:5]:  # limit to avoid noise
        tolerances.append(ToleranceZone(
            tolerance_type=ToleranceType.CIRCULARITY.value,
            value=0.025,
            entity_id=face.persistent_id,
            description=f"Circularity for cylindrical face (r={face.radius}mm)",
        ))

    return GDTAnnotationSet(datums=datums, tolerances=tolerances)
