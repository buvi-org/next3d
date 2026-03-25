"""Dimension tracking — annotate and query measurements on geometry.

Tracks linear, radial, angular, and diametral dimensions on entities.
Dimensions reference persistent IDs and can be exported to drawings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopoDS import TopoDS_Shape

from next3d.core.schema import SemanticGraph, Vec3


@dataclass(frozen=True)
class Dimension:
    """A dimension annotation on the geometry."""

    dim_id: str
    dim_type: str  # linear, radial, diametral, angular
    value: float  # measured value in mm or degrees
    entity_ids: list[str]  # persistent IDs of referenced entities
    label: str = ""  # display label (e.g. "∅10", "25.0")
    tolerance_plus: float = 0.0
    tolerance_minus: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "dim_id": self.dim_id,
            "type": self.dim_type,
            "value": round(self.value, 4),
            "label": self.label or self._auto_label(),
            "entity_ids": self.entity_ids,
        }
        if self.tolerance_plus or self.tolerance_minus:
            d["tolerance"] = {
                "plus": self.tolerance_plus,
                "minus": self.tolerance_minus,
            }
        return d

    def _auto_label(self) -> str:
        if self.dim_type == "diametral":
            return f"⌀{self.value:.2f}"
        elif self.dim_type == "radial":
            return f"R{self.value:.2f}"
        elif self.dim_type == "angular":
            return f"{self.value:.1f}°"
        return f"{self.value:.2f}"


class DimensionSet:
    """Collection of dimensions on a part."""

    def __init__(self) -> None:
        self._dimensions: list[Dimension] = []
        self._counter = 0

    @property
    def dimensions(self) -> list[Dimension]:
        return list(self._dimensions)

    def add(
        self,
        dim_type: str,
        value: float,
        entity_ids: list[str],
        label: str = "",
        tolerance_plus: float = 0.0,
        tolerance_minus: float = 0.0,
    ) -> Dimension:
        """Add a dimension annotation."""
        self._counter += 1
        dim = Dimension(
            dim_id=f"dim_{self._counter}",
            dim_type=dim_type,
            value=value,
            entity_ids=entity_ids,
            label=label,
            tolerance_plus=tolerance_plus,
            tolerance_minus=tolerance_minus,
        )
        self._dimensions.append(dim)
        return dim

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": len(self._dimensions),
            "dimensions": [d.to_dict() for d in self._dimensions],
        }

    def clear(self) -> None:
        self._dimensions.clear()


def auto_dimension(graph: SemanticGraph) -> list[Dimension]:
    """Auto-generate key dimensions from feature analysis.

    Extracts:
    - Hole diameters and depths
    - Fillet/chamfer radii
    - Slot widths and depths
    - Boss diameters and heights
    - Overall bounding box dimensions
    """
    dims: list[Dimension] = []
    counter = 0

    for feat in graph.features:
        counter += 1
        ft = feat.feature_type.value
        p = feat.parameters

        if ft in ("through_hole", "blind_hole"):
            if "diameter" in p:
                dims.append(Dimension(
                    dim_id=f"auto_{counter}",
                    dim_type="diametral",
                    value=p["diameter"],
                    entity_ids=[feat.persistent_id],
                    label=f"⌀{p['diameter']:.1f}",
                ))
            if "depth" in p and p["depth"]:
                counter += 1
                dims.append(Dimension(
                    dim_id=f"auto_{counter}",
                    dim_type="linear",
                    value=p["depth"],
                    entity_ids=[feat.persistent_id],
                    label=f"Depth {p['depth']:.1f}",
                ))

        elif ft == "fillet":
            if "radius" in p:
                dims.append(Dimension(
                    dim_id=f"auto_{counter}",
                    dim_type="radial",
                    value=p["radius"],
                    entity_ids=[feat.persistent_id],
                    label=f"R{p['radius']:.1f}",
                ))

        elif ft == "chamfer":
            if "distance" in p:
                dims.append(Dimension(
                    dim_id=f"auto_{counter}",
                    dim_type="linear",
                    value=p["distance"],
                    entity_ids=[feat.persistent_id],
                ))

        elif ft == "slot":
            for key in ("width", "depth", "length"):
                if key in p:
                    counter += 1
                    dims.append(Dimension(
                        dim_id=f"auto_{counter}",
                        dim_type="linear",
                        value=p[key],
                        entity_ids=[feat.persistent_id],
                        label=f"Slot {key} {p[key]:.1f}",
                    ))

        elif ft == "boss":
            if "diameter" in p:
                dims.append(Dimension(
                    dim_id=f"auto_{counter}",
                    dim_type="diametral",
                    value=p["diameter"],
                    entity_ids=[feat.persistent_id],
                ))
            if "height" in p:
                counter += 1
                dims.append(Dimension(
                    dim_id=f"auto_{counter}",
                    dim_type="linear",
                    value=p["height"],
                    entity_ids=[feat.persistent_id],
                ))

    # Overall bounding box
    if graph.solids:
        s = graph.solids[0]
        if hasattr(s, 'bounding_box') and s.bounding_box:
            bb = s.bounding_box
            counter += 1
            dims.append(Dimension(
                dim_id=f"auto_{counter}", dim_type="linear",
                value=bb.get("dx", 0), entity_ids=[], label="Overall X",
            ))
            counter += 1
            dims.append(Dimension(
                dim_id=f"auto_{counter}", dim_type="linear",
                value=bb.get("dy", 0), entity_ids=[], label="Overall Y",
            ))
            counter += 1
            dims.append(Dimension(
                dim_id=f"auto_{counter}", dim_type="linear",
                value=bb.get("dz", 0), entity_ids=[], label="Overall Z",
            ))

    return dims
