"""ISO metric fastener library — parametric bolt, nut, washer generation.

Generates simplified but dimensionally accurate fastener geometry.
Thread geometry is represented as a smooth cylinder (not helical threads)
which is standard practice for assembly/clearance work.

All dimensions per ISO 4762 (socket head), ISO 4032 (hex nut),
ISO 7089 (flat washer), ISO 4014 (hex bolt).
"""

from __future__ import annotations

import math

import cadquery as cq
from OCP.TopoDS import TopoDS_Shape


# ISO metric thread data: major diameter, pitch, and fastener dimensions
ISO_METRIC: dict[str, dict[str, float]] = {
    "M3": {
        "d": 3.0, "pitch": 0.5,
        "hex_head_d": 5.5, "hex_head_h": 2.0,
        "socket_head_d": 5.5, "socket_head_h": 3.0,
        "nut_d": 5.5, "nut_h": 2.4,
        "washer_od": 7.0, "washer_id": 3.2, "washer_h": 0.5,
    },
    "M4": {
        "d": 4.0, "pitch": 0.7,
        "hex_head_d": 7.0, "hex_head_h": 2.8,
        "socket_head_d": 7.0, "socket_head_h": 4.0,
        "nut_d": 7.0, "nut_h": 3.2,
        "washer_od": 9.0, "washer_id": 4.3, "washer_h": 0.8,
    },
    "M5": {
        "d": 5.0, "pitch": 0.8,
        "hex_head_d": 8.0, "hex_head_h": 3.5,
        "socket_head_d": 8.5, "socket_head_h": 5.0,
        "nut_d": 8.0, "nut_h": 4.0,
        "washer_od": 10.0, "washer_id": 5.3, "washer_h": 1.0,
    },
    "M6": {
        "d": 6.0, "pitch": 1.0,
        "hex_head_d": 10.0, "hex_head_h": 4.0,
        "socket_head_d": 10.0, "socket_head_h": 6.0,
        "nut_d": 10.0, "nut_h": 5.0,
        "washer_od": 12.0, "washer_id": 6.4, "washer_h": 1.6,
    },
    "M8": {
        "d": 8.0, "pitch": 1.25,
        "hex_head_d": 13.0, "hex_head_h": 5.3,
        "socket_head_d": 13.0, "socket_head_h": 8.0,
        "nut_d": 13.0, "nut_h": 6.5,
        "washer_od": 16.0, "washer_id": 8.4, "washer_h": 1.6,
    },
    "M10": {
        "d": 10.0, "pitch": 1.5,
        "hex_head_d": 16.0, "hex_head_h": 6.4,
        "socket_head_d": 16.0, "socket_head_h": 10.0,
        "nut_d": 16.0, "nut_h": 8.0,
        "washer_od": 20.0, "washer_id": 10.5, "washer_h": 2.0,
    },
    "M12": {
        "d": 12.0, "pitch": 1.75,
        "hex_head_d": 18.0, "hex_head_h": 7.5,
        "socket_head_d": 18.0, "socket_head_h": 12.0,
        "nut_d": 18.0, "nut_h": 10.0,
        "washer_od": 24.0, "washer_id": 13.0, "washer_h": 2.5,
    },
}


def list_available_sizes() -> list[str]:
    """Return available ISO metric sizes."""
    return list(ISO_METRIC.keys())


def _get_spec(size: str) -> dict[str, float]:
    """Look up size spec, raise if not found."""
    size = size.upper()
    if size not in ISO_METRIC:
        raise ValueError(f"Unknown size: {size}. Available: {', '.join(ISO_METRIC)}")
    return ISO_METRIC[size]


def _hexagon(circumscribed_d: float) -> cq.Workplane:
    """Create a regular hexagon sketch from circumscribed circle diameter."""
    # CadQuery polygon: inscribed radius = circumscribed_d / 2
    return cq.Workplane("XY").polygon(6, circumscribed_d)


def iso_hex_bolt(size: str, length: float) -> TopoDS_Shape:
    """Generate an ISO hex bolt (simplified — no threads).

    Args:
        size: ISO metric size, e.g. "M6", "M8".
        length: Shank length in mm (not including head).

    Returns:
        TopoDS_Shape with hex head + cylindrical shank.
    """
    spec = _get_spec(size)
    d = spec["d"]
    head_d = spec["hex_head_d"]
    head_h = spec["hex_head_h"]

    # Build hex head + shank
    # Head: hexagonal prism
    head = _hexagon(head_d).extrude(head_h)
    # Shank: cylinder extending downward from head base
    shank = (
        cq.Workplane("XY")
        .workplane(offset=-length)
        .circle(d / 2)
        .extrude(length)
    )
    # Union head + shank
    result = head.union(shank)
    return result.val().wrapped


def iso_hex_nut(size: str) -> TopoDS_Shape:
    """Generate an ISO hex nut (simplified).

    Args:
        size: ISO metric size, e.g. "M6".

    Returns:
        TopoDS_Shape: hex nut with center bore.
    """
    spec = _get_spec(size)
    d = spec["d"]
    nut_d = spec["nut_d"]
    nut_h = spec["nut_h"]

    result = (
        _hexagon(nut_d)
        .extrude(nut_h)
        .faces(">Z")
        .workplane()
        .hole(d)
    )
    return result.val().wrapped


def iso_flat_washer(size: str) -> TopoDS_Shape:
    """Generate an ISO flat washer.

    Args:
        size: ISO metric size, e.g. "M6".

    Returns:
        TopoDS_Shape: flat washer (annular disc).
    """
    spec = _get_spec(size)
    od = spec["washer_od"]
    id_ = spec["washer_id"]
    h = spec["washer_h"]

    result = (
        cq.Workplane("XY")
        .circle(od / 2)
        .circle(id_ / 2)
        .extrude(h)
    )
    return result.val().wrapped


def iso_socket_head_cap_screw(size: str, length: float) -> TopoDS_Shape:
    """Generate an ISO socket head cap screw (simplified).

    Args:
        size: ISO metric size, e.g. "M6".
        length: Shank length in mm.

    Returns:
        TopoDS_Shape: cylindrical head + shank.
    """
    spec = _get_spec(size)
    d = spec["d"]
    head_d = spec["socket_head_d"]
    head_h = spec["socket_head_h"]

    # Cylindrical head
    head = cq.Workplane("XY").circle(head_d / 2).extrude(head_h)
    # Shank
    shank = (
        cq.Workplane("XY")
        .workplane(offset=-length)
        .circle(d / 2)
        .extrude(length)
    )
    result = head.union(shank)
    return result.val().wrapped
