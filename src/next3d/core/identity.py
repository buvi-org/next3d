"""Persistent identity system for topological entities.

Assigns deterministic, stable IDs based on geometric and topological hashing.
Same geometry → same ID, regardless of STEP file internal ordering.
"""

from __future__ import annotations

import hashlib
import struct


def _float_bytes(value: float, precision: int = 6) -> bytes:
    """Round a float to `precision` decimals and pack as bytes."""
    return struct.pack("!d", round(value, precision))


def vertex_id(x: float, y: float, z: float) -> str:
    """Generate a persistent ID for a vertex from its coordinates."""
    h = hashlib.blake2s(digest_size=8)
    h.update(b"vertex")
    h.update(_float_bytes(x))
    h.update(_float_bytes(y))
    h.update(_float_bytes(z))
    return f"vertex_{h.hexdigest()}"


def edge_id(
    start_x: float,
    start_y: float,
    start_z: float,
    end_x: float,
    end_y: float,
    end_z: float,
    curve_type: str,
) -> str:
    """Generate a persistent ID for an edge from endpoints and curve type.

    Endpoints are sorted to ensure the same edge gets the same ID
    regardless of traversal direction.
    """
    h = hashlib.blake2s(digest_size=8)
    h.update(b"edge")
    h.update(curve_type.encode())

    # Sort endpoints so direction doesn't affect the hash
    p1 = (round(start_x, 6), round(start_y, 6), round(start_z, 6))
    p2 = (round(end_x, 6), round(end_y, 6), round(end_z, 6))
    for coord in sorted([p1, p2]):
        for c in coord:
            h.update(_float_bytes(c))

    return f"edge_{h.hexdigest()}"


def face_id(
    surface_type: str,
    centroid_x: float,
    centroid_y: float,
    centroid_z: float,
    area: float,
) -> str:
    """Generate a persistent ID for a face from surface type, centroid, and area."""
    h = hashlib.blake2s(digest_size=8)
    h.update(b"face")
    h.update(surface_type.encode())
    h.update(_float_bytes(centroid_x))
    h.update(_float_bytes(centroid_y))
    h.update(_float_bytes(centroid_z))
    h.update(_float_bytes(area))
    return f"face_{h.hexdigest()}"


def solid_id(centroid_x: float, centroid_y: float, centroid_z: float, volume: float) -> str:
    """Generate a persistent ID for a solid from its centroid and volume."""
    h = hashlib.blake2s(digest_size=8)
    h.update(b"solid")
    h.update(_float_bytes(centroid_x))
    h.update(_float_bytes(centroid_y))
    h.update(_float_bytes(centroid_z))
    h.update(_float_bytes(volume))
    return f"solid_{h.hexdigest()}"


def feature_id(feature_type: str, constituent_face_ids: list[str]) -> str:
    """Generate a persistent ID for a feature from its type and constituent faces."""
    h = hashlib.blake2s(digest_size=8)
    h.update(b"feature")
    h.update(feature_type.encode())
    for fid in sorted(constituent_face_ids):
        h.update(fid.encode())
    return f"feature_{h.hexdigest()}"
