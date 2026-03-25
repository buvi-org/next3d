"""Standard parts library — parametric ISO metric fasteners.

No designer models bolts from scratch. This library provides ready-to-use
parametric fastener geometry that can be inserted into assemblies.
"""

from next3d.parts.fasteners import (
    ISO_METRIC,
    iso_hex_bolt,
    iso_hex_nut,
    iso_flat_washer,
    iso_socket_head_cap_screw,
    list_available_sizes,
)

__all__ = [
    "ISO_METRIC",
    "iso_hex_bolt",
    "iso_hex_nut",
    "iso_flat_washer",
    "iso_socket_head_cap_screw",
    "list_available_sizes",
]
