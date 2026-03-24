"""STEP file metadata extraction.

Parses STEP (ISO 10303-21) file headers and product data entities
directly from the text, extracting:
- File header (description, name, schema, authoring tool)
- Product definitions (name, id, description)
- Application protocol (AP203, AP214, AP242)
- Geometric entity statistics
- Construction hints from entity structure

This works on any STEP file regardless of AP version.
No parametric history exists in standard STEP B-Rep (AP203/AP214).
AP242 can carry PMI (dimensions, tolerances, GD&T) which we also extract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class STEPHeader:
    """Parsed STEP file header."""

    description: str = ""
    file_name: str = ""
    author: str = ""
    organization: str = ""
    preprocessor: str = ""
    originating_system: str = ""
    authorization: str = ""
    schema: str = ""
    ap_version: str = ""  # e.g. "AP214", "AP203", "AP242"


@dataclass
class ProductInfo:
    """A PRODUCT entity extracted from STEP data."""

    entity_id: str
    product_id: str
    name: str
    description: str


@dataclass
class STEPMetadata:
    """Complete metadata extracted from a STEP file."""

    header: STEPHeader
    products: list[ProductInfo] = field(default_factory=list)
    entity_counts: dict[str, int] = field(default_factory=dict)
    total_entities: int = 0
    has_pmi: bool = False  # Product Manufacturing Information (GD&T)
    has_colors: bool = False
    has_layers: bool = False
    has_assembly: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "header": {
                "description": self.header.description,
                "file_name": self.header.file_name,
                "author": self.header.author,
                "organization": self.header.organization,
                "preprocessor": self.header.preprocessor,
                "originating_system": self.header.originating_system,
                "schema": self.header.schema,
                "ap_version": self.header.ap_version,
            },
            "products": [
                {"id": p.product_id, "name": p.name, "description": p.description}
                for p in self.products
            ],
            "entity_statistics": {
                "total": self.total_entities,
                "top_types": dict(
                    sorted(self.entity_counts.items(), key=lambda x: -x[1])[:20]
                ),
            },
            "capabilities": {
                "has_pmi": self.has_pmi,
                "has_colors": self.has_colors,
                "has_layers": self.has_layers,
                "has_assembly": self.has_assembly,
            },
        }


def _parse_string_arg(text: str) -> str:
    """Extract a quoted string from STEP argument text."""
    m = re.search(r"'([^']*)'", text)
    return m.group(1) if m else text.strip()


def _parse_string_list(text: str) -> list[str]:
    """Extract all quoted strings from parenthesized STEP list."""
    return re.findall(r"'([^']*)'", text)


def extract_metadata(step_path: str | Path) -> STEPMetadata:
    """Extract metadata from a STEP file by parsing its text.

    This is fast (no OCC needed) and works on any STEP file.

    Args:
        step_path: Path to the STEP file.

    Returns:
        STEPMetadata with header, products, entity stats.
    """
    path = Path(step_path)
    text = path.read_text(errors="replace")

    header = _parse_header(text)
    products = _parse_products(text)
    entity_counts, total = _count_entities(text)

    # Detect capabilities from entity types
    has_pmi = any(
        k for k in entity_counts
        if "TOLERANCE" in k or "DIMENSION" in k or "DATUM" in k or "GEOMETRIC_TOLERANCE" in k
    )
    has_colors = any(k for k in entity_counts if "COLOUR" in k or "COLOR" in k or "STYLED_ITEM" in k)
    has_layers = any(k for k in entity_counts if "LAYER" in k)
    has_assembly = any(
        k for k in entity_counts
        if k in ("NEXT_ASSEMBLY_USAGE_OCCURRENCE", "PRODUCT_DEFINITION_USAGE")
    )

    return STEPMetadata(
        header=header,
        products=products,
        entity_counts=entity_counts,
        total_entities=total,
        has_pmi=has_pmi,
        has_colors=has_colors,
        has_layers=has_layers,
        has_assembly=has_assembly,
    )


def _parse_header(text: str) -> STEPHeader:
    """Parse the HEADER section of a STEP file."""
    header = STEPHeader()

    # Extract header section
    header_match = re.search(r"HEADER;(.*?)ENDSEC;", text, re.DOTALL)
    if not header_match:
        return header
    header_text = header_match.group(1)

    # FILE_DESCRIPTION
    m = re.search(r"FILE_DESCRIPTION\s*\(\s*\(([^)]*)\)", header_text)
    if m:
        header.description = _parse_string_arg(m.group(1))

    # FILE_NAME
    m = re.search(r"FILE_NAME\s*\(([^;]+);", header_text, re.DOTALL)
    if m:
        args = m.group(1)
        strings = _parse_string_list(args)
        if len(strings) >= 1:
            header.file_name = strings[0]
        if len(strings) >= 3:
            header.author = strings[2]
        if len(strings) >= 4:
            header.organization = strings[3]
        if len(strings) >= 5:
            header.preprocessor = strings[4]
        if len(strings) >= 6:
            header.originating_system = strings[5]
        if len(strings) >= 7:
            header.authorization = strings[6]

    # FILE_SCHEMA
    m = re.search(r"FILE_SCHEMA\s*\(\s*\(([^)]*)\)", header_text)
    if m:
        schema_str = _parse_string_arg(m.group(1))
        header.schema = schema_str
        # Detect AP version
        if "214" in schema_str:
            header.ap_version = "AP214"
        elif "203" in schema_str:
            header.ap_version = "AP203"
        elif "242" in schema_str:
            header.ap_version = "AP242"
        elif "AUTOMOTIVE" in schema_str.upper():
            header.ap_version = "AP214"
        elif "CONFIG_CONTROL" in schema_str.upper():
            header.ap_version = "AP203"

    return header


def _parse_products(text: str) -> list[ProductInfo]:
    """Extract PRODUCT entities from STEP data section."""
    products = []
    # PRODUCT('id','name','description',(#context));
    pattern = re.compile(
        r"(#\d+)\s*=\s*PRODUCT\s*\(\s*'([^']*)'\s*,\s*'([^']*)'\s*,\s*'([^']*)'",
    )
    for m in pattern.finditer(text):
        products.append(
            ProductInfo(
                entity_id=m.group(1),
                product_id=m.group(2),
                name=m.group(3),
                description=m.group(4),
            )
        )
    return products


def _count_entities(text: str) -> tuple[dict[str, int], int]:
    """Count STEP entity types in the DATA section."""
    counts: dict[str, int] = {}
    total = 0
    # Match: #123 = ENTITY_NAME(...
    pattern = re.compile(r"#\d+\s*=\s*([A-Z_][A-Z0-9_]*)\s*\(")
    for m in pattern.finditer(text):
        entity_type = m.group(1)
        counts[entity_type] = counts.get(entity_type, 0) + 1
        total += 1
    return counts, total
