"""2D engineering drawing generation.

Generates multi-view engineering drawings from 3D geometry:
- Orthographic projections (front, top, right, isometric)
- Cross-section views
- Dimension annotations
- Export to SVG and DXF

Uses OpenCascade HLR (Hidden Line Removal) for accurate 2D projection
and CadQuery's export infrastructure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cadquery as cq
from OCP.TopoDS import TopoDS_Shape
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Pln, gp_Vec
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section


# ---------------------------------------------------------------------------
# View definitions
# ---------------------------------------------------------------------------

# Standard projection directions (eye looking at origin)
STANDARD_VIEWS = {
    "front":      {"dir": (0, -1, 0),   "up": (0, 0, 1),  "label": "Front"},
    "back":       {"dir": (0, 1, 0),    "up": (0, 0, 1),  "label": "Back"},
    "top":        {"dir": (0, 0, 1),    "up": (0, 1, 0),  "label": "Top"},
    "bottom":     {"dir": (0, 0, -1),   "up": (0, -1, 0), "label": "Bottom"},
    "right":      {"dir": (1, 0, 0),    "up": (0, 0, 1),  "label": "Right"},
    "left":       {"dir": (-1, 0, 0),   "up": (0, 0, 1),  "label": "Left"},
    "isometric":  {"dir": (1, 1, 1),    "up": (0, 0, 1),  "label": "Isometric"},
    "dimetric":   {"dir": (1, 1, 0.5),  "up": (0, 0, 1),  "label": "Dimetric"},
}


@dataclass
class ViewConfig:
    """Configuration for a single drawing view."""

    name: str
    projection_dir: tuple[float, float, float]
    up_dir: tuple[float, float, float] = (0, 0, 1)
    show_hidden: bool = True
    scale: float = 1.0
    label: str = ""

    # Section plane (optional)
    section_plane: str | None = None  # "XY", "XZ", "YZ"
    section_offset: float = 0.0  # offset along plane normal

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "projection_dir": list(self.projection_dir),
            "show_hidden": self.show_hidden,
            "scale": self.scale,
            "label": self.label or self.name,
        }
        if self.section_plane:
            d["section"] = {
                "plane": self.section_plane,
                "offset": self.section_offset,
            }
        return d


@dataclass
class DrawingView:
    """A rendered 2D view with SVG content."""

    config: ViewConfig
    svg_content: str
    width: float  # bounding width of the projected view
    height: float


@dataclass
class Drawing:
    """A complete engineering drawing with multiple views."""

    views: list[DrawingView]
    title: str = ""
    scale: str = "1:1"
    material: str = ""
    part_number: str = ""

    def to_svg(self, page_width: int = 1200, page_height: int = 800) -> str:
        """Combine all views into a single SVG drawing sheet."""
        # Layout views in a grid
        n = len(self.views)
        if n == 0:
            return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'

        cols = min(n, 3)
        rows = math.ceil(n / cols)

        margin = 40
        view_w = (page_width - margin * (cols + 1)) / cols
        view_h = (page_height - margin * (rows + 1) - 60) / rows  # 60 for title block

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{page_width}" height="{page_height}" '
            f'viewBox="0 0 {page_width} {page_height}">'
        ]

        # Background
        parts.append(
            f'<rect width="{page_width}" height="{page_height}" '
            f'fill="white" stroke="black" stroke-width="2"/>'
        )

        # Title block
        tb_y = page_height - 50
        parts.append(
            f'<rect x="0" y="{tb_y}" width="{page_width}" height="50" '
            f'fill="#f0f0f0" stroke="black"/>'
        )
        parts.append(
            f'<text x="{page_width/2}" y="{tb_y+20}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14" font-weight="bold">'
            f'{self.title or "Engineering Drawing"}</text>'
        )
        info_parts = []
        if self.scale:
            info_parts.append(f"Scale: {self.scale}")
        if self.material:
            info_parts.append(f"Material: {self.material}")
        if self.part_number:
            info_parts.append(f"P/N: {self.part_number}")
        if info_parts:
            parts.append(
                f'<text x="{page_width/2}" y="{tb_y+38}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="10">'
                f'{" | ".join(info_parts)}</text>'
            )

        # Render each view
        for i, view in enumerate(self.views):
            col = i % cols
            row = i // cols
            x = margin + col * (view_w + margin)
            y = margin + row * (view_h + margin)

            # View border
            parts.append(
                f'<rect x="{x}" y="{y}" width="{view_w}" height="{view_h}" '
                f'fill="none" stroke="#ccc" stroke-width="0.5" stroke-dasharray="4,2"/>'
            )

            # View label
            parts.append(
                f'<text x="{x + view_w/2}" y="{y + view_h - 5}" '
                f'text-anchor="middle" font-family="sans-serif" font-size="10" '
                f'fill="#666">{view.config.label or view.config.name}</text>'
            )

            # Embed the view SVG (scaled and translated to fit the view rect)
            inner, vb_x, vb_y, vb_w, vb_h = _extract_svg_content(view.svg_content)
            if inner:
                # Available space inside the view rect (with padding)
                pad = 15
                avail_w = view_w - 2 * pad
                avail_h = view_h - 2 * pad - 15  # extra space for label

                # Scale to fit the SVG viewBox into the available space
                sx = avail_w / max(vb_w, 0.1)
                sy = avail_h / max(vb_h, 0.1)
                s = min(sx, sy)

                # Center of the view rect (accounting for label)
                cx = x + pad + avail_w / 2
                cy = y + pad + avail_h / 2

                # Center of the SVG content in its own coordinate space
                svg_cx = vb_x + vb_w / 2
                svg_cy = vb_y + vb_h / 2

                # Translate so SVG center maps to view rect center
                tx = cx - svg_cx * s
                ty = cy - svg_cy * s

                parts.append(
                    f'<g transform="translate({tx:.2f},{ty:.2f}) scale({s:.4f})">'
                )
                parts.append(inner)
                parts.append('</g>')

        parts.append('</svg>')
        return '\n'.join(parts)


def _extract_svg_content(svg: str) -> tuple[str, float, float, float, float]:
    """Extract inner content and viewBox from an SVG string.

    Returns:
        (inner_svg, vb_x, vb_y, vb_width, vb_height)
        If no viewBox is found, dimensions default to (0, 0, 100, 100).
    """
    import re

    # Parse viewBox or width/height from the <svg> tag
    vb_x, vb_y, vb_w, vb_h = 0.0, 0.0, 100.0, 100.0

    svg_tag = re.search(r'<svg[^>]*>', svg)
    if svg_tag:
        tag = svg_tag.group(0)
        vb_match = re.search(r'viewBox\s*=\s*"([^"]*)"', tag)
        if vb_match:
            parts = vb_match.group(1).split()
            if len(parts) == 4:
                vb_x, vb_y, vb_w, vb_h = (float(p) for p in parts)
        else:
            # Try width/height attributes
            w_match = re.search(r'\bwidth\s*=\s*"([0-9.]+)', tag)
            h_match = re.search(r'\bheight\s*=\s*"([0-9.]+)', tag)
            if w_match:
                vb_w = float(w_match.group(1))
            if h_match:
                vb_h = float(h_match.group(1))

    # Strip outer wrappers
    inner = re.sub(r'<\?xml[^>]*\?>', '', svg)
    inner = re.sub(r'<svg[^>]*>', '', inner, count=1)
    inner = re.sub(r'</svg>\s*$', '', inner)

    return inner.strip(), vb_x, vb_y, vb_w, vb_h


# ---------------------------------------------------------------------------
# View rendering
# ---------------------------------------------------------------------------

def render_view(
    shape: TopoDS_Shape,
    config: ViewConfig,
) -> DrawingView:
    """Render a single 2D view of a 3D shape.

    Uses CadQuery's SVG exporter with HLR (Hidden Line Removal)
    for accurate orthographic/perspective projection.
    """
    # If this is a section view, cut the shape first
    render_shape = shape
    if config.section_plane:
        render_shape = _create_section(shape, config.section_plane, config.section_offset)

    solid = cq.Solid(render_shape)

    opts = {
        "projectionDir": config.projection_dir,
        "showHidden": config.show_hidden,
        "strokeWidth": 0.5,
        "strokeColor": (0, 0, 0),
        "hiddenColor": (180, 180, 180),
        "showAxes": False,
    }

    # getSVG requires a Shape, not a Workplane
    svg = cq.exporters.getSVG(solid, opts)

    # Get actual 2D dimensions from the SVG viewBox
    _, vb_x, vb_y, vb_w, vb_h = _extract_svg_content(svg)

    return DrawingView(
        config=config,
        svg_content=svg,
        width=vb_w,
        height=vb_h,
    )


def _create_section(
    shape: TopoDS_Shape,
    plane: str,
    offset: float = 0.0,
) -> TopoDS_Shape:
    """Create a cross-section of the shape at a given plane.

    Returns the original shape (section view shows the cut face
    with the shape behind it — we keep the full shape for HLR
    to handle correctly).
    """
    # For section views, we cut the shape with a half-space
    # to show only one half, revealing the internal structure

    planes = {
        "XY": (gp_Pnt(0, 0, offset), gp_Dir(0, 0, 1)),
        "XZ": (gp_Pnt(0, offset, 0), gp_Dir(0, 1, 0)),
        "YZ": (gp_Pnt(offset, 0, 0), gp_Dir(1, 0, 0)),
    }

    if plane not in planes:
        return shape

    origin, normal = planes[plane]

    # Create a large cutting box on one side of the plane
    import cadquery as cq
    size = 10000  # large enough to encompass any part

    nx, ny, nz = normal.X(), normal.Y(), normal.Z()
    cut_center = (
        origin.X() + nx * size / 2,
        origin.Y() + ny * size / 2,
        origin.Z() + nz * size / 2,
    )

    cut_box = cq.Workplane("XY").transformed(
        offset=cut_center
    ).box(size, size, size).val().wrapped

    from next3d.modeling.kernel import boolean_cut
    try:
        return boolean_cut(shape, cut_box)
    except Exception:
        return shape  # if cut fails, return original


# ---------------------------------------------------------------------------
# Drawing generation
# ---------------------------------------------------------------------------

def generate_drawing(
    shape: TopoDS_Shape,
    views: list[str | ViewConfig] | None = None,
    title: str = "",
    scale: str = "1:1",
    material: str = "",
    part_number: str = "",
    show_hidden: bool = True,
) -> Drawing:
    """Generate a multi-view engineering drawing.

    Args:
        shape: The 3D shape to draw.
        views: List of view names ("front", "top", "right", "isometric")
               or ViewConfig objects. Default: front + top + right + isometric.
        title: Drawing title.
        scale: Scale label.
        material: Material label.
        part_number: Part number label.
        show_hidden: Show hidden lines in all views.

    Returns:
        Drawing object with rendered views.
    """
    if views is None:
        views = ["front", "top", "right", "isometric"]

    configs: list[ViewConfig] = []
    for v in views:
        if isinstance(v, ViewConfig):
            configs.append(v)
        elif isinstance(v, str):
            std = STANDARD_VIEWS.get(v)
            if std:
                configs.append(ViewConfig(
                    name=v,
                    projection_dir=std["dir"],
                    up_dir=std.get("up", (0, 0, 1)),
                    show_hidden=show_hidden,
                    label=std["label"],
                ))
            else:
                raise ValueError(
                    f"Unknown view: {v}. Available: {', '.join(STANDARD_VIEWS)}"
                )

    rendered_views = []
    for config in configs:
        view = render_view(shape, config)
        rendered_views.append(view)

    return Drawing(
        views=rendered_views,
        title=title,
        scale=scale,
        material=material,
        part_number=part_number,
    )


def generate_section_drawing(
    shape: TopoDS_Shape,
    section_plane: str = "XZ",
    section_offset: float = 0.0,
    title: str = "",
) -> Drawing:
    """Generate a drawing with a cross-section view.

    Args:
        shape: The 3D shape.
        section_plane: "XY", "XZ", or "YZ".
        section_offset: Offset along the plane normal.
        title: Drawing title.
    """
    section_config = ViewConfig(
        name="section",
        projection_dir=STANDARD_VIEWS["front"]["dir"],
        show_hidden=False,
        section_plane=section_plane,
        section_offset=section_offset,
        label=f"Section {section_plane} at {section_offset}mm",
    )

    # Also include the full view for reference
    iso_config = ViewConfig(
        name="isometric",
        projection_dir=STANDARD_VIEWS["isometric"]["dir"],
        show_hidden=True,
        label="Isometric (reference)",
    )

    views = [
        render_view(shape, section_config),
        render_view(shape, iso_config),
    ]

    return Drawing(views=views, title=title or "Section Drawing")


def export_drawing(
    drawing: Drawing,
    path: str,
    page_width: int = 1200,
    page_height: int = 800,
) -> None:
    """Export a drawing to SVG file.

    Args:
        drawing: The Drawing to export.
        path: Output file path (.svg).
        page_width: Page width in pixels.
        page_height: Page height in pixels.
    """
    svg = drawing.to_svg(page_width, page_height)
    with open(path, 'w') as f:
        f.write(svg)


def export_view_dxf(
    shape: TopoDS_Shape,
    path: str,
    projection_dir: tuple[float, float, float] = (0, 0, 1),
) -> None:
    """Export a single projected view as DXF.

    Args:
        shape: The 3D shape.
        path: Output DXF file path.
        projection_dir: Projection direction.
    """
    # Create a section/projection at the given direction
    solid = cq.Solid(shape)
    wp = cq.Workplane(obj=solid)

    try:
        cq.exporters.export(wp, path, exportType="DXF")
    except Exception as e:
        # Fallback: section at Z=0
        section = wp.section(0)
        cq.exporters.export(section, path, exportType="DXF")
