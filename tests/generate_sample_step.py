"""Generate sample STEP files for testing.

Run: python tests/generate_sample_step.py

Creates:
- sample_block.step      — Block with 2 through holes + fillets
- sample_chamfer.step    — Block with chamfered edges
- sample_boss.step       — Plate with cylindrical boss protrusions
- sample_slot.step       — Block with a slot cut
- sample_complex.step    — Combined: holes, fillets, chamfers, boss
- sample_counterbore.step — Plate with counterbored holes
- sample_multibody.step  — Two separate solids in one file
- sample_pattern.step    — Block with a linear pattern of holes
"""

from pathlib import Path
import cadquery as cq

fixtures_dir = Path(__file__).parent / "fixtures"
fixtures_dir.mkdir(exist_ok=True)


def make_block_with_holes():
    """100x60x20 block, 2 through holes (d=10), fillets on vertical edges (r=2)."""
    return (
        cq.Workplane("XY")
        .box(100, 60, 20)
        .faces(">Z")
        .workplane()
        .pushPoints([(20, 0), (-20, 0)])
        .hole(10)
        .edges("|Z")
        .fillet(2)
    )


def make_chamfered_block():
    """80x50x25 block with chamfered top edges (2mm)."""
    return (
        cq.Workplane("XY")
        .box(80, 50, 25)
        .edges(">Z")
        .chamfer(2)
    )


def make_boss_plate():
    """120x80x10 plate with two cylindrical bosses (d=15, h=20) on top."""
    return (
        cq.Workplane("XY")
        .box(120, 80, 10)
        .faces(">Z")
        .workplane()
        .pushPoints([(30, 0), (-30, 0)])
        .circle(7.5)
        .extrude(20)
    )


def make_slot_block():
    """100x60x30 block with a slot cut (width=12, depth=15) along X axis."""
    return (
        cq.Workplane("XY")
        .box(100, 60, 30)
        .faces(">Z")
        .workplane()
        .slot2D(60, 12, 0)
        .cutBlind(-15)
    )


def make_complex_part():
    """A part combining: holes, fillets, chamfers, and boss."""
    result = (
        cq.Workplane("XY")
        .box(100, 80, 25)
        # Through holes
        .faces(">Z")
        .workplane()
        .pushPoints([(30, 20), (-30, 20), (30, -20), (-30, -20)])
        .hole(8)
        # Boss on top
        .faces(">Z")
        .workplane()
        .center(0, 0)
        .circle(10)
        .extrude(15)
        # Fillets on boss top edges
        .edges(">Z")
        .fillet(1)
        # Chamfer on base bottom edges
        .edges("<Z")
        .chamfer(1.5)
    )
    return result


def make_counterbore():
    """Plate with a counterbored hole: d=6 through hole + d=12 x 5mm deep counterbore."""
    return (
        cq.Workplane("XY")
        .box(60, 60, 20)
        .faces(">Z")
        .workplane()
        .cboreHole(6, 12, 5)
    )


def make_multibody():
    """Two separate blocks as a compound (multi-body STEP)."""
    block1 = cq.Workplane("XY").box(40, 30, 20)
    block2 = cq.Workplane("XY").center(80, 0).box(40, 30, 20)
    return block1.add(block2)


def make_pattern():
    """Block with a linear pattern of 4 evenly spaced through holes along X."""
    return (
        cq.Workplane("XY")
        .box(120, 40, 15)
        .faces(">Z")
        .workplane()
        .pushPoints([(-45, 0), (-15, 0), (15, 0), (45, 0)])
        .hole(8)
    )


parts = {
    "sample_block": make_block_with_holes,
    "sample_chamfer": make_chamfered_block,
    "sample_boss": make_boss_plate,
    "sample_slot": make_slot_block,
    "sample_complex": make_complex_part,
    "sample_counterbore": make_counterbore,
    "sample_multibody": make_multibody,
    "sample_pattern": make_pattern,
}

for name, factory in parts.items():
    result = factory()
    output = fixtures_dir / f"{name}.step"
    cq.exporters.export(result, str(output))
    print(f"Written: {output}")
