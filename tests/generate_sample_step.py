"""Generate a sample STEP file for testing: a block with holes and fillets.

Run this script once to create tests/fixtures/sample_block.step
"""

from pathlib import Path
import cadquery as cq

# A 100x60x20 block with:
# - 2 through holes (diameter 10mm)
# - Fillets on top edges (radius 2mm)
result = (
    cq.Workplane("XY")
    .box(100, 60, 20)
    .faces(">Z")
    .workplane()
    .pushPoints([(20, 0), (-20, 0)])
    .hole(10)
    .edges("|Z")
    .fillet(2)
)

fixtures_dir = Path(__file__).parent / "fixtures"
fixtures_dir.mkdir(exist_ok=True)
output = fixtures_dir / "sample_block.step"
cq.exporters.export(result, str(output))
print(f"Written: {output}")
