"""Feature recognition engine: holes, fillets, chamfers, slots, bosses, counterbores."""

from next3d.features.engine import register
from next3d.features.holes import HoleRecognizer
from next3d.features.fillets import FilletRecognizer
from next3d.features.chamfers import ChamferRecognizer
from next3d.features.slots import SlotRecognizer
from next3d.features.bosses import BossRecognizer
from next3d.features.counterbores import CounterboreRecognizer, CountersinkRecognizer

# Register all built-in recognizers
register(HoleRecognizer())
register(FilletRecognizer())
register(ChamferRecognizer())
register(SlotRecognizer())
register(BossRecognizer())
register(CounterboreRecognizer())
register(CountersinkRecognizer())
