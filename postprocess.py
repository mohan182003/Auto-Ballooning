# postprocess.py
import re
from typing import Any, Dict, List, Tuple

from schemas import DrawingResponse, Measurement


def _extract_numeric(text: str) -> str | None:
    """
    Grab the first signed / decimal number we see:  12   3.45  -0.2  etc.
    Returns None if nothing looks numeric.
    """
    m = re.search(r"[+-]?\d+(?:\.\d+)?", text)
    return m.group(0) if m else None


def _classify_measurement(text: str) -> str:
    """
    Very light-weight feature class:
        diameter - contains Ø, ⌀ or φ
        angle    - contains °
        length   - everything else
    """
    if any(sym in text for sym in ("Ø", "⌀", "φ")):
        return "diameter"
    if "°" in text:
        return "angle"
    return "length"



# ── Helpers ────────────────────────────────────────────────────────────────
def _center_to_bbox(cx: int, cy: int, half: int = 6) -> List[Tuple[int, int]]:
    """Build a 4-corner square bbox centred on (cx, cy).

    The box is (2 × half) px on a side; default = 12 px (half = 6).
    Order: TL, TR, BR, BL.
    """
    return [
        (cx - half, cy - half),  # top-left
        (cx + half, cy - half),  # top-right
        (cx + half, cy + half),  # bottom-right
        (cx - half, cy + half),  # bottom-left
    ]


def _harvest_bbox(
    item: Tuple[str, Tuple[int, int] | List[Tuple[int, int]]]
) -> List[Tuple[int, int]] | None:
    """Return a 4-point bbox from an OCR item.

    The raw item is either:
      • ("196", (277, 49))   → just a centre
      • ("196", [(x1,y1), …]) → already a polygon
    """
    _, payload = item

    # Already a 4-corner polygon?
    if (
        isinstance(payload, (list, tuple))
        and len(payload) == 4
        and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in payload)
    ):
        return [tuple(map(int, p)) for p in payload]

    # Otherwise treat it as (cx, cy) and synthesize a square
    if isinstance(payload, (list, tuple)) and len(payload) == 2:
        cx, cy = map(int, payload)
        return _center_to_bbox(cx, cy)

    return None  # should not happen, but keeps type-checkers happy

# ────────────────────────────────────────────────────────────────────────────────
# NEW build_measurements
# ────────────────────────────────────────────────────────────────────────────────
def build_measurements(filename: str, raw: Dict[str, Any]) -> DrawingResponse:
    """
    Convert the tuples coming from ocr_it.py into our pydantic schema.
    `raw["dim"]` and `raw["other"]` each look like:
        [["196", [277,  49]],
         ["118", [345,  78]],
         ...
        ]
    """

    dims: List[Tuple[str, Tuple[int, int]]] = raw.get("dim", [])
    misc: List[Tuple[str, Tuple[int, int]]] = raw.get("other", [])

    # 1️⃣   keep *everything* and sort so balloon numbers are deterministic
    items: List[Tuple[str, Tuple[int, int]]] = dims + misc
    items.sort(key=lambda it: (it[1][1], it[1][0]))      # y-then-x

    measurements: List[Measurement] = []
    for balloon_no, (txt, (cx, cy)) in enumerate(items, start=1):
        clean_txt = txt.strip()


        measurements.append(
            Measurement(
                balloon_number=balloon_no,
                value=_extract_numeric(clean_txt) or clean_txt,
                unit=None,
                measurement_type=_classify_measurement(clean_txt),
                section=None,
                bbox=_harvest_bbox((txt, (cx, cy))),  # ← NEW: always give a bbox
                # optional convenience: keep the raw centre
                **{"center": (cx, cy)},
            )
        )

    return DrawingResponse(filename=filename, measurements=measurements)