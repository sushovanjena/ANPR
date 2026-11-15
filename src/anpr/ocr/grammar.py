"""Indian registration-plate grammar and position-aware error correction.

Standard plates follow ``SS DD [A-Z]{0,3} NNNN``: a two-letter state or union
territory code, a one- or two-digit district number, an optional series of up
to three letters and a four-digit number. Bharat-series plates follow
``YY BH NNNN [A-Z]{1,2}``. OCR confuses look-alike characters (0/O, 1/I, 8/B,
5/S, 2/Z), and the grammar tells us which class each position must belong to,
so a wrong class can be repaired deterministically.
"""

from __future__ import annotations

import re

STATE_CODES = frozenset(
    "AN AP AR AS BR CH CG DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ "
    "NL OD OR PB PY RJ SK TN TR TS UK UP WB".split()
)

_STANDARD_STRICT = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{4})$")
_BHARAT = re.compile(r"^(\d{2})BH(\d{4})([A-Z]{1,2})$")

# Characters that look alike: digit form -> letter form and the reverse.
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}
_TO_DIGIT = {v: k for k, v in _TO_LETTER.items()} | {"Q": "0", "D": "0", "T": "7"}


def normalise(text: str) -> str:
    """Upper-case a raw OCR string and drop everything except A-Z and 0-9."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _fix(ch: str, want: str) -> str:
    """Map ``ch`` to the requested class (``"L"`` letter or ``"D"`` digit)."""
    if want == "D" and ch.isalpha():
        return _TO_DIGIT.get(ch, ch)
    if want == "L" and ch.isdigit():
        return _TO_LETTER.get(ch, ch)
    return ch


def _apply_template(text: str, template: str) -> str:
    return "".join(_fix(c, t) for c, t in zip(text, template, strict=True))


def is_valid(text: str) -> bool:
    """Return True if ``text`` is a well-formed Indian plate string."""
    m = _STANDARD_STRICT.match(text)
    if m and m.group(1) in STATE_CODES:
        # Single-digit districts exist only for Delhi (DL1C, DL8S, ...).
        return len(m.group(2)) == 2 or m.group(1) == "DL"
    return bool(_BHARAT.match(text))


def correct(text: str) -> tuple[str, bool]:
    """Repair look-alike confusions so that ``text`` fits the plate grammar.

    Tries the standard layout with 1 to 3 series letters, then the Bharat
    layout, and returns the candidate that needs the fewest substitutions and
    has a valid state code.

    Args:
        text: Raw OCR string (will be normalised).

    Returns:
        ``(string, valid)``. When no layout fits, the normalised input is
        returned with ``valid=False`` rather than a guess.
    """
    text = normalise(text)
    if is_valid(text):
        return text, True
    best: tuple[int, str] | None = None
    candidates: list[str] = []
    for district in (1, 2):
        for series in (1, 2, 3):
            candidates.append("LL" + "D" * district + "L" * series + "DDDD")
    for series in (1, 2):
        candidates.append("DDLLDDDD" + "L" * series)  # Bharat series
    for template in candidates:
        if len(template) != len(text):
            continue
        fixed = _apply_template(text, template)
        if not is_valid(fixed):
            continue
        cost = sum(a != b for a, b in zip(text, fixed, strict=True))
        if best is None or cost < best[0]:
            best = (cost, fixed)
    if best is None:
        return text, False
    return best[1], True


def split_fields(text: str) -> dict[str, str] | None:
    """Split a valid plate into its fields, or return None if it is invalid."""
    m = _STANDARD_STRICT.match(text)
    if m and is_valid(text):
        state, district, series, number = m.groups()
        return {
            "state": state,
            "district": district,
            "series": series,
            "number": number,
        }
    m = _BHARAT.match(text)
    if m:
        year, number, series = m.groups()
        return {"year": year, "series": series, "number": number, "state": "BH"}
    return None
