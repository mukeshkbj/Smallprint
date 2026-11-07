"""Verbatim evidence check: is the quoted evidence really in the clause?

This is the deterministic half of grounding (CuratorKIT's LLM judge covers the
explanation). It is deliberately strict: after light normalisation
(Unicode NFKC, case, whitespace, quote and dash styles) the evidence must
appear in the clause as a contiguous substring. Ellipses split the evidence
into segments that must each appear, in order.
"""

from __future__ import annotations

import re
import unicodedata

_QUOTES = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
})
_WS = re.compile(r"\s+")
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")

MIN_EVIDENCE_CHARS = 8


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").translate(_QUOTES).casefold()
    return _WS.sub(" ", t).strip()


def is_verbatim(evidence: str, clause: str, min_chars: int = MIN_EVIDENCE_CHARS) -> bool:
    ev = normalise(evidence).strip(" \"'.,;:")
    if len(ev) < min_chars:
        return False
    hay = normalise(clause)
    pos = 0
    for seg in _ELLIPSIS.split(ev):
        seg = seg.strip(" \"'.,;:")
        if not seg:
            continue
        idx = hay.find(seg, pos)
        if idx == -1:
            return False
        pos = idx + len(seg)
    return True
