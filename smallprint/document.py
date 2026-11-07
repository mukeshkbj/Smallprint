"""Whole-document reading: split a policy into clauses, classify each one,
then roll the grounded clause results up into a report.

The report is built deterministically from clause outputs. There is no second
LLM pass, so the document summary cannot say anything that a flagged clause
(with its verbatim evidence) does not support. This also keeps every model
call well inside Tiny Aya's 8K context.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .grounding import is_verbatim
from .labels import BY_KEY
from .schema import Parsed

# Abbreviations that end with a period but don't end a sentence (EN/DE/IT/PL).
_ABBREV = (
    "e.g", "i.e", "etc", "inc", "ltd", "llc", "co", "corp", "no", "nr", "vs", "mr", "mrs", "ms",
    "dr", "st", "art", "sec", "para", "u.s", "u.k", "z.b", "bzw", "ggf", "usw", "inkl", "abs",
    "s.p.a", "ecc", "pag", "np", "tj", "tzn", "ust", "pkt",
)
_ABBREV_RE = re.compile(r"\b(" + "|".join(re.escape(a) for a in _ABBREV) + r")\.$", re.IGNORECASE)
_SENT_END = re.compile(r"(?<=[.!?;])\s+(?=[\"'“(\[]?[A-ZÀ-ÞĄĆĘŁŃÓŚŹŻ0-9])")
_BULLET = re.compile(r"^\s*(?:[-*•]|\(?[0-9ivxIVX]{1,4}[.)]|[a-zA-Z][.)])\s+")
MIN_WORDS = 4
MAX_CHARS = 1200


def split_clauses(text: str) -> list[str]:
    """Paragraphs -> sentences, roughly matching CLAUDETTE's sentence-level units."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n|\r\n\s*\r\n", text or ""):
        para = " ".join(line.strip() for line in para.splitlines() if line.strip())
        if not para:
            continue
        pieces = _SENT_END.split(para)
        buf = ""
        for piece in pieces:
            buf = f"{buf} {piece}".strip() if buf else piece.strip()
            if _ABBREV_RE.search(buf):
                continue  # sentence ended on an abbreviation: keep accumulating
            out.append(buf)
            buf = ""
        if buf:
            out.append(buf)
    merged: list[str] = []
    for s in out:
        s = _BULLET.sub("", s).strip()
        if not s:
            continue
        if merged and len(s.split()) < MIN_WORDS:
            merged[-1] = f"{merged[-1]} {s}"
        else:
            merged.append(s)
    final: list[str] = []
    for s in merged:  # hard cap very long run-on clauses so prompts stay small
        while len(s) > MAX_CHARS:
            cut = s.rfind(", ", 0, MAX_CHARS)
            cut = cut if cut > MAX_CHARS // 2 else MAX_CHARS
            final.append(s[: cut + 1].strip())
            s = s[cut + 1 :].strip()
        if s:
            final.append(s)
    return [s for s in final if len(s.split()) >= MIN_WORDS]


@dataclass
class Finding:
    index: int
    clause: str
    categories: list[str]
    evidence: str
    explanation: str
    evidence_verified: bool


@dataclass
class Report:
    n_clauses: int
    findings: list[Finding] = field(default_factory=list)
    n_invalid: int = 0

    @property
    def category_counts(self) -> Counter:
        return Counter(k for f in self.findings for k in f.categories)

    def to_dict(self) -> dict:
        return {
            "n_clauses": self.n_clauses,
            "n_flagged": len(self.findings),
            "n_invalid_outputs": self.n_invalid,
            "category_counts": dict(self.category_counts),
            "findings": [f.__dict__ for f in self.findings],
        }

    def to_markdown(self) -> str:
        if not self.findings:
            return (f"Read {self.n_clauses} clauses. Nothing flagged as potentially unfair. "
                    "This is not legal advice; read anything important yourself.")
        counts = self.category_counts
        lines = [
            f"Read {self.n_clauses} clauses and flagged **{len(self.findings)}** as potentially unfair.",
            "",
            "| What to watch for | Clauses |",
            "|---|---:|",
        ]
        for k, v in counts.most_common():
            lines.append(f"| {BY_KEY[k].name} | {v} |")
        lines += ["", "### Flagged clauses", ""]
        for f in self.findings:
            names = ", ".join(BY_KEY[k].name for k in f.categories)
            check = "" if f.evidence_verified else " _(quote not found in text: treat with care)_"
            lines.append(f"**{names}**: {f.explanation}  ")
            lines.append(f"> {f.evidence}{check}")
            lines.append("")
        lines.append("This is not legal advice; read anything important yourself.")
        return "\n".join(lines)


def build_report(clauses: list[str], outputs: list[Parsed]) -> Report:
    rep = Report(n_clauses=len(clauses))
    for i, (c, p) in enumerate(zip(clauses, outputs)):
        if not p.valid:
            rep.n_invalid += 1
            continue
        if p.flagged:
            rep.findings.append(Finding(
                index=i, clause=c, categories=list(p.categories), evidence=p.evidence,
                explanation=p.explanation, evidence_verified=is_verbatim(p.evidence, c),
            ))
    return rep
