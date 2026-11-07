"""Dataset loading, decontamination, balancing and record building.

Sources (see docs/data-licences.md):
  * LexGLUE UNFAIR-ToS (coastalcph/lex_glue, config ``unfair_tos``), English, CC BY 4.0.
  * Multilingual CLAUDETTE ToS corpus (joelniklaus/online_terms_of_service),
    EN/DE/IT/PL, CC BY-NC 2.5.

Both derive from the same CLAUDETTE annotation project, so the same English
sentences can appear in one dataset's train split and the other's test split.
``decontaminate`` and ``companies_overlapping`` remove that leakage before
any training data is written.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from .grounding import normalise
from .labels import CLAUDETTE_CODE_TO_KEY, LEXGLUE_INDEX_TO_KEY
from .prompts import SYSTEM_PROMPT, user_message
from .schema import target_json


@dataclass
class Clause:
    id: str
    text: str
    lang: str
    source: str              # "lexglue" | "claudette_ml"
    split: str               # "train" | "validation" | "test"
    unfair: bool
    categories: list[str] = field(default_factory=list)
    severity: str | None = None   # potentially_unfair | clearly_unfair | None
    company: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Clause:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# --------------------------------------------------------------------------- loaders

def from_lexglue(rows: Iterable[dict], split: str) -> list[Clause]:
    """Rows from ``load_dataset("coastalcph/lex_glue", "unfair_tos")[split]``.

    ``labels`` is a list of class indices; an empty list means no unfair term.
    """
    out = []
    for i, r in enumerate(rows):
        cats = sorted({LEXGLUE_INDEX_TO_KEY[int(x)] for x in (r.get("labels") or [])})
        out.append(Clause(
            id=f"lexglue-{split}-{i}", text=str(r["text"]).strip(), lang="en",
            source="lexglue", split=split, unfair=bool(cats), categories=cats,
        ))
    return out


_LEVELS_STR = {
    "untagged": None, "clearly_fair": None,
    "potentially_unfair": "potentially_unfair", "clearly_unfair": "clearly_unfair",
}
# Original CLAUDETTE numbering: 1 clearly fair, 2 potentially unfair, 3 clearly unfair.
_LEVELS_INT = {0: None, 1: None, 2: "potentially_unfair", 3: "clearly_unfair"}


def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(v)


def _severity(level) -> str | None:
    if level is None:
        return None
    if isinstance(level, str):
        s = level.strip().lower().replace(" ", "_").replace("-", "_")
        if s.isdigit():
            return _LEVELS_INT.get(int(s))
        if s not in _LEVELS_STR:
            raise ValueError(f"unknown unfairness_level {level!r}")
        return _LEVELS_STR[s]
    return _LEVELS_INT.get(int(level))


def from_multilingual(rows: Iterable[dict], split: str) -> list[Clause]:
    """Rows from ``load_dataset("joelniklaus/online_terms_of_service")[split]``.

    A sentence counts as unfair when its level is potentially or clearly unfair;
    its categories are the topic tags set on it. Topic-tagged but clearly fair
    sentences are kept as fair examples: they teach the model that mentioning a
    topic (e.g. governing law) is not by itself unfair.
    If ``unfairness_level`` is a datasets ClassLabel, convert it to its string
    name before calling this (the scripts do).
    """
    out = []
    for i, r in enumerate(rows):
        sev = _severity(r.get("unfairness_level"))
        topics = sorted({key for code, key in CLAUDETTE_CODE_TO_KEY.items() if _truthy(r.get(code))})
        unfair = sev is not None and bool(topics)
        lang = str(r.get("language") or "en").strip().lower()[:2]
        company = str(r.get("company") or "").strip() or None
        out.append(Clause(
            id=f"ml-{split}-{lang}-{company or 'x'}-{r.get('line_number', i)}-{i}",
            text=str(r["sentence"]).strip(), lang=lang, source="claudette_ml", split=split,
            unfair=unfair, categories=topics if unfair else [],
            severity=sev if unfair else None, company=company,
        ))
    return out


# --------------------------------------------------------------------------- decontamination

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def dedup_key(text: str) -> str:
    return _PUNCT.sub("", normalise(text)).replace("  ", " ").strip()


def shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    toks = dedup_key(text).split()
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def _similarity(a: list[str], b: list[str]) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def decontaminate(train: list[Clause], refs: list[Clause], threshold: float = 0.85
                  ) -> tuple[list[Clause], list[Clause]]:
    """Drop training clauses that equal or near-duplicate any reference clause.

    Near-duplicate = token-level similarity ratio (difflib) >= ``threshold``,
    checked only against references sharing at least half of the shorter
    sentence's word trigrams (a cheap prefilter). Returns (kept, dropped).
    """
    exact = {dedup_key(c.text) for c in refs}
    ref_tok = [dedup_key(c.text).split() for c in refs]
    ref_sh = [shingles(c.text) for c in refs]
    index: dict[tuple, list[int]] = defaultdict(list)
    for j, sh in enumerate(ref_sh):
        for s in sh:
            index[s].append(j)
    kept, dropped = [], []
    for c in train:
        key = dedup_key(c.text)
        if key in exact:
            dropped.append(c)
            continue
        sh = shingles(c.text)
        shared: Counter = Counter()
        for s in sh:
            shared.update(index.get(s, ()))
        toks = key.split()
        hit = any(
            n / max(1, min(len(sh), len(ref_sh[j]))) >= 0.5 and _similarity(toks, ref_tok[j]) >= threshold
            for j, n in shared.items()
        )
        (dropped if hit else kept).append(c)
    return kept, dropped


def companies_overlapping(ml_train: list[Clause], refs: list[Clause], min_share: float = 0.2) -> set[str]:
    """Companies whose English training sentences overlap the reference set.

    The multilingual corpus has parallel DE/IT/PL versions of each document, so
    if a company's English text sits in a test set, its translations leak too.
    Any company with >= ``min_share`` of its English sentences found (exactly or
    near-exactly) in ``refs`` is dropped in every language.
    """
    by_company: dict[str, list[Clause]] = defaultdict(list)
    for c in ml_train:
        if c.lang == "en" and c.company:
            by_company[c.company].append(c)
    flagged = set()
    for company, rows in by_company.items():
        _, dropped = decontaminate(rows, refs)
        if rows and len(dropped) / len(rows) >= min_share:
            flagged.add(company)
    return flagged


# --------------------------------------------------------------------------- balancing

def balance(clauses: list[Clause], fair_per_unfair: float = 2.0, seed: int = 42) -> list[Clause]:
    """Keep every unfair clause; sample fair clauses up to ``fair_per_unfair`` x
    the unfair count, per (source, language) group so no language is starved."""
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[Clause]] = defaultdict(list)
    for c in clauses:
        groups[(c.source, c.lang)].append(c)
    out: list[Clause] = []
    for key in sorted(groups):
        rows = groups[key]
        unfair = [c for c in rows if c.unfair]
        fair = [c for c in rows if not c.unfair]
        rng.shuffle(fair)
        n_fair = min(len(fair), max(int(round(len(unfair) * fair_per_unfair)), 1 if fair else 0))
        out.extend(unfair + fair[:n_fair])
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------- records

def sft_record(c: Clause, evidence: str = "", explanation: str = "",
               annotation_source: str = "none", system_prompt: str = SYSTEM_PROMPT) -> dict:
    """One CuratorKIT input row.

    The clause goes in ``instruction`` (CuratorKIT dedups on instruction+output
    and the messages exporter turns it into the user turn) and is copied to
    ``source_chunk`` so the hallucination gate can check the explanation
    against it. The system prompt travels as ``system_prompt`` metadata, which
    the messages exporter writes as the system turn.
    """
    return {
        "instruction": user_message(c.text),
        "output": target_json(c.unfair, c.categories, evidence, explanation),
        "system_prompt": system_prompt,
        "source_chunk": c.text,
        "clause_id": c.id,
        "lang": c.lang,
        "source_dataset": c.source,
        "annotation_source": annotation_source,
    }


def eval_record(c: Clause) -> dict:
    return {
        "id": c.id, "text": c.text, "lang": c.lang, "source": c.source,
        "gold_unfair": c.unfair, "gold_categories": c.categories, "severity": c.severity,
    }


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarise(clauses: list[Clause]) -> dict:
    by: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "unfair": 0})
    for c in clauses:
        k = f"{c.source}/{c.split}/{c.lang}"
        by[k]["total"] += 1
        by[k]["unfair"] += int(c.unfair)
    return dict(sorted(by.items()))
