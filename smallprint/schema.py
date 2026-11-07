"""Output schema and a tolerant parser for model replies."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .labels import ALL_KEYS, normalise_category

OUTPUT_SCHEMA: dict = {
    "type": "object",
    "required": ["unfair", "categories", "evidence", "explanation"],
    "additionalProperties": False,
    "properties": {
        "unfair": {"type": "boolean"},
        "categories": {
            "type": "array",
            "items": {"type": "string", "enum": list(ALL_KEYS)},
            "uniqueItems": True,
        },
        "evidence": {"type": "string"},
        "explanation": {"type": "string"},
    },
}

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


@dataclass
class Parsed:
    """A parsed model reply.

    ``valid`` is strict: the reply is one JSON object that matches the schema
    *and* is internally consistent (unfair=false <=> no categories). Scoring
    treats an invalid reply as "fair" with no evidence, which is the
    conservative reading for a consumer tool.
    """

    raw: str
    valid: bool = False
    parsed_json: bool = False
    unfair: bool = False
    categories: list[str] = field(default_factory=list)
    evidence: str = ""
    explanation: str = ""
    error: str = ""

    @property
    def label_set(self) -> frozenset[str]:
        return frozenset(self.categories) if (self.valid and self.unfair) else frozenset()

    @property
    def flagged(self) -> bool:
        return self.valid and self.unfair

    def to_dict(self) -> dict:
        return {
            "unfair": self.unfair,
            "categories": list(self.categories),
            "evidence": self.evidence,
            "explanation": self.explanation,
        }


def _first_json_object(text: str) -> str | None:
    """Return the first balanced {...} span, respecting strings."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def parse_output(raw: str) -> Parsed:
    p = Parsed(raw=raw if isinstance(raw, str) else str(raw))
    text = _FENCE.sub("", p.raw.strip())
    blob = _first_json_object(text)
    if blob is None:
        p.error = "no_json_object"
        return p
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError as e:
        p.error = f"json_decode_error:{e.msg}"
        return p
    if not isinstance(obj, dict):
        p.error = "not_an_object"
        return p
    p.parsed_json = True

    problems: list[str] = []
    missing = [k for k in OUTPUT_SCHEMA["required"] if k not in obj]
    if missing:
        problems.append("missing:" + ",".join(missing))
    extra = [k for k in obj if k not in OUTPUT_SCHEMA["properties"]]
    if extra:
        problems.append("extra:" + ",".join(extra))

    unfair = obj.get("unfair")
    if isinstance(unfair, bool):
        p.unfair = unfair
    else:
        problems.append("unfair_not_bool")

    cats = obj.get("categories", [])
    if isinstance(cats, list):
        keys: list[str] = []
        for c in cats:
            k = normalise_category(c)
            if k is None:
                problems.append(f"unknown_category:{c}")
            elif k not in keys:
                keys.append(k)
        p.categories = keys
    else:
        problems.append("categories_not_list")

    for k in ("evidence", "explanation"):
        v = obj.get(k, "")
        if isinstance(v, str):
            setattr(p, k, v.strip())
        else:
            problems.append(f"{k}_not_string")

    if isinstance(unfair, bool):
        if unfair and not p.categories:
            problems.append("unfair_without_categories")
        if not unfair and p.categories:
            problems.append("categories_without_unfair")
        if unfair and not p.evidence:
            problems.append("unfair_without_evidence")

    p.valid = not problems
    p.error = ";".join(problems)
    return p


def target_json(unfair: bool, categories: list[str], evidence: str = "", explanation: str = "") -> str:
    """Canonical serialisation of a training target (stable key order)."""
    obj = {
        "unfair": bool(unfair),
        "categories": list(categories) if unfair else [],
        "evidence": evidence if unfair else "",
        "explanation": explanation if unfair else "",
    }
    return json.dumps(obj, ensure_ascii=False)
