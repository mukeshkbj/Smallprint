"""Corpus-level metrics for clause classification + evidence grounding.

Per-sample scores go through AuditKIT (see evaluate.py); the numbers that only
make sense over a whole test set (micro/macro F1, false-alarm rate, grounding
rate among flagged clauses) are computed here from the same predictions.

Category F1 follows LexGLUE's UNFAIR-ToS convention: multi-label, one binary
decision per category, micro- and macro-averaged over the category set.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .grounding import is_verbatim
from .schema import Parsed


@dataclass
class Gold:
    text: str
    unfair: bool
    categories: frozenset[str]
    lang: str = "en"


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def clause_metrics(golds: Sequence[Gold], preds: Sequence[Parsed], keys: Sequence[str]) -> dict:
    assert len(golds) == len(preds), "gold/pred length mismatch"
    n = len(golds)
    keys = list(keys)
    keyset = set(keys)

    valid = sum(p.valid for p in preds)
    parsed = sum(p.parsed_json for p in preds)

    # binary unfair detection
    tp = sum(g.unfair and p.flagged for g, p in zip(golds, preds))
    fp = sum((not g.unfair) and p.flagged for g, p in zip(golds, preds))
    fn = sum(g.unfair and not p.flagged for g, p in zip(golds, preds))
    bp, br, bf = _prf(tp, fp, fn)
    n_fair = sum(not g.unfair for g in golds)
    n_unfair = n - n_fair

    # per-category, restricted to the evaluated key set
    per: dict[str, dict] = {}
    TP = FP = FN = 0
    for k in keys:
        ctp = sum((k in g.categories) and (k in p.label_set) for g, p in zip(golds, preds))
        cfp = sum((k not in g.categories) and (k in p.label_set) for g, p in zip(golds, preds))
        cfn = sum((k in g.categories) and (k not in p.label_set) for g, p in zip(golds, preds))
        cp, cr, cf = _prf(ctp, cfp, cfn)
        per[k] = {"precision": cp, "recall": cr, "f1": cf, "support": ctp + cfn}
        TP, FP, FN = TP + ctp, FP + cfp, FN + cfn
    _, _, micro = _prf(TP, FP, FN)
    supported = [per[k]["f1"] for k in keys if per[k]["support"] > 0]
    macro = sum(supported) / len(supported) if supported else 0.0

    exact = sum((g.categories & keyset) == (p.label_set & keyset) for g, p in zip(golds, preds))

    flagged = [(g, p) for g, p in zip(golds, preds) if p.flagged]
    grounded = sum(is_verbatim(p.evidence, g.text) for g, p in flagged)

    return {
        "n": n, "n_unfair": n_unfair, "n_fair": n_fair,
        "json_valid_rate": valid / n if n else 0.0,
        "json_parse_rate": parsed / n if n else 0.0,
        "unfair_precision": bp, "unfair_recall": br, "unfair_f1": bf,
        "false_alarm_rate": fp / n_fair if n_fair else 0.0,
        "miss_rate": fn / n_unfair if n_unfair else 0.0,
        "category_micro_f1": micro, "category_macro_f1": macro,
        "label_exact_match": exact / n if n else 0.0,
        "evidence_grounding_rate": grounded / len(flagged) if flagged else 0.0,
        "n_flagged": len(flagged),
        "per_category": per,
    }


def by_language(golds: Sequence[Gold], preds: Sequence[Parsed], keys: Sequence[str]) -> dict[str, dict]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(golds):
        groups[g.lang].append(i)
    return {
        lang: clause_metrics([golds[i] for i in idx], [preds[i] for i in idx], keys)
        for lang, idx in sorted(groups.items())
    }


HEADLINE = (
    ("json_valid_rate", "Valid JSON"),
    ("unfair_f1", "Unfair-term F1"),
    ("category_micro_f1", "Category micro-F1"),
    ("category_macro_f1", "Category macro-F1"),
    ("false_alarm_rate", "False alarms (fair clauses)"),
    ("miss_rate", "Missed unfair clauses"),
    ("evidence_grounding_rate", "Evidence is a real quote"),
)
LOWER_IS_BETTER = {"false_alarm_rate", "miss_rate"}


def comparison_table(results: dict[str, dict]) -> str:
    """Markdown table: one column per model run, one row per headline metric."""
    names = list(results)
    lines = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for key, label in HEADLINE:
        arrow = " (lower is better)" if key in LOWER_IS_BETTER else ""
        cells = [f"{results[n].get(key, float('nan')):.3f}" for n in names]
        lines.append(f"| {label}{arrow} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
