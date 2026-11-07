"""Before/after evaluation with AuditKIT.

Per-sample scorers run inside ``auditkit.evaluate`` (schema validity, unfair
decision, exact label set, no fabricated quote), so every run gets AuditKIT's
fingerprint, latency stats and ``auditkit.compare`` significance output.
Corpus-level metrics (micro/macro F1, false-alarm rate, grounding rate among
flagged clauses, per-language breakdown) are computed from the same
predictions in ``metrics.py``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from .grounding import is_verbatim
from .labels import ALL_KEYS, LEXGLUE_KEYS
from .metrics import Gold, by_language, clause_metrics
from .schema import OUTPUT_SCHEMA, parse_output

ModelFn = Callable[[Sequence[str]], list[str]]


def _ak():
    import auditkit as ak
    return ak


def keys_for(source: str) -> tuple[str, ...]:
    return LEXGLUE_KEYS if source == "lexglue" else ALL_KEYS


def build_samples(rows: Sequence[dict]):
    ak = _ak()
    return [
        ak.Sample(
            input=r["text"], id=r["id"],
            target=json.dumps({"unfair": r["gold_unfair"], "categories": r["gold_categories"]}),
            metadata={"gold_unfair": r["gold_unfair"], "gold_categories": r["gold_categories"],
                      "lang": r["lang"], "source": r["source"]},
            tags=[r["lang"], r["source"]],
        )
        for r in rows
    ]


def make_scorers(keys: Sequence[str]):
    ak = _ak()
    keyset = set(keys)

    @ak.scorer(direction=ak.Direction.MAXIMIZE, name="unfair_correct")
    def unfair_correct(sample, output):
        return float(parse_output(output).flagged == bool(sample.metadata["gold_unfair"]))

    @ak.scorer(direction=ak.Direction.MAXIMIZE, name="labels_exact")
    def labels_exact(sample, output):
        gold = set(sample.metadata["gold_categories"]) & keyset
        return float(set(parse_output(output).label_set) & keyset == gold)

    @ak.scorer(direction=ak.Direction.MAXIMIZE, name="quote_not_fabricated")
    def quote_not_fabricated(sample, output):
        p = parse_output(output)
        return float((not p.flagged) or is_verbatim(p.evidence, sample.input))

    return [
        ak.JsonSchemaValid(OUTPUT_SCHEMA, backend="minimal", strip_fences=True),
        unfair_correct, labels_exact, quote_not_fabricated,
    ]


def run_eval(model_fn: ModelFn, rows: Sequence[dict], run_name: str, out_dir: str | Path,
             source: str | None = None, model_label: str | None = None):
    """Evaluate one model on one test set. Writes results/<run_name>.json and
    results/<run_name>.predictions.jsonl. Returns (summary_dict, auditkit RunResult)."""
    ak = _ak()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = source or rows[0]["source"]
    keys = keys_for(source)

    samples = build_samples(rows)
    result = ak.evaluate(samples, model_fn, make_scorers(keys), experiment_name="smallprint",
                         tags=[run_name, source])

    raw_by_id = {p.sample_id: (p.raw_output or "") for p in result.predictions}
    raws = [raw_by_id.get(r["id"], "") for r in rows]
    parsed = [parse_output(x) for x in raws]
    golds = [Gold(r["text"], bool(r["gold_unfair"]), frozenset(r["gold_categories"]), r["lang"]) for r in rows]

    summary = {
        "run": run_name,
        "model": model_label or getattr(model_fn, "name", repr(model_fn)),
        "test_set": source,
        "auditkit_fingerprint": result.fingerprint,
        "auditkit_headline": result.headline,
        **clause_metrics(golds, parsed, keys),
        "per_language": by_language(golds, parsed, keys) if len({g.lang for g in golds}) > 1 else {},
        "usage": dict(getattr(model_fn, "stats", {}) or {}),
    }
    if summary["usage"].get("clauses"):
        u = summary["usage"]
        u["output_tokens_per_clause"] = u["output_tokens"] / u["clauses"]
        u["input_tokens_per_clause"] = u["input_tokens"] / u["clauses"] if u.get("input_tokens") else None
        u["output_tokens_per_second"] = u["output_tokens"] / u["seconds"] if u.get("seconds") else None

    (out_dir / f"{run_name}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    with (out_dir / f"{run_name}.predictions.jsonl").open("w", encoding="utf-8") as f:
        for r, raw, p in zip(rows, raws, parsed):
            f.write(json.dumps({"id": r["id"], "lang": r["lang"], "gold_unfair": r["gold_unfair"],
                                "gold_categories": r["gold_categories"], "raw": raw,
                                "valid": p.valid, "error": p.error, **p.to_dict()},
                               ensure_ascii=False) + "\n")
    return summary, result


def compare(base_result, candidate_result) -> str:
    """AuditKIT's own before/after comparison (per-metric deltas, flips, latency)."""
    return _ak().compare(base_result, candidate_result).summary()
