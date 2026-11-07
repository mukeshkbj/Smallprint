#!/usr/bin/env python
"""Step 5: before/after evaluation with AuditKIT.

Runs each model on both test sets with the same system prompt:
  base   - Tiny Aya Water, zero-shot (the "before")
  sft    - base + SFT adapter            (optional, --with-sft)
  final  - SFT + DPO, merged             (the "after")
  ref    - an optional hosted model via LiteLLM (closed-model reference for the ROI table)

Writes results/<model>__<testset>.json (+ predictions), results/summary.md with
the comparison tables, and AuditKIT's own base->final comparison per test set.

Usage:
  python scripts/05_evaluate.py
  python scripts/05_evaluate.py --max-samples 300            # quick check
  python scripts/05_evaluate.py --final yourname/smallprint-tiny-aya-water-3b --with-sft
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smallprint.config import load_config  # noqa: E402
from smallprint.data import read_jsonl  # noqa: E402
from smallprint.evaluate import compare, run_eval  # noqa: E402
from smallprint.metrics import comparison_table  # noqa: E402

TEST_SETS = {"lexglue": "lexglue_test.jsonl", "claudette_ml": "claudette_ml_test.jsonl"}


def load_rows(path: Path, max_samples: int | None, seed: int) -> list[dict]:
    rows = list(read_jsonl(path))
    if max_samples and len(rows) > max_samples:
        rows = random.Random(seed).sample(rows, max_samples)  # uniform: keeps the natural class mix
    return rows


def make_model(kind: str, cfg: dict, model_id: str, adapter: str | None = None):
    e = cfg["eval"]
    if kind == "litellm":
        from smallprint.model import LiteLLMClauseModel
        return LiteLLMClauseModel(model_id, max_new_tokens=e["max_new_tokens"])
    if e["backend"] == "vllm":
        from smallprint.model import VLLMClauseModel
        return VLLMClauseModel(model_id, adapter_path=adapter, max_new_tokens=e["max_new_tokens"])
    from smallprint.model import HFClauseModel
    return HFClauseModel(model_id, adapter_id=adapter, load_in_4bit=e["load_in_4bit"],
                         batch_size=e["batch_size"], max_new_tokens=e["max_new_tokens"])


def replay(out: Path, name: str, set_name: str, rows: list[dict]):
    """A no-GPU 'model' that returns the raw outputs saved by an earlier run, so the
    baseline is scored by AuditKIT again (and can be compared) without regenerating it.
    Returns None unless the saved run covers exactly these clauses."""
    preds, summ = out / f"{name}__{set_name}.predictions.jsonl", out / f"{name}__{set_name}.json"
    if not preds.exists() or not summ.exists():
        return None
    raw = {p["id"]: p["raw"] for p in read_jsonl(preds)}
    if {r["id"] for r in rows} != set(raw):
        return None
    by_text = {r["text"]: raw[r["id"]] for r in rows}

    def fn(prompts):
        return [by_text.get(p, "") for p in prompts]
    fn.name = f"replay:{name}"
    return fn, json.loads(summ.read_text())


def free(model) -> None:
    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--final", help="final model dir or Hub id (default runs/dpo/final)")
    ap.add_argument("--with-sft", action="store_true", help="also score base+SFT adapter")
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--skip-final", action="store_true", help="baseline only (before any training)")
    ap.add_argument("--reuse-base", action="store_true",
                    help="re-score the saved baseline predictions instead of regenerating them")
    ap.add_argument("--max-samples", type=int)
    ap.add_argument("--test-sets", default="lexglue,claudette_ml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    e = cfg["eval"]
    out = Path(e["out_dir"])
    work = Path(cfg["data"]["work_dir"])
    max_samples = args.max_samples or e.get("max_samples")
    sets = {k: load_rows(work / "eval" / TEST_SETS[k], max_samples, cfg["seed"])
            for k in args.test_sets.split(",")}

    runs = []
    if not args.skip_base:
        runs.append(("base", "hf", cfg["model"]["base"], None))
    if args.with_sft:
        runs.append(("sft", "hf", cfg["model"]["base"], str(Path(cfg["sft"]["output_dir"]) / "adapter")))
    if not args.skip_final:
        runs.append(("final", "hf", args.final or str(Path(cfg["dpo"]["output_dir"]) / "final"), None))
    if e.get("reference_model"):
        runs.append(("reference", "litellm", e["reference_model"], None))

    summaries: dict[str, dict[str, dict]] = {k: {} for k in sets}
    ak_results: dict[str, dict] = {k: {} for k in sets}

    def record(name: str, set_name: str, s: dict, r) -> None:
        summaries[set_name][name] = s
        ak_results[set_name][name] = r
        print(f"{name} on {set_name}: unfair_f1={s['unfair_f1']:.3f} "
              f"macro_f1={s['category_macro_f1']:.3f} valid={s['json_valid_rate']:.3f}")

    for name, kind, model_id, adapter in runs:
        if name == "base" and args.reuse_base:
            replayed = {k: replay(out, "base", k, rows) for k, rows in sets.items()}
            if all(v is not None for v in replayed.values()):
                for set_name, rows in sets.items():
                    fn, saved = replayed[set_name]
                    s, r = run_eval(fn, rows, f"base__{set_name}", out, source=set_name, model_label=saved["model"])
                    s["usage"] = saved.get("usage", {})  # keep the real GPU timings from the original run
                    (out / f"base__{set_name}.json").write_text(json.dumps(s, indent=2, ensure_ascii=False, default=str))
                    record(name, set_name, s, r)
                print("base: re-scored saved predictions (no GPU)")
                continue
            print("base: saved predictions don't match this sample, running the model")
        model = make_model(kind, cfg, model_id, adapter)
        for set_name, rows in sets.items():
            s, r = run_eval(model, rows, f"{name}__{set_name}", out, source=set_name,
                            model_label=getattr(model, "name", model_id))
            record(name, set_name, s, r)
        free(model)

    lines = ["# Smallprint evaluation", ""]
    for set_name, by_model in summaries.items():
        n = next(iter(by_model.values()))["n"]
        lines += [f"## {set_name} test set (n={n})", "", comparison_table(by_model), ""]
        if set_name == "claudette_ml":
            for name, s in by_model.items():
                if s.get("per_language"):
                    lines += [f"### {name}: by language", "", comparison_table(s["per_language"]), ""]
        if "base" in ak_results[set_name] and "final" in ak_results[set_name]:
            txt = compare(ak_results[set_name]["base"], ak_results[set_name]["final"])
            (out / f"auditkit_compare__{set_name}.txt").write_text(txt)
            lines += ["AuditKIT base -> final comparison: "
                      f"[`auditkit_compare__{set_name}.txt`](auditkit_compare__{set_name}.txt)", ""]
    usage = {f"{m}__{k}": v[m]["usage"] for k, v in summaries.items() for m in v}
    (out / "usage.json").write_text(json.dumps(usage, indent=2))
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
