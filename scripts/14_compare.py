#!/usr/bin/env python
"""Step 14: one page that compares everything we tried, for the write-up.

Reads the result folders the other steps write and builds results/comparisons.md:

1. Approaches on Tiny Aya Water: base, SFT, DPO v1, DPO v2.
2. Base variants: Tiny Aya Water vs Tiny Aya Global (zero-shot and after the same SFT), per test set
   and per language.
3. Safety degradation (results/safety/summary.md from step 12).
4. CircuitKIT mixed-precision quantization (results/quantization/summary.md from step 13).

Sections whose results aren't there yet are marked "pending", so it can run at any time.

Usage:
  python scripts/14_compare.py [--water results] [--variant global=results-global ...] [--out results/comparisons.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smallprint.metrics import HEADLINE, LOWER_IS_BETTER  # noqa: E402

SETS = (("lexglue", "LexGLUE UNFAIR-ToS test (EN)"), ("claudette_ml", "Multilingual CLAUDETTE test (EN/DE/IT/PL)"))

# Run 1 (HF Job 6aef94790e0338d1bb83d766). The DPO v1 model was replaced on the Hub by v2, so its
# numbers are kept here rather than re-read from a results file.
DPO_V1 = {
    "lexglue": {"json_valid_rate": 0.999, "unfair_f1": 0.585, "category_micro_f1": 0.535,
                "category_macro_f1": 0.572, "false_alarm_rate": 0.163, "miss_rate": 0.023,
                "evidence_grounding_rate": 0.985},
    "claudette_ml": {"json_valid_rate": 0.986, "unfair_f1": 0.450, "category_micro_f1": 0.396,
                     "category_macro_f1": 0.358, "false_alarm_rate": 0.199, "miss_rate": 0.122,
                     "evidence_grounding_rate": 0.952},
}


def load(folder: Path, run: str, test_set: str) -> dict | None:
    f = folder / f"{run}__{test_set}.json"
    return json.loads(f.read_text()) if f.exists() else None


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "–"


def metric_table(cols: dict[str, dict | None], bold_best: bool = True) -> list[str]:
    names = list(cols)
    lines = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for key, label in HEADLINE:
        vals = [(cols[n] or {}).get(key) for n in names]
        nums = [v for v in vals if isinstance(v, (int, float))]
        best = (min(nums) if key in LOWER_IS_BETTER else max(nums)) if nums and bold_best else None
        cells = [f"**{fmt(v)}**" if best is not None and v == best and len(nums) > 1 else fmt(v) for v in vals]
        arrow = " (lower is better)" if key in LOWER_IS_BETTER else ""
        lines.append(f"| {label}{arrow} | " + " | ".join(cells) + " |")
    return lines


def score(runs: dict[str, dict | None]) -> float | None:
    """Promotion score used by step 9: mean of unfair F1 and category macro-F1 over both test sets."""
    vals = []
    for r in runs.values():
        if not r:
            return None
        vals += [r["unfair_f1"], r["category_macro_f1"]]
    return sum(vals) / len(vals)


def approaches(water: Path) -> list[str]:
    out = ["## 1. Approaches (Tiny Aya Water)", "",
           "Same data, same base. `final` in the results folder is DPO v2; DPO v1 is from run 1.", ""]
    for key, title in SETS:
        cols = {"base": load(water, "base", key), "SFT": load(water, "sft", key),
                "SFT + DPO v1": DPO_V1[key], "SFT + DPO v2": load(water, "final", key)}
        out += [f"### {title}", ""] + metric_table(cols) + [""]
    rows = []
    for name, run in (("base", "base"), ("SFT", "sft"), ("SFT + DPO v2", "final")):
        s = score({k: load(water, run, k) for k, _ in SETS})
        rows.append(f"| {name} | {fmt(s)} |")
    v1 = sum(DPO_V1[k][m] for k, _ in SETS for m in ("unfair_f1", "category_macro_f1")) / 4
    rows.insert(2, f"| SFT + DPO v1 | {fmt(v1)} |")
    out += ["Promotion score (mean of unfair-term F1 and category macro-F1 over both test sets):", "",
            "| Model | Score |", "|---|---:|"] + rows + [""]
    return out


def per_language(cols: dict[str, dict | None], key: str) -> list[str]:
    langs = sorted({lang for r in cols.values() if r for lang in r.get("per_language", {})})
    if not langs:
        return []
    names = list(cols)
    lines = ["| Language | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for lang in langs:
        vals = [((cols[n] or {}).get("per_language", {}).get(lang) or {}).get(key) for n in names]
        lines.append(f"| {lang.upper()} | " + " | ".join(fmt(v) for v in vals) + " |")
    return lines


def variants(water: Path, others: dict[str, Path]) -> list[str]:
    out = ["## 2. Base variants", "",
           "Same training data and SFT recipe on each Tiny Aya variant (all 3.35B, same architecture).", ""]
    folders = {"water": water, **others}
    have = {n: f for n, f in folders.items() if load(f, "sft", "lexglue") or load(f, "base", "lexglue")}
    if len(have) < 2:
        return out + ["Pending: only " + ", ".join(have) + " has results so far.", ""]
    for key, title in SETS:
        cols = {}
        for n, f in have.items():
            cols[f"{n.title()} base"] = load(f, "base", key)
            cols[f"{n.title()} SFT"] = load(f, "sft", key)
        out += [f"### {title}", ""] + metric_table(cols, bold_best=False) + [""]
    cl = {}
    for n, f in have.items():
        cl[f"{n.title()} base"] = load(f, "base", "claudette_ml")
        cl[f"{n.title()} SFT"] = load(f, "sft", "claudette_ml")
    for metric, label in (("unfair_f1", "unfair-term F1"), ("category_macro_f1", "category macro-F1"),
                          ("miss_rate", "missed unfair clauses, lower is better")):
        table = per_language(cl, metric)
        if table:
            out += [f"### CLAUDETTE by language: {label}", ""] + table + [""]
    rows = []
    for n, f in have.items():
        rows.append(f"| {n.title()} | {fmt(score({k: load(f, 'base', k) for k, _ in SETS}))} | "
                    f"{fmt(score({k: load(f, 'sft', k) for k, _ in SETS}))} |")
    out += ["Promotion score:", "", "| Variant | base | SFT |", "|---|---:|---:|"] + rows + [""]
    return out


def included(path: Path, title: str, what: str) -> list[str]:
    if not path.exists():
        return [f"## {title}", "", f"Pending: {what}.", ""]
    body = path.read_text().strip().splitlines()
    if body and body[0].startswith("# "):
        body = body[1:]
    return [f"## {title}", ""] + [ln.replace("\n", "") for ln in body] + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--water", default="results")
    ap.add_argument("--variant", action="append", default=[], help="name=folder, e.g. global=results-global")
    ap.add_argument("--out", default="results/comparisons.md")
    args = ap.parse_args()
    water = Path(args.water)
    others = dict(v.split("=", 1) for v in args.variant) or {"global": "results-global"}
    others = {k: Path(v) for k, v in others.items()}

    lines = ["# Smallprint: what we compared", ""]
    lines += approaches(water)
    lines += variants(water, others)
    lines += included(water / "safety" / "summary.md", "3. Safety degradation (base vs Smallprint)",
                      "run scripts/12_safety_eval.py")
    lines += included(water / "quantization" / "summary.md", "4. CircuitKIT mixed-precision quantization",
                      "run scripts/13_circuitkit_quantize.py")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
