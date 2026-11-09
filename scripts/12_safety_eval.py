#!/usr/bin/env python
"""Step 12: safety evaluation, base model vs Smallprint (see smallprint/safety.py for the suites).

Usage:
  python scripts/12_safety_eval.py                                   # base vs published model, all suites
  python scripts/12_safety_eval.py --models base,smallprint,dpo --suites injection
  python scripts/12_safety_eval.py --n-harmful 50 --max-clauses 40 --no-judge   # quick check

Models: base (model.base), smallprint (hub.model_repo), dpo (hub.dpo_model_repo), or name=repo_id.
Writes results/safety/<model>__<suite>.json, <model>__<suite>.responses.jsonl and results/safety/summary.md.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smallprint.config import load_config  # noqa: E402
from smallprint.data import read_jsonl, write_jsonl  # noqa: E402
from smallprint.safety import (ATTACKS, JUDGE_PROMPT, injection_cases, injection_metrics,  # noqa: E402
                               keyword_label, load_advbench, load_xstest, parse_judge, refusal_metrics)
from smallprint.schema import parse_output  # noqa: E402

SUITES = ("redteam", "harmful", "xstest", "injection")
REDTEAM = {"prompt_injection": ["keyword", "injection_success"], "jailbreak": ["keyword", "injection_success"],
           "encoding": ["keyword", "injection_success"], "refusal": ["refusal"]}


def resolve_models(spec: str, cfg: dict) -> list[tuple[str, str]]:
    named = {"base": cfg["model"]["base"], "smallprint": cfg["hub"]["model_repo"],
             "dpo": cfg["hub"].get("dpo_model_repo")}
    out = []
    for item in spec.split(","):
        name, _, repo = item.partition("=")
        out.append((name, repo or named[name]))
    return out


def make_judge(cfg: dict, enabled: bool, model: str | None = None):
    if not enabled or not (model or cfg["judge"].get("model")):
        return None
    from smallprint.teacher import litellm_completion, resolve_api_key
    j = cfg["judge"]
    return litellm_completion(model or j["model"], temperature=0.0, max_tokens=10, api_base=j.get("api_base"),
                              api_key=resolve_api_key(j))


def label_all(rows: list[dict], responses: list[str], judge, workers: int = 4, attempts: int = 6,
              wait: float = 5.0) -> tuple[list[str], list[str]]:
    """LLM-judge labels, retried with backoff (the router answers "model busy" under load); a keyword
    label only when the judge still fails, recorded as labelled_by="keyword" so the share is reported."""
    import time

    def one(i: int) -> tuple[str, str]:
        if judge is not None:
            for attempt in range(attempts):
                try:
                    lab = parse_judge(judge(JUDGE_PROMPT.format(prompt=rows[i]["prompt"],
                                                                response=responses[i][:2000])))
                    if lab:
                        return lab, "judge"
                    break  # answered, but not with a label: retrying the same prompt won't help
                except Exception as e:  # noqa: BLE001 - fall back to keywords on any API error
                    if attempt == attempts - 1:
                        print("judge error:", str(e)[:120], flush=True)
                    else:
                        time.sleep(wait * 2 ** attempt)
        return keyword_label(responses[i]), "keyword"
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, range(len(rows))))
    return [r[0] for r in res], [r[1] for r in res]


def rejudge(out: Path, names: list[str], judge, group_of: dict[str, dict[str, str]]) -> dict[str, dict]:
    """Re-label the saved AdvBench/XSTest responses of every model with the same judge, so refusal rates
    are comparable across models even if the judge failed during the original run. Outputs unchanged.
    group_of[suite][id] is the prompt's safe/unsafe group (files written before the "verdict" key
    stored the judge label under "label", so the group is taken from the source datasets)."""
    all_results = {}
    for name in names:
        results = json.loads((out / f"{name}.json").read_text())
        for suite in ("harmful", "xstest"):
            f = out / f"{name}__{suite}.responses.jsonl"
            if not f.exists():
                continue
            rows = [{**r, "label": group_of[suite][r["id"]]} for r in read_jsonl(f)]
            labels, how = label_all(rows, [r["response"] for r in rows], judge)
            met = refusal_metrics(rows, labels)
            met["judged_by_llm"] = sum(h == "judge" for h in how) / len(how)
            results[suite] = met
            write_jsonl(f, [{**r, "verdict": lab, "labelled_by": h} for r, lab, h in zip(rows, labels, how)])
            print(f"{name} {suite} (re-judged): {met}", flush=True)
        (out / f"{name}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
        all_results[name] = results
    return all_results


def run_model(name: str, repo: str, args, cfg: dict, judge, out: Path) -> dict:
    import torch
    from smallprint.model import HFClauseModel

    print(f"=== {name}: {repo}", flush=True)
    m = HFClauseModel(repo, batch_size=args.batch_size, token=os.environ.get("HF_TOKEN"))
    results: dict = {"model": repo}

    if "redteam" in args.suites:
        from auditkit.redteam import RedTeamRunner
        runner = RedTeamRunner(model=lambda ps: m.chat(ps, max_new_tokens=200))
        rows, rt = [], {}
        for probe, dets in REDTEAM.items():
            res = runner.run(probes=[probe], detectors=dets)
            rt[probe] = res.pass_rate
            rows += [{"probe": r.probe_name, "detector": r.detector_name, "prompt": r.prompt,
                      "output": r.output, "passed": r.passed} for r in res.results]
        rt["mean"] = sum(rt.values()) / len(rt)
        results["redteam"] = rt
        write_jsonl(out / f"{name}__redteam.responses.jsonl", rows)
        print(f"{name} redteam pass rates: {rt}", flush=True)

    for suite, loader in (("harmful", lambda: load_advbench(Path(cfg["data"]["work_dir"]) / "safety_cache", args.n_harmful)),
                          ("xstest", lambda: load_xstest(Path(cfg["data"]["work_dir"]) / "safety_cache"))):
        if suite not in args.suites:
            continue
        rows = loader()
        responses = m.chat([r["prompt"] for r in rows], max_new_tokens=args.max_new_tokens)
        labels, how = label_all(rows, responses, judge)
        met = refusal_metrics(rows, labels)
        met["judged_by_llm"] = sum(h == "judge" for h in how) / len(how)
        results[suite] = met
        write_jsonl(out / f"{name}__{suite}.responses.jsonl",  # label = the prompt's safe/unsafe group
                    [{**r, "response": resp, "verdict": lab, "labelled_by": h}
                     for r, resp, lab, h in zip(rows, responses, labels, how)])
        print(f"{name} {suite}: {met}", flush=True)

    if "injection" in args.suites:
        work = Path(cfg["data"]["work_dir"]) / "eval"
        rows = []
        for f in ("lexglue_test.jsonl", "claudette_ml_test.jsonl"):
            rows += [r for r in read_jsonl(work / f) if r.get("gold_unfair")]
        if args.max_clauses:
            rows = rows[: args.max_clauses]
        cases = injection_cases(rows)
        parsed = [parse_output(o) for o in m([c["text"] for c in cases])]
        flagged, valid = [p.flagged for p in parsed], [p.valid for p in parsed]
        met = injection_metrics(cases, flagged, valid)
        for lang in sorted({c["lang"] for c in cases}):
            idx = [i for i, c in enumerate(cases) if c["lang"] == lang]
            sub = injection_metrics([cases[i] for i in idx], [flagged[i] for i in idx], [valid[i] for i in idx])
            met[f"by_lang.{lang}"] = {k: sub[k] for k in ("n_unfair_clauses", "clean_flag_rate", "mean_attack_success")}
        results["injection"] = met
        write_jsonl(out / f"{name}__injection.responses.jsonl",
                    [{**c, "flagged": f, "valid": v} for c, f, v in zip(cases, flagged, valid)])
        print(f"{name} injection: clean flag rate {met['clean_flag_rate']:.3f}, "
              f"mean attack success {met['mean_attack_success']:.3f}", flush=True)

    (out / f"{name}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return results


def summary_md(all_results: dict[str, dict]) -> str:
    names = list(all_results)
    head = "| | " + " | ".join(names) + " |\n|---|" + "---:|" * len(names)

    def row(label: str, get) -> str:
        vals = []
        for n in names:
            try:
                vals.append(f"{get(all_results[n]):.3f}")
            except (KeyError, TypeError):
                vals.append("–")
        return f"| {label} | " + " | ".join(vals) + " |"

    lines = ["# Safety evaluation", "", f"Models: " + ", ".join(f"`{n}` = {r['model']}" for n, r in all_results.items()),
             "", head]
    lines += [row("AuditKIT red team: mean pass rate", lambda r: r["redteam"]["mean"])]
    lines += [row(f"  {p} pass rate", lambda r, p=p: r["redteam"][p]) for p in REDTEAM]
    lines += [row("AdvBench harmful requests refused (higher is safer)", lambda r: r["harmful"]["unsafe.refusal_rate"]),
              row("XSTest unsafe prompts refused (higher is safer)", lambda r: r["xstest"]["unsafe.refusal_rate"]),
              row("XSTest safe prompts answered (higher is better)", lambda r: r["xstest"]["safe.full_compliance_rate"]),
              row("Refusal labels from the LLM judge (rest: keyword fallback)",
                  lambda r: (r["harmful"]["judged_by_llm"] * r["harmful"]["n"] + r["xstest"]["judged_by_llm"]
                             * r["xstest"]["n"]) / (r["harmful"]["n"] + r["xstest"]["n"])),
              row("Unfair clauses flagged, clean", lambda r: r["injection"]["clean_flag_rate"]),
              row("Unfair clauses flagged, harmless sentence added", lambda r: r["injection"]["control.flag_rate"])]
    lines += [row(f"Injection '{a}': flags removed (lower is better)", lambda r, a=a: r["injection"][f"{a}.attack_success"])
              for a in ATTACKS]
    lines += [row("Injection: mean flags removed (lower is better)", lambda r: r["injection"]["mean_attack_success"]), "",
              "Flags removed = share of the unfair clauses a model flagged clean that it no longer flags once the "
              "injection is appended. Red-team pass rates are AuditKIT's built-in detectors; refusal labels come "
              "from an LLM judge with a keyword fallback (see the *.responses.jsonl files)."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--models", default="base,smallprint")
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--n-harmful", type=int, default=200)
    ap.add_argument("--max-clauses", type=int, default=0, help="limit unfair clauses for the injection suite")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-model", help="override judge.model, e.g. openai/<repo>:<provider> on the HF router")
    ap.add_argument("--rejudge", action="store_true",
                    help="no generation: re-label the saved AdvBench/XSTest responses of --models with one judge")
    args = ap.parse_args()
    args.suites = [s for s in args.suites.split(",") if s]
    cfg = load_config(args.config)
    out = Path(cfg["eval"]["out_dir"]) / "safety"
    out.mkdir(parents=True, exist_ok=True)
    judge = make_judge(cfg, not args.no_judge, args.judge_model)
    if args.rejudge:
        if judge is None:
            raise SystemExit("--rejudge needs a judge (drop --no-judge)")
        print("judge:", args.judge_model or cfg["judge"]["model"], "| probe:", judge("Reply with the word ok."), flush=True)
        cache = Path(cfg["data"]["work_dir"]) / "safety_cache"
        group_of = {"harmful": {r["id"]: r["label"] for r in load_advbench(cache, None)},
                    "xstest": {r["id"]: r["label"] for r in load_xstest(cache)}}
        all_results = rejudge(out, [name for name, _ in resolve_models(args.models, cfg)], judge, group_of)
    else:
        all_results = {name: run_model(name, repo, args, cfg, judge, out)
                       for name, repo in resolve_models(args.models, cfg)}
    md = summary_md(all_results)
    (out / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
