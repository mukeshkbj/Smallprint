#!/usr/bin/env python
"""Step 13: circuit-guided mixed-precision quantization with CircuitKIT, for a lighter on-device model.

1. Circuit discovery (EAP-IG, node level) on the decision Smallprint makes first: after the prompt and
   '{"unfair":', the next token is " true" or " false". Clean/corrupted pairs swap an unfair training
   clause for a fair one, so the circuit is what carries the fair/unfair decision.
2. Four versions of the same model, scored by AuditKIT on the same test clauses:
     bf16                full precision (reference)
     int4_uniform        every attention and MLP layer in 4-bit
     int4_circuit_<f>    4-bit, except the fraction f of layers the circuit ranks most important (kept bf16)
     int4_random_<f>     same budget, layers to keep chosen at random (the control for "does the circuit help?")
3. Weight memory of each version, and results/quantization/summary.md.

Usage:
  python scripts/13_circuitkit_quantize.py                       # published model, f = 0.15
  python scripts/13_circuitkit_quantize.py --high-fraction 0.3 --max-samples 800
  python scripts/13_circuitkit_quantize.py --push                # also upload the circuit-guided model
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smallprint.config import load_config  # noqa: E402
from smallprint.data import Clause, read_jsonl  # noqa: E402
from smallprint.evaluate import run_eval  # noqa: E402
from smallprint.metrics import HEADLINE  # noqa: E402
from smallprint.prompts import SYSTEM_PROMPT, chat_messages  # noqa: E402

TASK = "smallprint_unfair"
ANSWER_PREFIX = '{"unfair":'


# --------------------------------------------------------------------------- discovery data
def prompt_text(tok, clause: str) -> str:
    """The chat-templated prompt plus the start of the answer, so the next token is the decision."""
    return tok.apply_chat_template(chat_messages(clause, SYSTEM_PROMPT), tokenize=False,
                                   add_generation_prompt=True) + ANSWER_PREFIX


def answer_ids(tok, example_prompt: str) -> tuple[int, int]:
    """Token ids of ' true' / ' false' as they follow the answer prefix (must be one token each)."""
    base = tok(example_prompt, add_special_tokens=False)["input_ids"]
    out = []
    for word in (" true", " false"):
        ids = tok(example_prompt + word, add_special_tokens=False)["input_ids"]
        if ids[: len(base)] != base or len(ids) != len(base) + 1:
            raise ValueError(f"{word!r} is not a single token after the answer prefix for this tokenizer")
        out.append(ids[-1])
    return out[0], out[1]


def discovery_pairs(clauses: list[Clause], tok, n_pairs: int, seed: int = 0) -> list[dict]:
    """Label-swap pairs: half clean=unfair/corrupted=fair, half the reverse (as CircuitKIT's BoolQ task)."""
    rng = random.Random(seed)
    unfair = [c for c in clauses if c.unfair]
    fair = [c for c in clauses if not c.unfair]
    k = n_pairs // 2
    u, f = rng.sample(unfair, min(k * 2, len(unfair))), rng.sample(fair, min(k * 2, len(fair)))
    t_id, f_id = answer_ids(tok, prompt_text(tok, u[0].text))
    rows = []
    for i in range(min(k, len(u) // 2, len(f) // 2)):
        rows.append({"clean": prompt_text(tok, u[i].text), "corrupted": prompt_text(tok, f[i].text),
                     "correct_idx": t_id, "incorrect_idx": f_id})
        rows.append({"clean": prompt_text(tok, f[k + i].text), "corrupted": prompt_text(tok, u[k + i].text),
                     "correct_idx": f_id, "incorrect_idx": t_id})
    return rows


def register_task(csv_path: Path) -> None:
    """A CircuitKIT task that reads our pairs. Reuses the BoolQ task's yes/no machinery
    (EAP dataloader with templated text, probability-of-correct-token metric)."""
    from circuitkit.tasks.builtins.boolq import BoolQTaskSpec
    from circuitkit.tasks.registry import is_task_registered, register_task as _register

    class SmallprintTask(BoolQTaskSpec):
        name = TASK

        def __init__(self, path: Path):
            self.path = Path(path)

        def _get_or_generate_csv(self, discovery_cfg, model):
            return self.path, True  # prompts in the CSV are already chat-templated

        def artifact_metadata(self, discovery_cfg):
            return {"task": TASK, "data_source": "smallprint training clauses (label-swap pairs)",
                    "algorithm": discovery_cfg.get("algorithm"), "level": discovery_cfg.get("level"),
                    "chat_template_mode": "on"}

    if not is_task_registered(TASK):
        _register(SmallprintTask(csv_path))


def shuffled(circuit, seed: int = 0):
    """Same scores, randomly reassigned among attention heads and among MLPs: the random-tier control."""
    rng = random.Random(seed)
    out = copy.copy(circuit)
    scores = dict(circuit.scores)
    for is_mlp in (False, True):
        keys = [k for k in scores if k.upper().startswith("MLP") == is_mlp]
        vals = [scores[k] for k in keys]
        rng.shuffle(vals)
        scores.update(zip(keys, vals))
    out.scores = scores
    return out


# --------------------------------------------------------------------------- size
def weight_bytes(model) -> dict:
    """Bytes of weights by precision: quanto 4-bit/8-bit linears vs everything left in bf16/fp32."""
    from collections import Counter
    by = Counter()
    seen = set()
    for mod in model.modules():
        qtype = getattr(mod, "weight_qtype", None)
        w = getattr(mod, "weight", None)
        if w is None or id(w) in seen:
            continue
        seen.add(id(w))
        n = w.numel() if hasattr(w, "numel") else 0
        if qtype is not None:
            bits = getattr(qtype, "bits", 8)
            by[f"int{bits}"] += n * bits / 8
        else:
            by[str(w.dtype).replace("torch.", "")] += n * w.element_size()
    for name, p in model.named_parameters():  # biases, norms, and anything not attached as .weight
        if id(p) not in seen:
            seen.add(id(p))
            by[str(p.dtype).replace("torch.", "")] += p.numel() * p.element_size()
    total = sum(by.values())
    return {"total_gb": total / 1e9, **{k: v / 1e9 for k, v in by.items()}}


# --------------------------------------------------------------------------- main
def load_eval_rows(cfg: dict, max_samples: int | None, seed: int) -> dict[str, list[dict]]:
    spec = importlib.util.spec_from_file_location("ev", Path(__file__).with_name("05_evaluate.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    work = Path(cfg["data"]["work_dir"]) / "eval"
    return {k: ev.load_rows(work / f, max_samples, seed) for k, f in ev.TEST_SETS.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--model", help="merged model to quantize (default hub.model_repo)")
    ap.add_argument("--high-fraction", type=float, default=0.15)
    ap.add_argument("--n-pairs", type=int, default=64)
    ap.add_argument("--discovery-batch", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=1600, help="test clauses per set (0 = all)")
    ap.add_argument("--configs", default="bf16,int4_uniform,int4_circuit,int4_random")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--push", action="store_true", help="upload the circuit-guided model (private)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import circuitkit as ck

    cfg = load_config(args.config)
    repo = args.model or cfg["hub"]["model_repo"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg["eval"]["out_dir"]) / "quantization"
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(repo)

    # 1. discovery
    clauses = [Clause.from_dict(r) for r in read_jsonl(Path(cfg["data"]["work_dir"]) / "interim" / "train_clauses.jsonl")]
    pairs = discovery_pairs(clauses, tok, args.n_pairs)
    import pandas as pd
    csv_path = out / "discovery_pairs.csv"
    pd.DataFrame(pairs).to_csv(csv_path, index=False)
    register_task(csv_path)
    # CircuitKIT 0.2.1 maps a Cohere2 model to its TransformerLens port either by a known Hub name
    # (CohereLabs/tiny-aya-*) or by a local folder's config.json; an unlisted Hub id such as our
    # fine-tune falls through to TransformerLens, which rejects Cohere2. So load from a local copy.
    local = repo
    if not Path(repo).expanduser().is_dir():
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
    hooked = ck.load_model(local, dtype=args.dtype, device=device, algorithm="eap-ig")
    circuit = ck.discover(hooked, TASK, algorithm="eap-ig", level="node", n_examples=len(pairs),
                          batch_size=args.discovery_batch, output_path=str(out / "circuit.pt"))
    top = sorted(circuit.scores.items(), key=lambda kv: -abs(kv[1]))[:30]
    (out / "circuit_top_nodes.json").write_text(json.dumps(top, indent=2))
    print("top circuit nodes:", top[:10], flush=True)
    del hooked
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # 2. versions
    hf = args.high_fraction
    plans = {"bf16": None, "int4_uniform": (circuit, 0.0), f"int4_circuit_{hf:g}": (circuit, hf),
             f"int4_random_{hf:g}": (shuffled(circuit), hf)}
    wanted = {c.split("_")[0] + ("_" + c.split("_")[1] if "_" in c else "") for c in args.configs.split(",")}
    plans = {k: v for k, v in plans.items() if any(k.startswith(w) for w in wanted)}
    rows = load_eval_rows(cfg, args.max_samples or None, seed=0)
    from smallprint.model import HFClauseModel

    summary: dict = {"model": repo, "high_fraction": hf, "n_pairs": len(pairs), "versions": {}}
    for name, plan in plans.items():
        torch_dtype = getattr(torch, args.dtype)
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch_dtype).to(device)
        tier_plan = None
        if plan is not None:
            circ, frac = plan
            # quanto backend: 4-bit low tier, protected layers left in bf16; CircuitKIT freezes
            # (packs) the weights itself, so memory actually shrinks
            tier_plan = ck.quantize(model, circ, high_fraction=frac, backend="quanto")
        size = weight_bytes(model)
        m = HFClauseModel.from_loaded(model, tok, f"{name}:{repo}", batch_size=cfg["eval"]["batch_size"],
                                      max_new_tokens=cfg["eval"]["max_new_tokens"])
        res = {"weights": size, "plan": tier_plan}
        for set_name, set_rows in rows.items():
            s, _ = run_eval(m, set_rows, f"{name}__{set_name}", out, source=set_name, model_label=name)
            res[set_name] = {k: s[k] for k, _ in HEADLINE if k in s}
            print(f"{name} on {set_name}: unfair_f1={s['unfair_f1']:.3f} macro_f1={s['category_macro_f1']:.3f} "
                  f"weights={size['total_gb']:.2f} GB", flush=True)
        res["clauses_per_second"] = m.stats["clauses"] / m.stats["seconds"] if m.stats["seconds"] else None
        summary["versions"][name] = res
        if args.push and name.startswith("int4_circuit"):
            ck.export_checkpoint(model, None, str(out / "ckpt_circuit"), intervention="quantization",
                                 push_to_hub=True, hub_repo=f"{cfg['hub']['model_repo']}-int4-circuit",
                                 hub_private=True)
        del m, model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    md = summary_md(summary)
    (out / "summary.md").write_text(md)
    print(md)


def summary_md(summary: dict) -> str:
    vs = summary["versions"]
    sets = [k for k in next(iter(vs.values())) if k not in ("weights", "plan", "clauses_per_second")] if vs else []
    lines = [f"# Quantization with CircuitKIT ({summary['model']})", "",
             f"Circuit found with EAP-IG on {summary['n_pairs']} fair/unfair label-swap pairs; "
             f"`circuit` keeps the top {summary['high_fraction']:.0%} of layers in bf16, `random` keeps the same "
             "number chosen at random, `uniform` keeps none.", "",
             "| Version | Weights (GB) | " + " | ".join(f"{s} unfair F1 | {s} macro-F1" for s in sets) + " | clauses/s |",
             "|---|---:|" + "---:|---:|" * len(sets) + "---:|"]
    for name, r in vs.items():
        cells = [f"{r['weights']['total_gb']:.2f}"]
        for s in sets:
            cells += [f"{r[s].get('unfair_f1', float('nan')):.3f}", f"{r[s].get('category_macro_f1', float('nan')):.3f}"]
        cps = r.get("clauses_per_second")
        cells.append(f"{cps:.1f}" if cps else "–")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
