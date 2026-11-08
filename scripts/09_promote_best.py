#!/usr/bin/env python
"""Step 9: publish the better of SFT and SFT+DPO as hub.model_repo, chosen by AuditKIT scores.

Score = mean of unfair-term F1 and category macro-F1 over both test sets. DPO only replaces SFT if it
scores higher; the decision and the numbers go into the model card and results/promotion.json.

Usage:
  python scripts/09_promote_best.py --dry-run     # print the decision only
  python scripts/09_promote_best.py               # upload the winner
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smallprint.config import load_config  # noqa: E402

SETS = ("lexglue", "claudette_ml")
KEYS = ("unfair_f1", "category_macro_f1")


def score(out: Path, run: str) -> dict | None:
    parts = {}
    for s in SETS:
        f = out / f"{run}__{s}.json"
        if not f.exists():
            return None
        m = json.loads(f.read_text())
        parts.update({f"{s}.{k}": m[k] for k in KEYS})
    parts["score"] = sum(parts.values()) / len(parts)
    return parts


def decide(sft: dict | None, final: dict | None) -> str:
    if final and (not sft or final["score"] > sft["score"]):
        return "final"
    return "sft"


def results_md(name: str, s: dict, other: str, o: dict | None) -> str:
    rows = ["| | " + name + (" | " + other if o else "") + " |", "|---|---:" + ("|---:" if o else "") + "|"]
    for k in [f"{st}.{m}" for st in SETS for m in KEYS] + ["score"]:
        rows.append(f"| {k} | {s[k]:.3f}" + (f" | {o[k]:.3f}" if o else "") + " |")
    return "AuditKIT results (natural class mix, same prompt for every model):\n\n" + "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out, hub = Path(cfg["eval"]["out_dir"]), cfg["hub"]
    s_sft, s_final = score(out, "sft"), score(out, "final")
    winner = decide(s_sft, s_final)
    print("sft  :", s_sft)
    print("final:", s_final)
    print("winner:", winner)
    (out / "promotion.json").write_text(json.dumps({"winner": winner, "sft": s_sft, "final": s_final}, indent=2))
    if args.dry_run:
        return

    src = Path(cfg["dpo"]["output_dir"]) / "final" if winner == "final" else Path(cfg["sft"]["output_dir"]) / "merged"
    if not (src / "config.json").exists():
        raise SystemExit(f"{src} has no merged model")
    stage = Path("/tmp/smallprint-promote")
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(src, stage)
    from smallprint.hub import model_card, safe_push
    recipe = "LoRA SFT then DPO" if winner == "final" else "LoRA SFT"
    names = {"final": "SFT + DPO", "sft": "SFT"}
    loser = "sft" if winner == "final" else "final"
    (stage / "README.md").write_text(model_card(
        cfg["model"]["base"], hub["dataset_repo"], recipe,
        results_md(names[winner], s_sft if winner == "sft" else s_final, names[loser],
                   s_final if winner == "sft" else s_sft)))

    from huggingface_hub import HfApi
    api = HfApi()

    def push():
        api.create_repo(hub["model_repo"], private=hub["private"], exist_ok=True)
        api.upload_folder(folder_path=str(stage), repo_id=hub["model_repo"],
                          commit_message=f"Publish {names[winner]} (best AuditKIT score)")
    if not safe_push(hub["model_repo"], push):
        raise SystemExit("upload failed")
    print("pushed", f"https://huggingface.co/{hub['model_repo']}", f"({names[winner]})")


if __name__ == "__main__":
    main()
