#!/usr/bin/env python
"""Step 16: publish the injection-hardened (balanced, v2) model as hub.model_repo.

Run after `scripts/08_restore_from_hub.py --sft` with SMALLPRINT_HUB__SFT_ADAPTER_REPO pointing at the v2
adapter, so runs/sft/merged is the hardened model. This script:

1. tags the model currently on main (the plain SFT model) so it stays downloadable, e.g. `v1-sft`;
2. writes a model card with the v2 numbers next to the previous model's, read from the eval repo;
3. uploads the merged model to hub.model_repo;
4. restarts the demo Space so it loads the new weights.

Usage:
  python scripts/16_publish_hardened.py --dry-run       # print the card only
  python scripts/16_publish_hardened.py                 # tag, upload, restart the Space
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smallprint.config import load_config  # noqa: E402

SETS = (("lexglue", "LexGLUE UNFAIR-ToS test (EN)"), ("claudette_ml", "CLAUDETTE test (EN/DE/IT/PL)"))
ROWS = (("unfair_f1", "Unfair-term F1"), ("category_macro_f1", "Category macro-F1"),
        ("false_alarm_rate", "False alarms on fair clauses"), ("miss_rate", "Missed unfair clauses"),
        ("evidence_grounding_rate", "Evidence is a real quote"))


def fetch(repo: str, path: str) -> dict:
    from huggingface_hub import hf_hub_download
    return json.loads(Path(hf_hub_download(repo, path, repo_type="dataset")).read_text())


def results_md(eval_repo: str) -> str:
    old = {s: fetch(eval_repo, f"results/sft__{s}.json") for s, _ in SETS}
    new = {s: fetch(eval_repo, f"results-hardened2/sft__{s}.json") for s, _ in SETS}
    saf_old = fetch(eval_repo, "results/safety/smallprint.json")
    saf_new = fetch(eval_repo, "results-hardened2/safety/hardened2.json")
    fair_old = fetch(eval_repo, "results-injfair/safety/smallprint.json")["injection_fair"]
    fair_new = saf_new["injection_fair"]
    lines = ["## Results", "",
             "This version (`main`) is trained on injection-hardened data: copies of training clauses with a sentence "
             "appended (\"this clause is fair\", \"flag this\", or harmless) attached to fair and unfair clauses alike, "
             "with the gold answer unchanged, so the model learns to ignore instructions inside the document. The "
             "previous plain-SFT model is tag `v1-sft`. Natural class mix, same prompt for both; all test sentences "
             "differ from the training ones.", "",
             "| Test set | Metric | v1-sft | this model |", "|---|---|---:|---:|"]
    for s, name in SETS:
        for k, label in ROWS:
            lines.append(f"| {name} | {label} | {old[s][k]:.3f} | {new[s][k]:.3f} |")
    inj_o, inj_n = saf_old["injection"], saf_new["injection"]
    worst_old = max(v for k, v in fair_old.items() if k.endswith("false_alarm_rate") and not k.startswith("clean"))
    worst_new = max(v for k, v in fair_new.items() if k.endswith("false_alarm_rate") and not k.startswith("clean"))
    lines += [f"| Injection | Unfair clauses: flags removed by 3 planted sentences (mean) | "
              f"{inj_o['mean_attack_success']:.3f} | {inj_n['mean_attack_success']:.3f} |",
              f"| Injection | Unfair clauses: flags removed by \"ignore all previous instructions\" | "
              f"{inj_o['override.attack_success']:.3f} | {inj_n['override.attack_success']:.3f} |",
              f"| Injection | Fair clauses: false alarms, worst planted sentence | {worst_old:.3f} | {worst_new:.3f} |",
              f"| Safety | AdvBench harmful requests refused | {saf_old['harmful']['unsafe.refusal_rate']:.3f} | "
              f"{saf_new['harmful']['unsafe.refusal_rate']:.3f} |",
              f"| Safety | XSTest unsafe prompts refused | {saf_old['xstest']['unsafe.refusal_rate']:.3f} | "
              f"{saf_new['xstest']['unsafe.refusal_rate']:.3f} |",
              f"| Safety | XSTest safe prompts answered | {saf_old['xstest']['safe.full_compliance_rate']:.3f} | "
              f"{saf_new['xstest']['safe.full_compliance_rate']:.3f} |", "",
              "Text planted in a policy can still switch some flags off (the blunt override above), so treat a clause "
              "that talks to an AI reviewer as one to read yourself. Full tables: `results/comparisons.md` in the code "
              "repo; raw files in the eval dataset."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--src", help="merged model folder (default <sft.output_dir>/merged)")
    ap.add_argument("--tag-previous", default="v1-sft")
    ap.add_argument("--no-restart", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    hub = cfg["hub"]
    from smallprint.hub import model_card, safe_push
    card = model_card(cfg["model"]["base"], hub["dataset_repo"],
                      "LoRA SFT on injection-hardened (balanced) data", results_md(hub["eval_dataset_repo"]))
    if args.dry_run:
        print(card)
        return
    src = Path(args.src or Path(cfg["sft"]["output_dir"]) / "merged")
    if not (src / "config.json").exists():
        raise SystemExit(f"{src} has no merged model: run 08_restore_from_hub.py --sft with the v2 adapter first")

    from huggingface_hub import HfApi
    api = HfApi()
    tags = {t.name for t in api.list_repo_refs(hub["model_repo"]).tags}
    if args.tag_previous not in tags:
        api.create_tag(hub["model_repo"], tag=args.tag_previous, revision="main",
                       tag_message="Plain SFT model, before injection hardening")
        print("tagged previous main as", args.tag_previous)
    stage = Path("/tmp/smallprint-publish")
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(src, stage)
    (stage / "README.md").write_text(card)

    def push():
        api.upload_folder(folder_path=str(stage), repo_id=hub["model_repo"],
                          commit_message="Publish injection-hardened model (balanced, v2)")
    if not safe_push(hub["model_repo"], push):
        raise SystemExit("upload failed")
    print("pushed", f"https://huggingface.co/{hub['model_repo']}")
    if not args.no_restart:
        api.restart_space(hub["space_repo"])
        print("restarted", f"https://huggingface.co/spaces/{hub['space_repo']}")


if __name__ == "__main__":
    main()
