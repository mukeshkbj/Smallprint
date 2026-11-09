#!/usr/bin/env python
"""Put earlier steps' outputs back on disk from the Hub, so a failed HF Job can be
resumed in a fresh container (scripts/hf_job.py --from-step N adds this step itself).

  --data     training data (dataset_repo) -> data/smallprint-sft, test sets -> data/eval
  --results  saved results (eval_dataset_repo results/) -> results/, e.g. the baseline
  --sft      SFT adapter (sft_adapter_repo) -> runs/sft/adapter, then merged -> runs/sft/merged
  --final    final model (model_repo) -> runs/dpo/final
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smallprint.config import load_config  # noqa: E402
from smallprint.data import Clause, read_jsonl, write_jsonl  # noqa: E402


def rebuild_interim(sft_dir: Path, interim: Path) -> tuple[int, int]:
    """Recreate the files step 3 (DPO) reads from the published SFT data: every SFT
    example is a clause (user turn) with its gold JSON (assistant turn), so the
    clause list and the evidence/explanation per unfair clause can be read back.
    Language and source aren't in the export; DPO only uses them as metadata."""
    clauses, ann = [], []
    for f in sorted(sft_dir.glob("*/sft_messages.jsonl")):
        for r in read_jsonl(f):
            msgs = {m["role"]: m["content"] for m in r["messages"]}
            text, gold = msgs["user"], json.loads(msgs["assistant"])
            cid = "restored-" + hashlib.sha1(text.encode()).hexdigest()[:12]
            clauses.append(Clause(id=cid, text=text, lang="unknown", source="restored", split="train",
                                  unfair=bool(gold["unfair"]), categories=list(gold["categories"])).to_dict())
            if gold["unfair"]:
                ann.append({"clause_id": cid, "evidence": gold["evidence"], "explanation": gold["explanation"],
                            "annotation_source": "restored"})
    write_jsonl(interim / "train_clauses.jsonl", clauses)
    write_jsonl(interim / "final_annotations.jsonl", ann)
    return len(clauses), len(ann)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    for flag in ("data", "results", "sft", "final"):
        ap.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--results-from", action="append", default=[], metavar="RUN@REVISION",
                    help="take one run's saved results from an earlier commit of the eval repo, "
                         "e.g. sft@b1451b8 (repeatable; applied after --results)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    hub, work = cfg["hub"], Path(cfg["data"]["work_dir"])
    from huggingface_hub import snapshot_download

    if args.data:
        sft_dir = work / "smallprint-sft"
        snapshot_download(hub["dataset_repo"], repo_type="dataset", local_dir=sft_dir)
        snapshot_download(hub["eval_dataset_repo"], repo_type="dataset", local_dir=work / "eval",
                          allow_patterns=["*.jsonl"])
        print("restored data ->", sft_dir, "and", work / "eval")
        interim = work / "interim"
        if (sft_dir / "interim" / "train_clauses.jsonl").exists():   # uploaded by 01_build_data --push
            interim.mkdir(parents=True, exist_ok=True)
            for f in (sft_dir / "interim").glob("*.jsonl"):
                shutil.copy2(f, interim / f.name)
            print("restored interim files ->", interim)
        else:
            n, a = rebuild_interim(sft_dir, interim)
            print(f"rebuilt interim files from the SFT data: {n} clauses, {a} annotations")
    if args.results:
        tmp = Path("/tmp/smallprint-results")
        try:
            snapshot_download(hub["eval_dataset_repo"], repo_type="dataset", local_dir=tmp,
                              allow_patterns=["results/*"])
        except Exception as e:  # noqa: BLE001 - nothing saved yet is fine
            print("no saved results:", e)
        if (tmp / "results").exists():
            out = Path(cfg["eval"]["out_dir"])
            out.mkdir(parents=True, exist_ok=True)
            for f in (tmp / "results").iterdir():  # includes safety/ and quantization/ subfolders
                if f.is_dir():
                    shutil.copytree(f, out / f.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(f, out / f.name)
            print("restored results ->", out)
    for spec in args.results_from:
        run, rev = spec.split("@", 1)
        tmp = Path(f"/tmp/smallprint-results-{run}")
        snapshot_download(hub["eval_dataset_repo"], repo_type="dataset", local_dir=tmp, revision=rev,
                          allow_patterns=[f"results/{run}__*"])
        out = Path(cfg["eval"]["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        files = list((tmp / "results").glob(f"{run}__*"))
        if not files:
            raise SystemExit(f"no results/{run}__* at revision {rev}")
        for f in files:
            shutil.copy2(f, out / f.name)
        print(f"restored {len(files)} {run} result files from revision {rev}")
    if args.sft:
        sft_out = Path(cfg["sft"]["output_dir"])
        snapshot_download(hub["sft_adapter_repo"], local_dir=sft_out / "adapter")
        import importlib.util
        spec = importlib.util.spec_from_file_location("sft", Path(__file__).with_name("02_train_sft.py"))
        sft = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sft)
        sft.merge_adapter(cfg["model"]["base"], str(sft_out / "adapter"), str(sft_out / "merged"))
    if args.final:
        snapshot_download(hub["model_repo"], local_dir=Path(cfg["dpo"]["output_dir"]) / "final")
        print("restored final model")


if __name__ == "__main__":
    main()
