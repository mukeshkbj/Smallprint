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
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smallprint.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    for flag in ("data", "results", "sft", "final"):
        ap.add_argument(f"--{flag}", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    hub, work = cfg["hub"], Path(cfg["data"]["work_dir"])
    from huggingface_hub import snapshot_download

    if args.data:
        snapshot_download(hub["dataset_repo"], repo_type="dataset", local_dir=work / "smallprint-sft")
        snapshot_download(hub["eval_dataset_repo"], repo_type="dataset", local_dir=work / "eval",
                          allow_patterns=["*.jsonl"])
        print("restored data ->", work / "smallprint-sft", "and", work / "eval")
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
            for f in (tmp / "results").iterdir():
                shutil.copy2(f, out / f.name)
            print("restored results ->", out)
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
