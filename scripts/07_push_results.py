#!/usr/bin/env python
"""Step 7: upload results/ (metrics JSON, summary.md, AuditKIT comparisons, predictions)
to the eval dataset repo under results/, so they survive a Colab session or HF Job.

Usage:
  python scripts/07_push_results.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smallprint.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    repo = cfg["hub"]["eval_dataset_repo"]
    if "/" not in repo:
        raise SystemExit("Set hub.user (or SMALLPRINT_HUB__USER) first")
    out = Path(cfg["eval"]["out_dir"])
    if not out.exists():
        raise SystemExit(f"{out} does not exist: run scripts/05_evaluate.py first")

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=cfg["hub"]["private"], exist_ok=True)
    api.upload_folder(folder_path=str(out), path_in_repo="results", repo_id=repo, repo_type="dataset",
                      commit_message="Upload evaluation results")
    print("pushed", f"https://huggingface.co/datasets/{repo}/tree/main/results")


if __name__ == "__main__":
    main()
