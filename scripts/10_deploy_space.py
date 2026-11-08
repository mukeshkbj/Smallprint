#!/usr/bin/env python
"""Step 10: create or update the demo Space (Gradio on ZeroGPU) from space/.

The Space loads hub.model_repo. While that repo is private, the Space needs a token
secret to read it (Space settings -> Variables and secrets -> HF_TOKEN, read-only is
enough); once the model is public it needs none. Without either, the Space starts and
says the model couldn't be loaded.

Usage:
  python scripts/10_deploy_space.py              # private Space (default)
  python scripts/10_deploy_space.py --public
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smallprint.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--hardware", default="zero-a10g", help="zero-a10g = ZeroGPU")
    args = ap.parse_args()
    cfg = load_config(args.config)
    hub = cfg["hub"]
    repo = hub["space_repo"]
    if "/" not in repo:
        raise SystemExit("Set hub.user (or SMALLPRINT_HUB__USER) first")

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="space", space_sdk="gradio", space_hardware=args.hardware,
                    private=not args.public, exist_ok=True)
    api.add_space_variable(repo, "SMALLPRINT_MODEL", hub["model_repo"])
    api.upload_folder(folder_path=str(ROOT / "space"), repo_id=repo, repo_type="space",
                      commit_message="Deploy Smallprint demo")
    info = api.get_space_runtime(repo)
    print("space:", f"https://huggingface.co/spaces/{repo}", "| hardware:", info.hardware,
          "| requested:", info.requested_hardware, "| stage:", info.stage)


if __name__ == "__main__":
    main()
