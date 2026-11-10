#!/usr/bin/env python
"""Step 10: create or update the demo Space (Gradio on ZeroGPU) from space/.

The Space loads hub.model_repo. While that repo is private, the Space needs a token
secret to read it (Space settings -> Variables and secrets -> HF_TOKEN, read-only is
enough); once the model is public it needs none. Without either, the Space starts and
says the model couldn't be loaded.

Usage:
  python scripts/10_deploy_space.py              # private Space (default)
  python scripts/10_deploy_space.py --public
  python scripts/10_deploy_space.py --public --hardware t4-small --sleep-time 900   # without PRO
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smallprint.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--hardware", default="zero-a10g", help="zero-a10g = ZeroGPU (needs PRO), t4-small, l4x1, cpu-basic")
    ap.add_argument("--sleep-time", type=int, help="seconds idle before a paid Space sleeps (e.g. 900)")
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
    runtime = api.get_space_runtime(repo)
    if runtime.requested_hardware != args.hardware or args.sleep_time:
        api.request_space_hardware(repo, hardware=args.hardware, sleep_time=args.sleep_time)
    # Pin the code to this commit, so every deploy changes requirements.txt and the Space
    # rebuilds with exactly the code that was tested (a bare git URL would stay cached).
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    stage = Path(tempfile.mkdtemp()) / "space"
    shutil.copytree(ROOT / "space", stage)
    req = stage / "requirements.txt"
    pkg = "smallprint @ git+https://github.com/mukeshkbj/smallprint"
    req.write_text(req.read_text().replace(pkg, f"{pkg}@{sha}" if sha else pkg))
    api.upload_folder(folder_path=str(stage), repo_id=repo, repo_type="space",
                      commit_message=f"Deploy Smallprint demo ({sha[:7] or 'unpinned'})")
    info = api.get_space_runtime(repo)
    print("space:", f"https://huggingface.co/spaces/{repo}", "| hardware:", info.hardware,
          "| requested:", info.requested_hardware, "| stage:", info.stage)


if __name__ == "__main__":
    main()
