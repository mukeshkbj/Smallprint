#!/usr/bin/env python
"""Step 11: make the release repos public and restart the demo Space so it loads the model.

Public: hub.model_repo, hub.dataset_repo, hub.eval_dataset_repo, hub.space_repo.
Left as they are: the DPO model, the SFT adapter and the DPO preference data.

Usage:
  python scripts/11_make_public.py --dry-run
  python scripts/11_make_public.py
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
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    hub = load_config(args.config)["hub"]
    targets = [(hub["model_repo"], "model"), (hub["dataset_repo"], "dataset"),
               (hub["eval_dataset_repo"], "dataset"), (hub["space_repo"], "space")]
    from huggingface_hub import HfApi
    api = HfApi()
    for repo, kind in targets:
        before = api.repo_info(repo, repo_type=kind).private
        if not args.dry_run and before:
            api.update_repo_settings(repo, private=False, repo_type=kind)
        after = api.repo_info(repo, repo_type=kind).private
        print(f"{kind:8} {repo}: private {before} -> {after}")
    if not args.dry_run:
        api.restart_space(hub["space_repo"])
        print("restarted", f"https://huggingface.co/spaces/{hub['space_repo']}")


if __name__ == "__main__":
    main()
