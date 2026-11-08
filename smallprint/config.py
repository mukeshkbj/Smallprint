"""Load configs/smallprint.yaml with environment-variable overrides.

Any key can be overridden as SMALLPRINT_<SECTION>__<KEY>, e.g.
SMALLPRINT_HUB__USER=yourname or SMALLPRINT_MODEL__BASE=CohereLabs/tiny-aya-global.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "smallprint.yaml"


def _coerce(v: str):
    low = v.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none", ""}:
        return None
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def load_config(path: str | Path | None = None) -> dict:
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    for k, v in os.environ.items():
        if not k.startswith("SMALLPRINT_") or "__" not in k:
            continue
        section, key = k[len("SMALLPRINT_"):].lower().split("__", 1)
        cfg.setdefault(section, {})[key] = _coerce(v)
    hub = cfg.get("hub", {})
    user = hub.get("user")
    for name in ("model_repo", "dpo_model_repo", "sft_adapter_repo", "dataset_repo", "eval_dataset_repo", "dpo_dataset_repo", "space_repo"):
        if user and hub.get(name) and "/" not in hub[name]:
            hub[name] = f"{user}/{hub[name]}"
    return cfg
