#!/usr/bin/env python
"""Step 15: build the injection-hardened SFT dataset (see smallprint/harden.py).

Copies the SFT dataset folder and appends augmented copies of training clauses to the train split:
an injected sentence after the clause, gold answer unchanged. The validation split is left alone.

Usage:
  python scripts/15_harden_data.py                       # balanced (v2): data/smallprint-sft -> ...-hardened
  python scripts/15_harden_data.py --mode opposite --n-unfair 1000 --n-fair 500   # v1, learns a shortcut
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smallprint.config import load_config  # noqa: E402
from smallprint.data import read_jsonl, write_jsonl  # noqa: E402
from smallprint.harden import augment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--src")
    ap.add_argument("--out")
    ap.add_argument("--mode", choices=["balanced", "opposite"], default="balanced")
    ap.add_argument("--n-unfair", type=int, default=700, help="unfair clauses copied with an appended sentence")
    ap.add_argument("--n-fair", type=int, default=1400, help="fair clauses copied with an appended sentence")
    args = ap.parse_args()
    cfg = load_config(args.config)
    work = Path(cfg["data"]["work_dir"])
    src = Path(args.src or work / "smallprint-sft")
    out = Path(args.out or work / "smallprint-sft-hardened")
    train = src / "train" / "sft_messages.jsonl"
    if not train.exists():
        raise SystemExit(f"{train} not found: run scripts/08_restore_from_hub.py --data first")

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out)
    rows = list(read_jsonl(train))
    extra = augment(rows, args.n_unfair, args.n_fair, seed=cfg["seed"], mode=args.mode)
    write_jsonl(out / "train" / "sft_messages.jsonl",
                rows + [{"messages": r["messages"]} for r in extra])  # same schema as the original rows

    report = {"mode": args.mode, "original_train_rows": len(rows), "augmented_rows": len(extra),
              "by_bank_and_label": dict(Counter(f"{r['augmentation']['bank']}|unfair="
                                                 f"{json.loads(r['messages'][-1]['content'])['unfair']}" for r in extra)),
              "by_language": dict(Counter(r["augmentation"]["lang"] for r in extra)),
              "by_injection": dict(Counter(r["augmentation"]["injection"] for r in extra))}
    (out / "hardening_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "by_injection"}, indent=2))
    print("hardened SFT data ->", out)


if __name__ == "__main__":
    main()
