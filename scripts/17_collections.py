#!/usr/bin/env python
"""Step 17: the public Hugging Face collections the Lexsi submission asks for.

Makes every Smallprint model, adapter and dataset repo public, then creates (or updates) two public
collections: models + adapters + the demo Space, and datasets (training data, test sets, results).

Usage:
  python scripts/17_collections.py --dry-run
  python scripts/17_collections.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smallprint.config import load_config  # noqa: E402


def items(user: str) -> tuple[list, list]:
    m = f"{user}/smallprint-tiny-aya"
    models = [
        (f"{m}-water-3b", "model", "Smallprint, published model: injection-hardened SFT on Tiny Aya Water (tag v1-sft = plain SFT)"),
        (f"{m}-water-3b-hardened2-sft-lora", "model", "LoRA adapter of the published model (balanced injection hardening, v2)"),
        (f"{m}-water-3b-sft-lora", "model", "LoRA adapter, plain SFT (v1-sft)"),
        (f"{m}-water-3b-hardened-sft-lora", "model", "LoRA adapter, hardening v1 ('opposite'): learned a shortcut, kept for the record"),
        (f"{m}-water-3b-dpo", "model", "SFT + DPO v2: scored below SFT, kept for comparison"),
        (f"{m}-global-3b-sft-lora", "model", "LoRA adapter, same SFT recipe on Tiny Aya Global (variant comparison)"),
        (f"{user}/smallprint-demo", "space", "Demo: paste a policy, get flagged clauses with exact quotes"),
    ]
    datasets = [
        (f"{user}/smallprint-clauses-4lang", "dataset", "Training data: expert-labelled clauses (EN/DE/IT/PL) with grounded evidence and explanations"),
        (f"{user}/smallprint-eval", "dataset", "Test sets, predictions, AuditKIT results, safety and injection suites, quantization"),
    ]
    return models, datasets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    user = cfg["hub"]["user"]
    if not user:
        raise SystemExit("Set hub.user (or SMALLPRINT_HUB__USER) first")
    models, datasets = items(user)

    from huggingface_hub import HfApi
    api = HfApi()
    for repo, kind, _ in models + datasets:
        if kind == "space":
            continue
        info = api.repo_info(repo, repo_type=kind)
        if info.private:
            print(("would make public: " if args.dry_run else "making public: ") + repo)
            if not args.dry_run:
                api.update_repo_settings(repo, private=False, gated=False, repo_type=kind)
    if args.dry_run:
        return

    existing = {c.title: c for c in api.list_collections(owner=user)}
    for title, desc, entries in (
            ("Smallprint models", "Tiny Aya fine-tunes that flag unfair terms in Terms of Service and privacy "
                                  "policies, with every adapter and comparison model (Lexsi hackathon).", models),
            ("Smallprint datasets", "Training data, test sets and every evaluation result behind Smallprint "
                                    "(Lexsi hackathon).", datasets)):
        col = existing.get(title) or api.create_collection(title, namespace=user, description=desc, private=False)
        have = {(i.item_id, i.item_type) for i in api.get_collection(col.slug).items}
        for repo, kind, note in entries:
            if (repo, kind) not in have:
                api.add_collection_item(col.slug, item_id=repo, item_type=kind, note=note)
        api.update_collection_metadata(col.slug, private=False)
        print(title, "->", f"https://huggingface.co/collections/{col.slug}")


if __name__ == "__main__":
    main()
