#!/usr/bin/env python
"""Step 2: QLoRA supervised fine-tuning of Tiny Aya with AlignTune (TRL backend).

Reads the CuratorKIT output folder (config ``sft_messages``), trains a LoRA
adapter, saves it, then merges it into a full-precision copy of the base model
(runs/sft/merged) which step 3 and step 4 start from.

Usage:
  python scripts/02_train_sft.py
  python scripts/02_train_sft.py --max-steps 30     # smoke test
  python scripts/02_train_sft.py --push             # upload the adapter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smallprint.config import load_config  # noqa: E402


def compute_dtype() -> str:
    """bf16 only on GPUs that support it natively (Ampere+). On a T4, PyTorch reports bf16 as
    'supported' via slow emulation, so ask without emulation."""
    from smallprint.model import native_bf16
    return "bfloat16" if native_bf16() else "float16"


def merge_adapter(base_id: str, adapter_dir: str, out_dir: str) -> None:
    """Merge a LoRA adapter into a full-precision copy of the base model."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if compute_dtype() == "bfloat16" else torch.float16
    # On the GPU when there is one: merging a 3.35B model on CPU needs more RAM than a free
    # Colab has (it stalled there). Small shards keep the save's peak memory low too.
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype, device_map=device_map)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True, max_shard_size="2GB")
    AutoTokenizer.from_pretrained(base_id).save_pretrained(out_dir)
    print("merged ->", out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    s, m = cfg["sft"], cfg["model"]
    out = Path(s["output_dir"])

    from aligntune.core.backend_factory import create_sft_trainer

    trainer = create_sft_trainer(
        model_name=m["base"],
        dataset_name=str(Path(cfg["data"]["work_dir"]) / "smallprint-sft"),
        config_name="sft_messages",
        split="train",
        backend=s["backend"],
        output_dir=str(out),
        num_epochs=s["epochs"],
        max_steps=args.max_steps,
        batch_size=s["batch_size"],
        gradient_accumulation_steps=s["grad_accum"],
        learning_rate=s["learning_rate"],
        max_seq_length=m["max_seq_length"],
        use_peft=True,
        lora_r=s["lora_r"],
        lora_alpha=s["lora_alpha"],
        lora_dropout=s["lora_dropout"],
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        quantization={"load_in_4bit": True, "bnb_4bit_compute_dtype": compute_dtype(),
                      "bnb_4bit_quant_type": "nf4"} if s["load_in_4bit"] else {},
        precision="bf16" if compute_dtype() == "bfloat16" else "fp32",  # no native bf16 (T4): skip AMP; 4-bit layers still compute in fp16
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        seed=cfg["seed"],
    )
    trainer.train()
    adapter_dir = trainer.save_model(str(out / "adapter")) or str(out / "adapter")
    print("adapter ->", adapter_dir)

    if args.push:
        repo = cfg["hub"]["sft_adapter_repo"]
        if "/" not in repo:
            raise SystemExit("Set hub.user (or SMALLPRINT_HUB__USER) before --push")
        from smallprint.hub import safe_push
        safe_push(repo, trainer.push_to_hub, repo, private=cfg["hub"]["private"])

    if not args.no_merge:
        del trainer
        import gc

        import torch
        gc.collect()
        torch.cuda.empty_cache()
        merge_adapter(m["base"], adapter_dir, str(out / "merged"))


if __name__ == "__main__":
    main()
