#!/usr/bin/env python
"""Step 4: DPO on top of the merged SFT model with AlignTune, then merge and
publish the final model (the one the Space and the on-device build use).

Usage:
  python scripts/04_train_dpo.py
  python scripts/04_train_dpo.py --max-steps 20     # smoke test
  python scripts/04_train_dpo.py --push             # upload the final merged model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smallprint.config import load_config  # noqa: E402

MODEL_CARD = """---
license: cc-by-nc-4.0
base_model: {base}
language: [en, de, it, pl]
tags: [legal, consumer-protection, terms-of-service, privacy-policy, json, lexsi-hackathon]
datasets: [{dataset}]
---
# Smallprint (Tiny Aya Water, 3.35B)

Reads one clause of a Terms of Service or privacy policy and returns JSON:
`{{"unfair": bool, "categories": [...], "evidence": "<exact quote>", "explanation": "<one plain sentence>"}}`.

Fine-tuned from `{base}` with QLoRA SFT then DPO (AlignTune), on expert-labelled clauses from
LexGLUE UNFAIR-ToS and the multilingual CLAUDETTE corpus, curated with CuratorKIT and evaluated with AuditKIT.
Use the system prompt in `smallprint/prompts.py` of the code repo: {code}

Results: see the evaluation table in the code repo README.

**Not legal advice.** Licence: CC BY-NC 4.0, inherited from Tiny Aya (non-commercial; follow Cohere Labs' Acceptable Use Policy).
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dp, m = cfg["dpo"], cfg["model"]
    sft_merged = str(Path(cfg["sft"]["output_dir"]) / "merged")
    out = Path(dp["output_dir"])

    import importlib.util
    spec = importlib.util.spec_from_file_location("sft", Path(__file__).with_name("02_train_sft.py"))
    sft = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sft)

    from aligntune.core.backend_factory import create_rl_trainer

    trainer = create_rl_trainer(
        model_name=sft_merged,
        dataset_name=str(Path(cfg["data"]["work_dir"]) / "smallprint-dpo-chat"),
        split="train",
        algorithm="dpo",
        backend="trl",
        output_dir=str(out),
        num_epochs=dp["epochs"],
        max_steps=args.max_steps,
        batch_size=dp["batch_size"],
        gradient_accumulation_steps=dp["grad_accum"],
        learning_rate=dp["learning_rate"],
        max_seq_length=m["max_seq_length"],
        max_prompt_length=m["max_seq_length"] - 192,
        beta=dp["beta"],
        use_peft=True,
        lora_r=dp["lora_r"],
        lora_alpha=dp["lora_alpha"],
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        quantization={"load_in_4bit": True, "bnb_4bit_compute_dtype": sft.compute_dtype(),
                      "bnb_4bit_quant_type": "nf4"} if dp.get("load_in_4bit", True) else {},
        precision="bf16" if sft.compute_dtype() == "bfloat16" else "fp32",  # no native bf16 (T4): skip AMP; 4-bit layers still compute in fp16
        logging_steps=10,
        seed=cfg["seed"],
    )
    trainer.train()
    adapter_dir = trainer.save_model(str(out / "adapter")) or str(out / "adapter")
    del trainer
    import gc

    import torch
    gc.collect()
    torch.cuda.empty_cache()

    final_dir = out / "final"
    sft.merge_adapter(sft_merged, adapter_dir, str(final_dir))
    hub = cfg["hub"]
    (final_dir / "README.md").write_text(MODEL_CARD.format(
        base=m["base"], dataset=hub["dataset_repo"], code="https://github.com/mukeshkbj/smallprint"))

    if args.push:
        if "/" not in hub["model_repo"]:
            raise SystemExit("Set hub.user (or SMALLPRINT_HUB__USER) before --push")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(hub["model_repo"], private=hub["private"], exist_ok=True)
        api.upload_folder(folder_path=str(final_dir), repo_id=hub["model_repo"])
        print("pushed", f"https://huggingface.co/{hub['model_repo']}")


if __name__ == "__main__":
    main()
