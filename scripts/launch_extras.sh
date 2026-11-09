#!/usr/bin/env bash
# Launch the follow-up HF Jobs suggested by the Lexsi team (global, safety, quant; hardened is opt-in). Needs HF_TOKEN in the environment
# (Colab secret) or a prior `hf auth login`. Run from the repo root:  bash scripts/launch_extras.sh [global|safety|quant|hardened ...]
set -euo pipefail
cd "$(dirname "$0")/.."
which=("${@:-global safety quant}")

for job in ${which[@]}; do
  case "$job" in
    global)   # same data and recipe on Tiny Aya Global: zero-shot and SFT scores per language
      python scripts/hf_job.py --flavor a100-large --timeout 3h --name smallprint-global \
        --env SMALLPRINT_MODEL__BASE=CohereLabs/tiny-aya-global \
        --env SMALLPRINT_HUB__SFT_ADAPTER_REPO=smallprint-tiny-aya-global-3b-sft-lora \
        --env SMALLPRINT_EVAL__OUT_DIR=results-global \
        --step "python scripts/08_restore_from_hub.py --data" \
        --step "python scripts/05_evaluate.py --skip-final" \
        --step "python scripts/02_train_sft.py --push" \
        --step "python scripts/05_evaluate.py --with-sft --sft-merged --reuse base --skip-final" \
        --step "python scripts/07_push_results.py" ;;
    safety)   # base vs Smallprint: red team, AdvBench, XSTest, in-clause prompt injection
      python scripts/hf_job.py --flavor a100-large --timeout 2h --name smallprint-safety \
        --step "python scripts/08_restore_from_hub.py --data" \
        --step "python scripts/12_safety_eval.py --models base,smallprint" \
        --step "python scripts/07_push_results.py" ;;
    quant)    # CircuitKIT circuit-guided mixed precision vs uniform 4-bit vs random tiers
      python scripts/hf_job.py --flavor a100-large --timeout 3h --name smallprint-quant \
        --step "python scripts/08_restore_from_hub.py --data" \
        --step "pip install -q -r requirements-circuit.txt" \
        --step "python scripts/13_circuitkit_quantize.py" \
        --step "python scripts/07_push_results.py" ;;
    hardened) # injection-hardened SFT (step 15 data), task scores + full safety suite with the same judge
      python scripts/hf_job.py --flavor a100-large --timeout 3h --name smallprint-hardened \
        --env SMALLPRINT_HUB__SFT_ADAPTER_REPO=smallprint-tiny-aya-water-3b-hardened-sft-lora \
        --env SMALLPRINT_EVAL__OUT_DIR=results-hardened \
        --step "python scripts/08_restore_from_hub.py --data" \
        --step "python scripts/15_harden_data.py" \
        --step "python scripts/02_train_sft.py --dataset-dir data/smallprint-sft-hardened --push" \
        --step "python scripts/05_evaluate.py --skip-base --with-sft --sft-merged --skip-final" \
        --step "python scripts/12_safety_eval.py --models hardened=runs/sft/merged --judge-model openai/deepseek-ai/DeepSeek-V3.2" \
        --step "python scripts/07_push_results.py" ;;
    *) echo "unknown job: $job" >&2; exit 1 ;;
  esac
done
