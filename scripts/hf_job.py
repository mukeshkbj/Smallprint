#!/usr/bin/env python
"""Run the full Smallprint pipeline as a Hugging Face Job (billed to your HF credits).

One job runs every step on a single GPU and pushes the datasets, the SFT adapter,
the final model and the results/ folder to the Hub. Nothing is lost when the job
ends.

Default: an A100 80 GB ($2.50/h at the time of writing; check `hf jobs hardware`).
With native bf16 there is no need for 4-bit loading, and batches are larger.

Usage (needs HF_TOKEN with write access in the environment):
  python scripts/hf_job.py --dry-run             # print the command, launch nothing
  python scripts/hf_job.py                       # launch; prints the job URL
  python scripts/hf_job.py --flavor l40sx1 --timeout 6h
  python scripts/hf_job.py --logs <job_id>       # stream logs of a running job
"""

from __future__ import annotations

import argparse
import os
import shlex

REPO = "https://github.com/mukeshkbj/smallprint.git"
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"

# Settings for a GPU with native bf16 and >= 40 GB (A100 / L40S). Overrides configs/smallprint.yaml.
GPU_ENV = {
    "SMALLPRINT_SFT__LOAD_IN_4BIT": "false",
    "SMALLPRINT_SFT__BATCH_SIZE": "4",
    "SMALLPRINT_SFT__GRAD_ACCUM": "4",
    "SMALLPRINT_DPO__LOAD_IN_4BIT": "false",
    "SMALLPRINT_DPO__BATCH_SIZE": "2",
    "SMALLPRINT_DPO__GRAD_ACCUM": "8",
    "SMALLPRINT_EVAL__BATCH_SIZE": "32",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONUNBUFFERED": "1",
}

STEPS = [
    "python scripts/00_check_providers.py --n 2",
    "python scripts/01_build_data.py --push",
    "python scripts/05_evaluate.py --skip-final",
    "python scripts/02_train_sft.py --push",
    "python scripts/03_build_dpo.py",
    "python scripts/04_train_dpo.py --push",
    "python scripts/05_evaluate.py --with-sft --reuse-base",
    "python scripts/07_push_results.py",
]


def build_command(branch: str, steps: list[str]) -> list[str]:
    script = " && ".join([
        "set -euo pipefail",
        "apt-get update -qq && apt-get install -y -qq git >/dev/null",
        f"git clone -q --branch {shlex.quote(branch)} {REPO} /work",
        "cd /work",
        'pip install -q -e ".[data,train,eval]"',
        "nvidia-smi --query-gpu=name,memory.total --format=csv",
        *[f'echo "=== {s} ===" && {s}' for s in steps],
        'echo "=== done ==="',
    ])
    return ["bash", "-lc", script]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flavor", default="a100-large")
    ap.add_argument("--timeout", default="5h")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--hf-user", default=os.environ.get("SMALLPRINT_HUB__USER", "mukeshkbj"))
    ap.add_argument("--from-step", type=int, default=0, help="skip the first N steps (resume)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--logs", metavar="JOB_ID")
    args = ap.parse_args()

    if args.logs:
        from huggingface_hub import fetch_job_logs
        for line in fetch_job_logs(job_id=args.logs):
            print(line, flush=True)
        return

    env = {**GPU_ENV, "SMALLPRINT_HUB__USER": args.hf_user}
    cmd = build_command(args.branch, STEPS[args.from_step:])
    if args.dry_run:
        print("flavor:", args.flavor, "timeout:", args.timeout)
        print("env:", env)
        print(cmd[-1].replace(" && ", "\n  && "))
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set (needs write access: the job pushes datasets and models)")
    from huggingface_hub import run_job
    job = run_job(image=IMAGE, command=cmd, env=env, secrets={"HF_TOKEN": token},
                  flavor=args.flavor, timeout=args.timeout, name="smallprint-full-run")
    print("job id:", job.id)
    print("follow it at:", getattr(job, "url", None) or f"https://huggingface.co/jobs/{args.hf_user}/{job.id}")
    print("logs:   python scripts/hf_job.py --logs", job.id)


if __name__ == "__main__":
    main()
