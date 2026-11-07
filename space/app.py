"""Smallprint demo: paste a Terms of Service or privacy policy, get the clauses
that may be unfair, each with an exact quote and a one-line explanation."""

from __future__ import annotations

import os

import spaces  # no-op outside ZeroGPU
import torch

import gradio as gr

from smallprint.document import build_report, split_clauses
from smallprint.model import HFClauseModel
from smallprint.schema import parse_output

MODEL_ID = os.environ.get("SMALLPRINT_MODEL", "mukeshkbj/smallprint-tiny-aya-water-3b")
MAX_CLAUSES = int(os.environ.get("SMALLPRINT_MAX_CLAUSES", "250"))
MAX_GPU_SECONDS = int(os.environ.get("SMALLPRINT_MAX_GPU_SECONDS", "120"))

# Load once at start-up. On ZeroGPU `.to("cuda")` here only registers the weights;
# the GPU is attached inside the @spaces.GPU call. bf16 is native on ZeroGPU hardware.
_device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = HFClauseModel(MODEL_ID, batch_size=16, device_map=None, token=os.environ.get("HF_TOKEN"),
                      dtype=torch.bfloat16 if _device == "cuda" else torch.float32)
MODEL.model.to(_device)


def gpu_seconds(text: str) -> int:
    """Ask ZeroGPU for roughly what this document needs (about 0.4 s per clause), so
    short policies don't reserve, or get refused for, a full two minutes of quota."""
    n = min(len(split_clauses(text or "")), MAX_CLAUSES)
    return max(15, min(MAX_GPU_SECONDS, 15 + int(0.4 * n)))


EXAMPLE = """Welcome to Example Cloud. These Terms apply to your use of our storage service.

1. Agreement. By accessing or using the Service you agree to be bound by these Terms, whether or not you have read them.

2. Changes. We may modify these Terms or any part of the Service at any time, at our sole discretion, without notice to you.

3. Your content. You keep ownership of the files you upload. We may remove or delete any content at any time for any reason or no reason.

4. Account. You can close your account from the settings page whenever you like. We may suspend or terminate your account at any time without prior notice.

5. Liability. To the maximum extent permitted by law, Example Cloud shall not be liable for any loss of data or any indirect or consequential damages.

6. Disputes. Any dispute arising from these Terms shall be resolved exclusively by binding arbitration in Delaware, and you waive your right to a jury trial.

7. Support. You can contact our support team by email and we aim to reply within two working days."""


@spaces.GPU(duration=gpu_seconds)
def analyse(text: str):
    clauses = split_clauses(text or "")[:MAX_CLAUSES]
    if not clauses:
        return "Paste some terms or a privacy policy first.", {}
    raws = MODEL(clauses)
    report = build_report(clauses, [parse_output(r) for r in raws])
    return report.to_markdown(), report.to_dict()


with gr.Blocks(title="Smallprint") as demo:
    gr.Markdown(
        "# Smallprint\n"
        "Reads the small print, on your device. A 3.35B Tiny Aya model fine-tuned to flag potentially unfair "
        "terms in Terms of Service and privacy policies (English, German, Italian, Polish). "
        "Every flag comes with an exact quote from the text. **Not legal advice.**"
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Terms or privacy policy", lines=18, value=EXAMPLE)
            btn = gr.Button("Read the small print", variant="primary")
        with gr.Column():
            out_md = gr.Markdown()
            with gr.Accordion("JSON", open=False):
                out_json = gr.JSON()
    btn.click(analyse, inp, [out_md, out_json])

if __name__ == "__main__":
    demo.launch()
