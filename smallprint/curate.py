"""CuratorKIT stages.

Stage A, the grounding gate (unfair clauses only): CuratorKIT's LLM judge checks each
  teacher explanation against its clause (``source_chunk``), with a
  clause-specific judge prompt. Fair clauses have no explanation, so they skip it.
Stage B, assembly: fair + gated unfair rows go through schema check, exact
  dedup and cleaning, then export to ``sft_messages`` with a validation split.
  The output folder is a Hugging Face dataset that AlignTune loads directly.
DPO: preference pairs go through the same schema check and dedup, then
  export to ``dpo``.

Note: CuratorKIT dedups SFT rows on instruction+output (``input`` is ignored),
which is why the clause lives in ``instruction``. Its MinHash mode collapses
near-identical clauses that share a label, so exact dedup is the default here.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .prompts import GROUNDING_JUDGE_PROMPT

log = logging.getLogger(__name__)


def _curator():
    from curatorkit import Curator, CuratorConfig
    return Curator, CuratorConfig


def grounding_gate(input_jsonl: str | Path, output_dir: str | Path, judge_model: str | None,
                   threshold: float = 0.7, judge_api_base: str | None = None,
                   concurrency: int = 8, judge_api_key: str | None = None) -> tuple[set[str], list[dict]]:
    """Run the hallucination gate. Returns (ids that passed, rejected rows with reasons)."""
    Curator, CuratorConfig = _curator()
    kw = dict(
        dataset=str(input_jsonl), dedup="none", clean=False, min_tokens=3, max_tokens=4096,
        export_formats=["alpaca"], output_dir=str(output_dir), name="smallprint_grounding_gate",
        write_hf_readme=False,
    )
    if judge_model:
        kw.update(
            llm_model=judge_model, llm_api_base=judge_api_base, llm_api_key=judge_api_key,
            judge_llm_model=judge_model, judge_llm_api_base=judge_api_base,
            hallucination_threshold=threshold, hallucination_prompt_template=GROUNDING_JUDGE_PROMPT,
            judge_concurrency=concurrency, judge_on_error="reject",
        )
    else:
        log.warning("No judge model configured: grounding gate skipped (schema check only).")
    res = Curator(CuratorConfig(**kw)).run()
    passed = {s.metadata.get("clause_id") or s.instruction for s in res.passed}
    rejected = [
        {"clause_id": s.metadata.get("clause_id"), "reason": getattr(s, "rejection_reason", ""),
         "step": getattr(s, "rejecting_step", "")}
        for s in res.rejected
    ]
    return passed, rejected


def assemble_sft(input_jsonl: str | Path, output_dir: str | Path, val_fraction: float = 0.05,
                 seed: int = 42):
    Curator, CuratorConfig = _curator()
    res = Curator(CuratorConfig(
        dataset=str(input_jsonl), dedup="exact", clean=True, min_tokens=3, max_tokens=4096,
        export_formats=["messages", "alpaca"],
        output_split={"train": 1 - val_fraction, "validation": val_fraction}, output_split_seed=seed,
        output_dir=str(output_dir), name="smallprint_sft",
    )).run()
    res.print_summary()
    return res


def assemble_dpo(input_jsonl: str | Path, output_dir: str | Path, val_fraction: float = 0.05,
                 seed: int = 42):
    Curator, CuratorConfig = _curator()
    res = Curator(CuratorConfig(
        dataset=str(input_jsonl), dedup="exact", clean=False, min_tokens=1, max_tokens=4096,
        export_formats=["dpo"],
        output_split={"train": 1 - val_fraction, "validation": val_fraction}, output_split_seed=seed,
        output_dir=str(output_dir), name="smallprint_dpo",
    )).run()
    res.print_summary()
    return res
