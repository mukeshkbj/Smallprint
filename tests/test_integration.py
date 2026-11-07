"""Run the real CuratorKIT and AuditKIT code paths with a stub model (no GPU, no network)."""

import json

import pytest

from smallprint.data import Clause, eval_record, sft_record, write_jsonl
from smallprint.schema import target_json

CLAUSES = [
    Clause("u1", "We may terminate your account at any time, without notice, for any reason.", "en", "lexglue", "train", True, ["unilateral_termination"]),
    Clause("u2", "Any dispute shall be resolved exclusively by binding arbitration in San Francisco.", "en", "lexglue", "train", True, ["arbitration"]),
    Clause("f1", "You can download a copy of your data from the account settings page.", "en", "lexglue", "train", False),
    Clause("f2", "Wir verwenden Cookies, um die Nutzung unserer Website zu analysieren.", "de", "claudette_ml", "train", False),
    Clause("f3", "Le presenti condizioni sono disponibili in italiano e in inglese.", "it", "claudette_ml", "train", False),
]


def test_curatorkit_assembly_keeps_distinct_fair_clauses(tmp_path):
    pytest.importorskip("curatorkit")
    from smallprint.curate import assemble_sft

    rows = [sft_record(c, c.text[:30] if c.unfair else "", "x" if c.unfair else "") for c in CLAUSES]
    rows.append(dict(rows[2]))  # exact duplicate must be removed
    src = tmp_path / "sft_input.jsonl"
    write_jsonl(src, rows)
    res = assemble_sft(src, tmp_path / "sft", val_fraction=0.2)
    assert len(res.passed) == len(CLAUSES)   # duplicate dropped, all distinct fair clauses kept
    train = (tmp_path / "sft" / "train" / "sft_messages.jsonl").read_text().splitlines()
    msg = json.loads(train[0])["messages"]
    assert [m["role"] for m in msg] == ["system", "user", "assistant"]
    assert msg[0]["content"].startswith("You are Smallprint")


def test_judge_prompt_passes_curatorkit_validation():
    pytest.importorskip("curatorkit")
    from curatorkit.gates.hallucination import HallucinationGate

    from smallprint.prompts import GROUNDING_JUDGE_PROMPT

    # Constructing the gate runs CuratorKIT's own placeholder validation (no LLM call is made).
    HallucinationGate(llm=object(), threshold=0.7, prompt_template=GROUNDING_JUDGE_PROMPT)
    GROUNDING_JUDGE_PROMPT.format(source_text="c", question="c", answer="{}")


def test_auditkit_eval_with_stub_model(tmp_path):
    pytest.importorskip("auditkit")
    from smallprint.evaluate import compare, run_eval

    rows = [eval_record(c) for c in CLAUSES]
    gold = {c.text: c for c in CLAUSES}

    def perfect(prompts):
        out = []
        for p in prompts:
            c = gold[p]
            out.append(target_json(c.unfair, c.categories, c.text[:25], "x" if c.unfair else ""))
        return out

    def always_fair(prompts):
        return [target_json(False, []) for _ in prompts]

    base, base_r = run_eval(always_fair, rows, "base", tmp_path, source="lexglue", model_label="stub-fair")
    tuned, tuned_r = run_eval(perfect, rows, "tuned", tmp_path, source="lexglue", model_label="stub-perfect")
    assert base["unfair_f1"] == 0.0 and tuned["unfair_f1"] == 1.0
    assert tuned["evidence_grounding_rate"] == 1.0 and tuned["json_valid_rate"] == 1.0
    assert tuned["auditkit_headline"]["json_schema_valid"] == 1.0
    assert set(tuned["per_language"]) == {"en", "de", "it"}
    assert (tmp_path / "tuned.json").exists() and (tmp_path / "tuned.predictions.jsonl").exists()
    assert "unfair_correct" in compare(base_r, tuned_r)

    # Replaying saved predictions re-scores them through AuditKIT with identical metrics (no model needed).
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("ev", Path(__file__).resolve().parents[1] / "scripts" / "05_evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    fn, saved = ev.replay(tmp_path, "tuned", "lexglue", rows)
    again, _ = run_eval(fn, rows, "tuned_replay", tmp_path, source="lexglue")
    assert again["unfair_f1"] == tuned["unfair_f1"] == saved["unfair_f1"]
    assert ev.replay(tmp_path, "tuned", "lexglue", rows[:2]) is None   # different sample -> no replay
