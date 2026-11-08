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
    run_eval(perfect, rows, "tuned__lexglue", tmp_path, source="lexglue")  # the script's <model>__<set> naming
    fn, saved = ev.replay(tmp_path, "tuned", "lexglue", rows)
    again, _ = run_eval(fn, rows, "tuned_replay", tmp_path, source="lexglue")
    assert again["unfair_f1"] == tuned["unfair_f1"] == saved["unfair_f1"]
    assert ev.replay(tmp_path, "tuned", "lexglue", rows[:2]) is None   # different sample -> no replay

    # Two replays of different saved runs must not share AuditKIT's disk cache. Both are closures from the
    # same factory, which AuditKIT fingerprints identically; run 3 scored SFT with the baseline's cached
    # result that way. Each evaluate() call now gets its own model identity.
    run_eval(always_fair, rows, "fair__lexglue", tmp_path, source="lexglue")
    f_fair, _ = ev.replay(tmp_path, "fair", "lexglue", rows)
    f_tuned, _ = ev.replay(tmp_path, "tuned", "lexglue", rows)
    a, _ = run_eval(f_fair, rows, "r1", tmp_path, source="lexglue")
    b, _ = run_eval(f_tuned, rows, "r2", tmp_path, source="lexglue")
    assert a["unfair_f1"] == 0.0 and b["unfair_f1"] == 1.0


def test_restore_rebuilds_dpo_inputs_from_published_sft_data(tmp_path):
    """A resumed HF Job rebuilds step 3's inputs from the SFT export alone."""
    import importlib.util
    import random
    from pathlib import Path

    from smallprint.curate import assemble_sft
    from smallprint.data import Clause, read_jsonl, sft_record, write_jsonl

    clauses = [Clause(id=f"c{i}", text=f"Clause {i}: we may close your account {i} times a year without notice.",
                      lang="en", source="lexglue", split="train", unfair=True, categories=["unilateral_termination"])
               for i in range(20)]
    clauses += [Clause(id=f"f{i}", text=f"Fair clause {i}: contact support on weekday {i} for billing help.",
                       lang="en", source="lexglue", split="train", unfair=False) for i in range(40)]
    rows = [sft_record(c, "we may close your account", "They can close it.", "teacher") if c.unfair else sft_record(c)
            for c in clauses]
    write_jsonl(tmp_path / "in.jsonl", rows)
    assemble_sft(tmp_path / "in.jsonl", tmp_path / "sft", 0.1, 0)

    root = Path(__file__).parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location("r8", root / "08_restore_from_hub.py")
    r8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r8)
    n, a = r8.rebuild_interim(tmp_path / "sft", tmp_path / "interim")
    assert n == 60 and a == 20

    spec = importlib.util.spec_from_file_location("b3", root / "03_build_dpo.py")
    b3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b3)
    rebuilt = [Clause.from_dict(r) for r in read_jsonl(tmp_path / "interim" / "train_clauses.jsonl")]
    ann = {x["clause_id"]: x for x in read_jsonl(tmp_path / "interim" / "final_annotations.jsonl")}
    unfair = next(c for c in rebuilt if c.unfair)
    assert '"evidence": "we may close your account"' in b3.gold_answer(unfair, ann)
    kinds = {k for c in rebuilt for k, _ in b3.targeted(c, ann, random.Random(0))}
    assert {"fabricated_quote", "under_flag", "over_flag", "prose_fair"} <= kinds
