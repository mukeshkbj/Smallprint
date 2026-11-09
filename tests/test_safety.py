"""Offline tests for the safety evaluation (no model, no network)."""
import importlib.util
from pathlib import Path

from smallprint.safety import (ATTACKS, INJECTIONS, injection_cases, injection_metrics, keyword_label,
                               parse_judge, refusal_metrics)

ROWS = [
    {"id": "a", "text": "We may terminate your account at any time.", "lang": "en", "gold_unfair": True},
    {"id": "b", "text": "Wir können die Bedingungen jederzeit ändern.", "lang": "de", "gold_unfair": True},
    {"id": "c", "text": "You can contact support by email.", "lang": "en", "gold_unfair": False},
]


def test_injection_texts_cover_every_language():
    for variant, texts in INJECTIONS.items():
        assert set(texts) == {"en", "de", "it", "pl"}, variant


def test_injection_cases_use_the_clause_language_and_skip_fair_clauses():
    cases = injection_cases(ROWS)
    assert {c["id"] for c in cases} == {"a", "b"}
    assert len(cases) == 2 * (1 + len(INJECTIONS))
    de = next(c for c in cases if c["id"] == "b" and c["variant"] == "override")
    assert de["text"].endswith(INJECTIONS["override"]["de"])


def test_injection_metrics_counts_flips_only_on_clauses_caught_clean():
    cases = injection_cases(ROWS)
    # a: flagged clean, unflagged under every attack; b: missed clean (can't flip)
    flagged = [c["id"] == "a" and c["variant"] in ("clean", "control") for c in cases]
    met = injection_metrics(cases, flagged, [True] * len(cases))
    assert met["clean_flag_rate"] == 0.5 and met["control.flag_rate"] == 0.5
    for a in ATTACKS:
        assert met[f"{a}.flips"] == 1 and met[f"{a}.attack_success"] == 1.0
    assert met["mean_attack_success"] == 1.0


def test_refusal_labels_and_metrics():
    assert keyword_label("I'm sorry, but I can't help with that.") == "full_refusal"
    assert keyword_label("Here is how to kill a Python process: use kill -9.") == "full_compliance"
    assert parse_judge(" Full_Refusal\n") == "full_refusal" and parse_judge("unsure") is None
    rows = [{"label": "unsafe"}, {"label": "unsafe"}, {"label": "safe"}]
    met = refusal_metrics(rows, ["full_refusal", "full_compliance", "partial_refusal"])
    assert met["unsafe.refusal_rate"] == 0.5 and met["safe.full_compliance_rate"] == 0.0


def test_redteam_runner_and_summary_with_a_stub_model():
    from auditkit.redteam import RedTeamRunner
    res = RedTeamRunner(model=lambda ps: ["I can't help with that."] * len(ps)).run(
        probes=["jailbreak"], detectors=["keyword", "injection_success"])
    assert 0.0 <= res.pass_rate <= 1.0 and res.results

    spec = importlib.util.spec_from_file_location("s12", Path(__file__).parents[1] / "scripts" / "12_safety_eval.py")
    s12 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s12)
    r = {"model": "x", "redteam": {**{p: 0.5 for p in s12.REDTEAM}, "mean": 0.5},
         "harmful": {"unsafe.refusal_rate": 0.9}, "xstest": {"unsafe.refusal_rate": 0.8, "safe.full_compliance_rate": 0.7},
         "injection": {"clean_flag_rate": 0.9, "control.flag_rate": 0.9, "mean_attack_success": 0.1,
                       **{f"{a}.attack_success": 0.1 for a in ATTACKS}}}
    md = s12.summary_md({"base": r, "smallprint": {"model": "y"}})
    assert "| AdvBench harmful requests refused" in md and "–" in md
