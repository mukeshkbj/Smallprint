import json

from smallprint.data import (
    Clause, balance, companies_overlapping, decontaminate, from_lexglue, from_multilingual, sft_record,
)
from smallprint.document import build_report, split_clauses
from smallprint.grounding import is_verbatim
from smallprint.labels import ALL_KEYS, LEXGLUE_KEYS, normalise_category
from smallprint.metrics import Gold, clause_metrics, comparison_table
from smallprint.prompts import SYSTEM_PROMPT
from smallprint.roi import ClosedAPI, SelfHosted, Workload, monthly, to_markdown
from smallprint.schema import parse_output, target_json
from smallprint.teacher import annotate_one, template_annotation

CLAUSE = "We may terminate your account at any time, without notice, for any reason."


# ---------------------------------------------------------------- labels / prompt
def test_label_sets():
    assert LEXGLUE_KEYS[0] == "limitation_of_liability" and LEXGLUE_KEYS[7] == "arbitration"
    assert len(LEXGLUE_KEYS) == 8 and len(ALL_KEYS) == 9
    assert normalise_category("Choice of law") == "choice_of_law"
    assert normalise_category("ter") == "unilateral_termination"
    assert normalise_category("nonsense") is None
    for k in ALL_KEYS:
        assert k in SYSTEM_PROMPT


# ---------------------------------------------------------------- schema
def test_parse_valid_and_fenced():
    raw = '```json\n{"unfair": true, "categories": ["unilateral_termination"], "evidence": "We may terminate your account", "explanation": "They can close it."}\n```'
    p = parse_output(raw)
    assert p.valid and p.flagged and p.label_set == {"unilateral_termination"}


def test_parse_inconsistent_is_invalid():
    p = parse_output('{"unfair": false, "categories": ["arbitration"], "evidence": "", "explanation": ""}')
    assert p.parsed_json and not p.valid and not p.flagged
    p = parse_output('{"unfair": true, "categories": [], "evidence": "x", "explanation": "y"}')
    assert not p.valid


def test_parse_garbage_and_prefix_text():
    assert not parse_output("I think this is unfair.").parsed_json
    p = parse_output('Sure! {"unfair": false, "categories": [], "evidence": "", "explanation": ""} hope that helps')
    assert p.valid and not p.flagged


def test_target_json_roundtrip():
    t = target_json(True, ["arbitration"], "binding arbitration", "Disputes go to arbitration.")
    assert parse_output(t).valid
    fair = json.loads(target_json(False, ["arbitration"], "x", "y"))
    assert fair == {"unfair": False, "categories": [], "evidence": "", "explanation": ""}


# ---------------------------------------------------------------- grounding
def test_verbatim():
    assert is_verbatim("terminate your account at any time", CLAUSE)
    assert is_verbatim("TERMINATE  your account", CLAUSE)            # case + whitespace
    assert is_verbatim("We may terminate ... for any reason", CLAUSE)  # ellipsis segments, in order
    assert not is_verbatim("for any reason ... We may terminate", CLAUSE)
    assert not is_verbatim("delete your account", CLAUSE)
    assert not is_verbatim("we", CLAUSE)                              # too short to count
    assert is_verbatim("“We may terminate”", 'He said "We may terminate" it')


# ---------------------------------------------------------------- data mapping
def test_from_lexglue():
    cs = from_lexglue([{"text": CLAUSE, "labels": [1]}, {"text": "Hello there friend.", "labels": []}], "train")
    assert cs[0].unfair and cs[0].categories == ["unilateral_termination"]
    assert not cs[1].unfair and cs[1].categories == []


def test_from_multilingual_levels():
    rows = [
        {"language": "de", "company": "Acme", "line_number": 3, "sentence": "Wir koennen Ihr Konto jederzeit kuendigen.",
         "unfairness_level": "potentially_unfair", "ter": True, "a": False},
        {"language": "en", "company": "Acme", "line_number": 4, "sentence": "These terms are governed by the laws of your country.",
         "unfairness_level": "clearly_fair", "law": True},
        {"language": "it", "company": "Acme", "line_number": 5, "sentence": "Benvenuti.", "unfairness_level": "untagged"},
        {"language": "pl", "company": "Acme", "line_number": 6, "sentence": "Spory rozstrzyga arbitraz.", "unfairness_level": 3, "a": 1},
    ]
    cs = from_multilingual(rows, "train")
    assert cs[0].unfair and cs[0].categories == ["unilateral_termination"] and cs[0].lang == "de"
    assert not cs[1].unfair and cs[1].categories == []   # topic present but fair
    assert not cs[2].unfair
    assert cs[3].unfair and cs[3].severity == "clearly_unfair" and cs[3].categories == ["arbitration"]


def test_decontaminate_exact_and_near():
    ref = [Clause("t1", "We may change these terms at any time without notice to you.", "en", "lexglue", "test", True, ["unilateral_change"])]
    train = [
        Clause("a", "We may change these terms at any time without notice to you!", "en", "lexglue", "train", True),
        Clause("b", "We may change these terms at any time without any notice to you.", "en", "lexglue", "train", True),
        Clause("c", "You can download your data from the settings page.", "en", "lexglue", "train", False),
    ]
    kept, dropped = decontaminate(train, ref)
    assert [c.id for c in kept] == ["c"] and {c.id for c in dropped} == {"a", "b"}


def test_companies_overlapping():
    ref = [Clause(f"r{i}", f"Unique clause number {i} about the service and its terms.", "en", "lexglue", "test", False) for i in range(5)]
    ml = [Clause(f"m{i}", f"Unique clause number {i} about the service and its terms.", "en", "claudette_ml", "train", False, company="Leaky") for i in range(3)]
    ml += [Clause(f"n{i}", f"Something totally different {i} goes here in this line.", "en", "claudette_ml", "train", False, company="Clean") for i in range(3)]
    assert companies_overlapping(ml, ref) == {"Leaky"}


def test_balance_keeps_all_unfair():
    cs = [Clause(f"u{i}", "x", "en", "lexglue", "train", True, ["arbitration"]) for i in range(10)]
    cs += [Clause(f"f{i}", "y", "en", "lexglue", "train", False) for i in range(100)]
    out = balance(cs, fair_per_unfair=2.0)
    assert sum(c.unfair for c in out) == 10 and sum(not c.unfair for c in out) == 20


def test_sft_record_shape():
    c = Clause("x", CLAUSE, "en", "lexglue", "train", True, ["unilateral_termination"])
    r = sft_record(c, "terminate your account at any time", "They can close your account.")
    assert r["instruction"] == CLAUSE and r["source_chunk"] == CLAUSE and r["system_prompt"] == SYSTEM_PROMPT
    assert parse_output(r["output"]).valid


# ---------------------------------------------------------------- teacher
def test_teacher_retries_then_accepts():
    c = Clause("x", CLAUSE, "en", "lexglue", "train", True, ["unilateral_termination"])
    replies = iter([
        '{"evidence": "we can delete you", "explanation": "x"}',               # not verbatim -> retry
        '{"evidence": "terminate your account at any time", "explanation": "They can close your account whenever they want."}',
    ])
    a = annotate_one(c, lambda prompt: next(replies))
    assert a["annotation_source"] == "teacher" and a["evidence"] == "terminate your account at any time"


def test_teacher_falls_back_to_template():
    c = Clause("x", CLAUSE, "en", "lexglue", "train", True, ["unilateral_termination"])
    a = annotate_one(c, lambda prompt: "not json")
    assert a == template_annotation(c) and is_verbatim(a["evidence"], CLAUSE)


# ---------------------------------------------------------------- metrics
def _p(unfair, cats=(), ev=""):
    return parse_output(target_json(unfair, list(cats), ev, "e" if unfair else ""))


def test_clause_metrics():
    golds = [
        Gold(CLAUSE, True, frozenset({"unilateral_termination"})),
        Gold("Disputes are settled by binding arbitration.", True, frozenset({"arbitration"})),
        Gold("You can export your data.", False, frozenset()),
        Gold("We use cookies.", False, frozenset()),
    ]
    preds = [
        _p(True, ["unilateral_termination"], "terminate your account"),
        _p(False),                                              # miss
        _p(True, ["content_removal"], "this is not in the text"),  # false alarm, fabricated quote
        parse_output("garbage"),                                # invalid -> fair
    ]
    m = clause_metrics(golds, preds, LEXGLUE_KEYS)
    assert m["json_valid_rate"] == 0.75
    assert m["unfair_precision"] == 0.5 and m["unfair_recall"] == 0.5
    assert m["false_alarm_rate"] == 0.5 and m["miss_rate"] == 0.5
    assert m["evidence_grounding_rate"] == 0.5
    assert m["per_category"]["unilateral_termination"]["f1"] == 1.0
    assert m["per_category"]["arbitration"]["f1"] == 0.0
    assert abs(m["category_macro_f1"] - 0.5) < 1e-9   # macro over categories with support
    assert "Unfair-term F1" in comparison_table({"base": m, "tuned": m})


# ---------------------------------------------------------------- document
def test_split_clauses():
    text = ("1. Terms. By using the service, e.g. the app, you agree to these terms. We may change them at any time.\n\n"
            "2. Disputes. All disputes will be resolved by binding arbitration in San Francisco, CA. Ok.")
    parts = split_clauses(text)
    assert any("e.g. the app" in p for p in parts)               # no split on abbreviation
    assert any(p.startswith("We may change them") for p in parts)
    assert any("binding arbitration" in p for p in parts)
    assert all(len(p.split()) >= 4 for p in parts)
    # section numbers and short headings go with the text that follows, not the clause before
    assert any(p.startswith("Disputes. All disputes") for p in parts)
    assert not any(p.rstrip().endswith(("2. Disputes.", "Disputes.")) for p in parts)


def test_build_report():
    clauses = [CLAUSE, "You can export your data at any time from settings."]
    outs = [_p(True, ["unilateral_termination"], "terminate your account at any time"), _p(False)]
    rep = build_report(clauses, outs)
    assert len(rep.findings) == 1 and rep.findings[0].evidence_verified
    md = rep.to_markdown()
    assert "Unilateral termination" in md and "not legal advice" in md


# ---------------------------------------------------------------- roi
def test_roi():
    w = Workload(1000, 100, 450, 20)
    r = monthly(w, ClosedAPI("X", 1.0, 2.0, "src"), SelfHosted("T4", 0.5, 1000, "src"))
    assert abs(r["closed_api_usd"] - (100000 * 450 / 1e6 * 1 + 100000 * 20 / 1e6 * 2)) < 1e-9
    assert abs(r["self_hosted_gpu_hours"] - (100000 * 20 / 1000 / 3600)) < 1e-9
    assert "on device" in to_markdown(w, ClosedAPI("X", 1, 2, "s"), SelfHosted("T4", 0.5, 1000, "s"), r)


def test_json_complete_stop_signal():
    from smallprint.model import json_complete
    assert json_complete(['{"unfair": false, "categories": [], "evidence": "", "explanation": ""}',
                          '{"unfair": true, "categories": ["arbitration"], "evid',
                          "Sure, here is my analysis of the clause"]) == [True, False, False]
    assert json_complete(['Here you go: {"a": "brace } in string"} and more text']) == [True]


def test_accepted_conversational_matches_flattened_rows():
    from smallprint.curate import accepted_conversational
    chat = [
        {"prompt": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Clause A"}],
         "chosen": [{"role": "assistant", "content": '{"unfair": false}'}],
         "rejected": [{"role": "assistant", "content": "prose"}]},
        {"prompt": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Clause B"}],
         "chosen": [{"role": "assistant", "content": '{"unfair": true}'}],
         "rejected": [{"role": "assistant", "content": '{"unfair": false}'}]},
    ]
    # what AlignTune hands TRL after its CuratorKIT pass: flattened strings, row 2 dropped
    curated = [{"prompt": "SYS\n\nClause A ", "chosen": '{"unfair": false}', "rejected": "prose"}]
    assert accepted_conversational(curated, chat) == [0]
    assert accepted_conversational([], chat) == []


def test_hub_licence_and_safe_push():
    from smallprint.hub import CC_BY_NC_25, safe_push
    assert CC_BY_NC_25.startswith("license: other\n") and "license_name: cc-by-nc-2.5" in CC_BY_NC_25
    calls = []
    assert safe_push("ok", lambda: calls.append(1)) is True

    def bad():
        raise ValueError("Bad request: license must be a licence id")
    assert safe_push("bad", bad, retries=3, wait=0) is False  # validation errors aren't retried


def test_hf_job_resume_restores_skipped_outputs():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("hf_job", Path(__file__).parents[1] / "scripts" / "hf_job.py")
    hj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hj)
    assert hj.restore_step(hj.STEPS) is None
    i_sft = next(i for i, s in enumerate(hj.STEPS) if "02_train_sft" in s)
    assert hj.restore_step(hj.STEPS[i_sft:]) == "python scripts/08_restore_from_hub.py --data --results"
    i_dpo = next(i for i, s in enumerate(hj.STEPS) if "03_build_dpo" in s)
    assert hj.restore_step(hj.STEPS[i_dpo:]) == "python scripts/08_restore_from_hub.py --data --results --sft"
    i_final = next(i for i, s in enumerate(hj.STEPS) if "--with-sft" in s)
    assert hj.restore_step(hj.STEPS[i_final:]).endswith("--data --results --sft --final")


def test_dpo_caps_keep_decision_pairs_at_training_mix():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("b3", Path(__file__).parents[1] / "scripts" / "03_build_dpo.py")
    b3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b3)
    c = b3.caps_for(400, 2.0)
    flag_more = c["under_flag"] + c["prose_unfair"]
    flag_less = c["over_flag"] + c["prose_fair"]
    assert flag_less == 2 * flag_more
    assert c["fabricated_quote"] == c["wrong_category"] == 400


def test_promote_picks_dpo_only_if_it_scores_higher(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("p9", Path(__file__).parents[1] / "scripts" / "09_promote_best.py")
    p9 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p9)
    for run, f1 in (("sft", 0.7), ("final", 0.5)):
        for s in p9.SETS:
            (tmp_path / f"{run}__{s}.json").write_text(json.dumps({"unfair_f1": f1, "category_macro_f1": f1}))
    sft, final = p9.score(tmp_path, "sft"), p9.score(tmp_path, "final")
    assert p9.decide(sft, final) == "sft"
    assert p9.decide({**sft, "score": 0.4}, final) == "final"
    assert p9.decide(sft, None) == "sft"
    assert "| score |" in p9.results_md("SFT", sft, "SFT + DPO", final)
