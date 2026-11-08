"""Run scripts 01 (data) and 03 (DPO pairs, --no-mine) end to end on synthetic
datasets shaped like the real ones. No network, no GPU, no teacher."""

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

UNFAIR = [
    ("We may terminate your account at any time for any reason {n}.", [1], "ter"),
    ("We reserve the right to change these terms at any time without notice {n}.", [2], "ch"),
    ("All disputes shall be resolved by binding arbitration in Delaware {n}.", [7], "a"),
    ("We are not liable for any indirect or consequential damages {n}.", [0], "ltd"),
]
FAIR = [
    "You can download your data from the settings page {n}.",
    "Our support team replies to emails within two working days {n}.",
    "These terms are available in several languages {n}.",
    "Please keep your password secret and do not share it {n}.",
]


_RNG = random.Random(7)
_VOCAB = [f"tok{i}" for i in range(5000)]


def _fill(text):
    """Append distinct filler so fixture sentences are not near-duplicates of each other."""
    return text + " " + " ".join(_RNG.choice(_VOCAB) for _ in range(12))


def _lex(split, k):
    rows = []
    for i in range(k):
        t, lab, _ = UNFAIR[i % 4]
        rows.append({"text": _fill(t.format(n=f"{split}{i}")), "labels": lab})
        rows.append({"text": _fill(FAIR[i % 4].format(n=f"{split}{i}")), "labels": []})
        rows.append({"text": _fill(FAIR[(i + 1) % 4].format(n=f"{split}x{i}")), "labels": []})
    return rows


def _ml(split, k, companies):
    rows = []
    for comp in companies:
        for lang in ("en", "de", "it", "pl"):
            for i in range(k):
                t, _, code = UNFAIR[i % 4]
                r = {"language": lang, "company": comp, "line_number": i, "sentence": _fill(f"[{lang}] " + t.format(n=f"{comp}{i}")),
                     "unfairness_level": "potentially_unfair", "a": False, "ch": False, "cr": False, "j": False,
                     "law": False, "ltd": False, "ter": False, "use": False, "pinc": False}
                r[code] = True
                rows.append(r)
                rows.append({**{k2: False for k2 in r if len(k2) <= 4 and k2 not in ("language",)},
                             "language": lang, "company": comp, "line_number": 100 + i,
                             "sentence": _fill(f"[{lang}] " + FAIR[i % 4].format(n=f"{comp}{i}")), "unfairness_level": "untagged"})
    return rows


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_data_and_dpo_offline(tmp_path, monkeypatch):
    datasets = pytest.importorskip("datasets")
    pytest.importorskip("curatorkit")

    lex_test = _lex("test", 6)
    lex = datasets.DatasetDict({
        "train": datasets.Dataset.from_list(_lex("train", 30)),
        "validation": datasets.Dataset.from_list(_lex("val", 6)),
        "test": datasets.Dataset.from_list(lex_test),
    })
    ml_train_rows = _ml("train", 6, ["Alpha", "Beta"])
    # "Leaky": its English sentences are copies of LexGLUE test sentences -> whole company must be dropped
    leaky = [{"language": "en", "company": "Leaky", "line_number": i, "sentence": r["text"], "unfairness_level": "untagged"}
             for i, r in enumerate(lex_test)]
    leaky += [{"language": "de", "company": "Leaky", "line_number": i, "sentence": _fill(f"Deutscher Satz Nummer {i} hier."),
               "unfairness_level": "untagged"} for i in range(5)]
    ml = datasets.DatasetDict({
        "train": datasets.Dataset.from_list(ml_train_rows + leaky),
        "validation": datasets.Dataset.from_list(_ml("val", 2, ["Gamma"])),
        "test": datasets.Dataset.from_list(_ml("test", 3, ["Zeta"])),
    })
    monkeypatch.setattr(datasets, "load_dataset", lambda repo, *a, **k: lex if "lex_glue" in repo else ml)
    monkeypatch.setenv("SMALLPRINT_DATA__WORK_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)

    build = _load(ROOT / "scripts" / "01_build_data.py")
    monkeypatch.setattr(sys, "argv", ["01", "--no-teacher"])
    build.main()

    work = tmp_path / "data"
    report = json.loads((work / "build_report.json").read_text())
    assert report["leaky_companies_dropped"] == ["Leaky"]
    assert report["annotation_sources"].get("template", 0) > 0
    train_msgs = (work / "smallprint-sft" / "train" / "sft_messages.jsonl").read_text().splitlines()
    assert len(train_msgs) > 50
    langs = {json.loads(line)["lang"] for line in (work / "interim" / "sft_input.jsonl").read_text().splitlines()}
    assert {"en", "de", "it", "pl"} <= langs
    eval_rows = [json.loads(x) for x in (work / "eval" / "claudette_ml_test.jsonl").read_text().splitlines()]
    assert any(r["gold_unfair"] for r in eval_rows) and any(not r["gold_unfair"] for r in eval_rows)

    dpo = _load(ROOT / "scripts" / "03_build_dpo.py")
    monkeypatch.setattr(sys, "argv", ["03", "--no-mine"])
    dpo.main()
    chat = work / "smallprint-dpo-chat"
    pairs = [json.loads(x) for x in (chat / "train.jsonl").read_text().splitlines()]
    assert pairs and pairs[0]["prompt"][0]["role"] == "system" and pairs[0]["chosen"][0]["role"] == "assistant"
    assert all(p["chosen"][0]["content"] != p["rejected"][0]["content"] for p in pairs)
    kinds = json.loads((chat / "negative_types.json").read_text())
    assert {"fabricated_quote", "under_flag", "over_flag", "wrong_category", "prose_unfair", "prose_fair"} <= set(kinds)
