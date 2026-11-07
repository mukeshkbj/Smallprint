#!/usr/bin/env python
"""Step 1: build the training and test data.

  LexGLUE UNFAIR-ToS + multilingual CLAUDETTE
    -> leakage removal (company-level + near-duplicate) against both test sets
    -> balance fair/unfair for training (test sets keep their natural mix)
    -> teacher writes evidence + explanation for unfair clauses (verbatim-quote checked)
    -> CuratorKIT stage A: LLM grounding gate on the explanations
    -> CuratorKIT stage B: schema, exact dedup, clean, sft_messages export with val split
    -> optional push of the SFT dataset and the eval sets to the Hub

Usage:
  python scripts/01_build_data.py                  # full run (needs HF_TOKEN: teacher + judge go through the HF router)
  python scripts/01_build_data.py --no-teacher     # offline: template explanations, no judge
  python scripts/01_build_data.py --limit 300 --no-teacher   # smoke test
  python scripts/01_build_data.py --push           # also upload datasets to the Hub
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smallprint.config import load_config  # noqa: E402
from smallprint.hub import CC_BY_NC_25, safe_push  # noqa: E402
from smallprint.curate import assemble_sft, grounding_gate  # noqa: E402
from smallprint.data import (  # noqa: E402
    balance, companies_overlapping, decontaminate, eval_record, from_lexglue, from_multilingual,
    sft_record, summarise, write_jsonl,
)
from smallprint.teacher import annotate, litellm_completion, resolve_api_key, template_annotation  # noqa: E402

log = logging.getLogger("build_data")


def _level_to_str(ds):
    """If unfairness_level is a ClassLabel, map ints to names so labels can't be misread."""
    feat = ds.features.get("unfairness_level")
    if hasattr(feat, "int2str"):
        return ds.map(lambda r: {"unfairness_level": feat.int2str(r["unfairness_level"])})
    return ds


def _rows(ds, limit):
    return ds.select(range(min(limit, len(ds)))) if limit else ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--no-teacher", action="store_true", help="template explanations only")
    ap.add_argument("--no-judge", action="store_true", help="skip CuratorKIT's LLM grounding gate")
    ap.add_argument("--limit", type=int, default=0, help="rows per split, for smoke tests")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for noisy in ("LiteLLM", "litellm", "httpx", "httpcore"):  # one INFO line per API call otherwise
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = load_config(args.config)
    d = cfg["data"]
    work = Path(d["work_dir"])
    from datasets import load_dataset

    lex = load_dataset(d["lexglue_repo"], d["lexglue_config"])
    ml = load_dataset(d["multilingual_repo"])
    ml = {k: _level_to_str(v) for k, v in ml.items()}

    lex_train = from_lexglue(_rows(lex["train"], args.limit), "train") + \
        from_lexglue(_rows(lex["validation"], args.limit), "validation")
    lex_test = from_lexglue(_rows(lex["test"], args.limit), "test")
    ml_train = from_multilingual(_rows(ml["train"], args.limit), "train") + \
        from_multilingual(_rows(ml["validation"], args.limit), "validation")
    ml_test = from_multilingual(_rows(ml["test"], args.limit), "test")

    # 1. leakage: drop multilingual companies whose English text sits in LexGLUE test (all languages)
    leaky = companies_overlapping(ml_train, lex_test, d["company_overlap_share"])
    ml_train = [c for c in ml_train if c.company not in leaky]
    # 2. leakage: near-duplicates of any test sentence
    pool = lex_train + ml_train
    kept, dropped = decontaminate(pool, lex_test + ml_test, d["decontam_similarity"])
    train = balance(kept, d["fair_per_unfair"], cfg["seed"])
    log.info("train pool %d -> %d after decontamination -> %d balanced", len(pool), len(kept), len(train))

    write_jsonl(work / "interim" / "train_clauses.jsonl", (c.to_dict() for c in train))
    write_jsonl(work / "eval" / "lexglue_test.jsonl", (eval_record(c) for c in lex_test))
    write_jsonl(work / "eval" / "claudette_ml_test.jsonl", (eval_record(c) for c in ml_test))

    # 3. teacher annotations for unfair clauses (gold labels never change)
    t = cfg["teacher"]
    complete = None if (args.no_teacher or not t.get("model")) else \
        litellm_completion(t["model"], api_base=t.get("api_base"), api_key=resolve_api_key(t))
    ann = annotate(train, complete, work / "interim" / "teacher_annotations.jsonl", t.get("concurrency", 8))

    unfair = [c for c in train if c.unfair]
    fair = [c for c in train if not c.unfair]
    unfair_rows = {c.id: sft_record(c, ann[c.id]["evidence"], ann[c.id]["explanation"], ann[c.id]["annotation_source"])
                   for c in unfair}
    write_jsonl(work / "interim" / "unfair_candidates.jsonl", unfair_rows.values())

    # 4. CuratorKIT stage A: LLM judge on explanations
    j = cfg["judge"]
    judge = None if (args.no_judge or args.no_teacher) else j.get("model")
    passed, rejected = grounding_gate(work / "interim" / "unfair_candidates.jsonl", work / "interim" / "gate",
                                      judge, j.get("threshold", 0.7), j.get("api_base"), j.get("concurrency", 8),
                                      resolve_api_key(j) if judge else None)
    write_jsonl(work / "interim" / "gate_rejected.jsonl", rejected)
    # A rejected explanation is the teacher's fault, not the label's: keep the expert label, use the template text.
    final_unfair, final_ann = [], []
    for c in unfair:
        a = ann[c.id] if c.id in passed else {**template_annotation(c), "annotation_source": "template_after_gate"}
        final_unfair.append(sft_record(c, a["evidence"], a["explanation"], a["annotation_source"]))
        final_ann.append(a)
    # Step 3 (DPO) reads these so its "chosen" answers match the SFT targets exactly.
    write_jsonl(work / "interim" / "final_annotations.jsonl", final_ann)

    rows = final_unfair + [sft_record(c) for c in fair]
    write_jsonl(work / "interim" / "sft_input.jsonl", rows)

    # 5. CuratorKIT stage B: assemble the trainer-ready dataset
    sft_dir = work / "smallprint-sft"
    res = assemble_sft(work / "interim" / "sft_input.jsonl", sft_dir, d["val_fraction"], cfg["seed"])

    sources: dict[str, int] = {}
    for r in rows:
        sources[r["annotation_source"]] = sources.get(r["annotation_source"], 0) + 1
    report = {
        "leaky_companies_dropped": sorted(leaky),
        "near_duplicates_dropped": len(dropped),
        "train_clauses": summarise(train),
        "test_clauses": summarise(lex_test + ml_test),
        "annotation_sources": sources,
        "gate_rejected": len(rejected),
        "curatorkit_passed": len(res.passed),
        "curatorkit_rejected": len(res.rejected),
    }
    (work / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if args.push:
        push_datasets(cfg, sft_dir, work / "eval")


EVAL_README = """---
""" + CC_BY_NC_25 + """language: [en, de, it, pl]
configs:
- config_name: lexglue_test
  data_files: lexglue_test.jsonl
- config_name: claudette_ml_test
  data_files: claudette_ml_test.jsonl
---
# Smallprint evaluation sets

Clause-level test sets used to score Smallprint before and after fine-tuning.

* `lexglue_test`: LexGLUE UNFAIR-ToS test split (English, CC BY 4.0), 8 unfair-term categories.
* `claudette_ml_test`: multilingual CLAUDETTE test companies (EN/DE/IT/PL, CC BY-NC 2.5), 9 categories.

Natural class distribution (no balancing). Training data was decontaminated against both sets.
Sources: Chalkidis et al. 2022 (LexGLUE); Lippi et al. 2019; Drawzeski et al. 2021.
"""


LICENCE_NOTE = """

## Licence and sources

Non-commercial use only. Built from LexGLUE UNFAIR-ToS (CC BY 4.0; Chalkidis et al. 2022, from Lippi et al. 2019)
and the multilingual CLAUDETTE ToS corpus (CC BY-NC 2.5; Drawzeski et al. 2021). Gold labels are the experts';
evidence quotes and explanations for unfair clauses were written by a teacher model and checked by
CuratorKIT's grounding gate plus a verbatim-quote check. Not legal advice.
"""


def _add_licence(readme: Path) -> None:
    """CuratorKIT writes the dataset card; add the licence it can't know about."""
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    if text.startswith("---\n") and "\nlicense:" not in text.split("\n---", 1)[0]:
        text = "---\n" + CC_BY_NC_25 + text[4:]
    if "## Licence and sources" not in text:
        text += LICENCE_NOTE
    readme.write_text(text, encoding="utf-8")


def push_datasets(cfg: dict, sft_dir: Path, eval_dir: Path) -> None:
    from huggingface_hub import HfApi

    hub = cfg["hub"]
    api = HfApi()
    for repo, folder in ((hub["dataset_repo"], sft_dir), (hub["eval_dataset_repo"], eval_dir)):
        if "/" not in repo:
            raise SystemExit("Set hub.user (or SMALLPRINT_HUB__USER) before --push")
        if folder == eval_dir:
            (eval_dir / "README.md").write_text(EVAL_README)
        else:
            _add_licence(folder / "README.md")
        def push(repo=repo, folder=folder):
            api.create_repo(repo, repo_type="dataset", private=hub["private"], exist_ok=True)
            api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type="dataset")
        if safe_push(repo, push):
            print("pushed", f"https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
