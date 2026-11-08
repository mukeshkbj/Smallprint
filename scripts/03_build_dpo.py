#!/usr/bin/env python
"""Step 3: build DPO preference pairs.

chosen   = the gold answer (expert label + checked evidence/explanation)
rejected = (a) on-policy mistakes: the SFT model's own wrong answers on training clauses
           (wrong decision, wrong categories, invalid JSON, or a quote that isn't in the clause)
           (b) targeted negatives for the failure modes we care about most:
               fabricated_quote  - right label, evidence not in the clause
               over_flag         - a fair clause flagged as unfair
               under_flag        - an unfair clause marked fair
               wrong_category    - right decision, wrong category
               prose_unfair/fair - an answer that isn't JSON

Label balance matters. Pairs whose chosen and rejected answers differ on the
fair/unfair decision move the model's prior. Run 1 had ~2 "flag it" pairs for
every "leave it" pair, and DPO tripled false alarms on fair clauses (4.5% ->
16.3% on LexGLUE). Decision-moving pairs now follow the training mix (1 unfair
: 2 fair, the mix SFT was calibrated on): over_flag and prose_fair get twice
the cap of under_flag and prose_unfair. fabricated_quote and wrong_category
keep the decision fixed, so they don't move it.

Pairs go through CuratorKIT (schema check, exact dedup, provenance, split) and
are then rewritten in TRL's conversational format (system+user prompt,
assistant chosen/rejected) so the chat template is applied during DPO.

Usage:
  python scripts/03_build_dpo.py
  python scripts/03_build_dpo.py --no-mine          # targeted negatives only (no GPU needed)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smallprint.config import load_config  # noqa: E402
from smallprint.curate import assemble_dpo  # noqa: E402
from smallprint.data import Clause, read_jsonl, write_jsonl  # noqa: E402
from smallprint.grounding import is_verbatim  # noqa: E402
from smallprint.labels import ALL_KEYS, BY_KEY  # noqa: E402
from smallprint.prompts import SYSTEM_PROMPT, user_message  # noqa: E402
from smallprint.schema import parse_output, target_json  # noqa: E402


def gold_answer(c: Clause, ann: dict) -> str:
    a = ann.get(c.id, {})
    return target_json(c.unfair, c.categories, a.get("evidence", ""), a.get("explanation", ""))


def is_wrong(c: Clause, raw: str) -> bool:
    p = parse_output(raw)
    if not p.valid or p.flagged != c.unfair or set(p.label_set) != set(c.categories):
        return True
    return p.flagged and not is_verbatim(p.evidence, c.text)


def targeted(c: Clause, ann: dict, rng: random.Random) -> list[tuple[str, str]]:
    out = []
    a = ann.get(c.id, {})
    if c.unfair:
        fake = BY_KEY[c.categories[0]].plain_template.rstrip(".")
        out.append(("fabricated_quote", target_json(True, c.categories, fake, a.get("explanation", ""))))
        out.append(("under_flag", target_json(False, [])))
        others = [k for k in ALL_KEYS if k not in c.categories]
        out.append(("wrong_category", target_json(True, [rng.choice(others)], a.get("evidence", ""),
                                                  a.get("explanation", ""))))
        out.append(("prose_unfair", f"This clause looks unfair because of {BY_KEY[c.categories[0]].name.lower()}."))
    else:
        words = c.text.split()
        k = rng.choice(ALL_KEYS)
        out.append(("over_flag", target_json(True, [k], " ".join(words[: min(8, len(words))]),
                                             BY_KEY[k].plain_template)))
        out.append(("prose_fair", "This clause looks fine; there is nothing unfair in it."))
    return out


def caps_for(per_type: int, fair_per_unfair: float) -> dict[str, int]:
    """Per-type caps that keep decision-moving pairs at the training label mix."""
    fair = int(round(per_type * fair_per_unfair))
    half = per_type // 2
    return {"fabricated_quote": per_type, "wrong_category": per_type,
            "under_flag": half, "over_flag": int(round(half * fair_per_unfair)),
            "prose_unfair": half, "prose_fair": int(round(half * fair_per_unfair)),
            "_fair_total": fair}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--model", help="model to mine mistakes from (default runs/sft/merged)")
    ap.add_argument("--no-mine", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = random.Random(cfg["seed"])
    work = Path(cfg["data"]["work_dir"])
    dp = cfg["dpo"]

    clauses = [Clause.from_dict(r) for r in read_jsonl(work / "interim" / "train_clauses.jsonl")]
    ann = {a["clause_id"]: a for a in read_jsonl(work / "interim" / "final_annotations.jsonl")}
    rng.shuffle(clauses)

    pairs: list[dict] = []

    def add(c: Clause, rejected: str, kind: str) -> None:
        pairs.append({"prompt": user_message(c.text), "chosen": gold_answer(c, ann), "rejected": rejected,
                      "system_prompt": SYSTEM_PROMPT, "clause_id": c.id, "negative_type": kind,
                      "lang": c.lang})

    # (a) on-policy mistakes
    if not args.no_mine:
        from smallprint.model import HFClauseModel
        model_id = args.model or str(Path(cfg["sft"]["output_dir"]) / "merged")
        sample = clauses[: dp["mine_from_train"]]
        model = HFClauseModel(model_id, batch_size=cfg["eval"]["batch_size"],
                              max_new_tokens=cfg["eval"]["max_new_tokens"])
        raws = model([c.text for c in sample])
        mined = 0
        for c, raw in zip(sample, raws):
            if is_wrong(c, raw):
                add(c, raw.strip(), "on_policy")
                mined += 1
        print(f"on-policy mistakes: {mined}/{len(sample)}")

    # (b) targeted negatives, capped per type
    limit = caps_for(dp["synthetic_per_type"], cfg["data"]["fair_per_unfair"])
    caps: dict[str, int] = {}
    for c in clauses:
        for kind, rej in targeted(c, ann, rng):
            if caps.get(kind, 0) < limit[kind]:
                caps[kind] = caps.get(kind, 0) + 1
                add(c, rej, kind)
    print("targeted negatives:", caps)

    write_jsonl(work / "interim" / "dpo_input.jsonl", pairs)
    out_dir = work / "smallprint-dpo"
    assemble_dpo(work / "interim" / "dpo_input.jsonl", out_dir, cfg["data"]["val_fraction"], cfg["seed"])

    # CuratorKIT dpo.jsonl (string fields) -> TRL conversational format
    chat_dir = work / "smallprint-dpo-chat"
    for f in sorted(out_dir.rglob("dpo.jsonl")):
        split = f.parent.name if f.parent != out_dir else "train"
        rows = [{
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r["prompt"]}],
            "chosen": [{"role": "assistant", "content": r["chosen"]}],
            "rejected": [{"role": "assistant", "content": r["rejected"]}],
        } for r in read_jsonl(f)]
        n = write_jsonl(chat_dir / f"{split}.jsonl", rows)
        print(f"{split}: {n} pairs -> {chat_dir / (split + '.jsonl')}")
    (chat_dir / "negative_types.json").write_text(json.dumps(
        {k: sum(p["negative_type"] == k for p in pairs) for k in {p["negative_type"] for p in pairs}}, indent=2))


if __name__ == "__main__":
    main()
