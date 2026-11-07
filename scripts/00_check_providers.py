#!/usr/bin/env python
"""Step 0: check the teacher and judge endpoints before the full data build.

Sends a few sample clauses to the teacher and one annotation to the judge,
using exactly the config the build will use. It reports whether each call
worked, whether the teacher's quote is verbatim, latency and token usage
per call, so you can estimate the cost of the full run before starting it.

Usage:
  HF_TOKEN=hf_... python scripts/00_check_providers.py
  python scripts/00_check_providers.py --n 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smallprint.config import load_config  # noqa: E402
from smallprint.data import Clause  # noqa: E402
from smallprint.grounding import is_verbatim  # noqa: E402
from smallprint.labels import BY_KEY  # noqa: E402
from smallprint.prompts import GROUNDING_JUDGE_PROMPT, TEACHER_PROMPT  # noqa: E402
from smallprint.schema import _first_json_object, target_json  # noqa: E402
from smallprint.teacher import resolve_api_key  # noqa: E402

SAMPLES = [
    Clause("s1", "We may suspend or terminate your account at any time, for any reason, without prior notice.",
           "en", "check", "train", True, ["unilateral_termination"]),
    Clause("s2", "Wir behalten uns das Recht vor, diese Nutzungsbedingungen jederzeit ohne Ankündigung zu ändern.",
           "de", "check", "train", True, ["unilateral_change"]),
    Clause("s3", "Qualsiasi controversia sarà risolta esclusivamente mediante arbitrato vincolante a Milano.",
           "it", "check", "train", True, ["arbitration"]),
    Clause("s4", "Nie ponosimy odpowiedzialności za jakiekolwiek szkody pośrednie wynikające z korzystania z serwisu.",
           "pl", "check", "train", True, ["limitation_of_liability"]),
    Clause("s5", "By continuing to use the app you accept these terms, whether or not you have read them.",
           "en", "check", "train", True, ["contract_by_using"]),
]


def call(section: dict, prompt: str) -> tuple[str, dict, float]:
    import litellm
    t0 = time.perf_counter()
    r = litellm.completion(model=section["model"], messages=[{"role": "user", "content": prompt}],
                           api_base=section.get("api_base"), api_key=resolve_api_key(section),
                           temperature=0.1, max_tokens=300, num_retries=1)
    u = getattr(r, "usage", None)
    usage = {"in": getattr(u, "prompt_tokens", 0) or 0, "out": getattr(u, "completion_tokens", 0) or 0}
    return r.choices[0].message.content or "", usage, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(args.config)
    t, j = cfg["teacher"], cfg["judge"]
    ok_all = True

    print(f"teacher: {t['model']} via {t.get('api_base') or 'provider default'}")
    first = None
    for c in SAMPLES[: args.n]:
        prompt = TEACHER_PROMPT.format(lang=c.lang, clause=c.text,
                                       categories=", ".join(BY_KEY[k].name for k in c.categories))
        try:
            text, usage, secs = call(t, prompt)
        except Exception as e:  # show the provider's message (402 = out of credits, 401 = bad token, ...)
            print(f"  [{c.lang}] FAILED: {type(e).__name__}: {str(e)[:300]}")
            ok_all = False
            continue
        blob = _first_json_object(text)
        obj = json.loads(blob) if blob else {}
        verbatim = is_verbatim(obj.get("evidence", ""), c.text)
        print(f"  [{c.lang}] {secs:.1f}s, {usage['in']} in / {usage['out']} out tokens, "
              f"quote verbatim: {verbatim}\n       evidence: {obj.get('evidence')!r}\n"
              f"       explanation: {obj.get('explanation')!r}")
        if first is None and obj:
            first = (c, obj)
        ok_all &= verbatim

    if j.get("model") and first:
        c, obj = first
        print(f"judge: {j['model']} via {j.get('api_base') or 'provider default'}")
        answer = target_json(True, c.categories, obj.get("evidence", ""), obj.get("explanation", ""))
        try:
            text, usage, secs = call(j, GROUNDING_JUDGE_PROMPT.format(source_text=c.text, question=c.text,
                                                                      answer=answer))
            blob = _first_json_object(text)
            score = json.loads(blob).get("grounding_score") if blob else None
            print(f"  {secs:.1f}s, {usage['in']} in / {usage['out']} out tokens, grounding_score: {score}")
            ok_all &= score is not None
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")
            ok_all = False

    print("\nAll good: run scripts/01_build_data.py" if ok_all else
          "\nSomething failed: fix it before the full build (or run 01 with --no-teacher).")
    print("Cost: full build is about 1 teacher call (+1 retry if the quote isn't verbatim) and 1 judge call "
          "per unfair training clause; data/build_report.json lists the count. Check usage at "
          "https://huggingface.co/settings/billing")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
