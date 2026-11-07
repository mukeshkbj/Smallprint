"""Teacher annotations: a verbatim evidence quote and a one-sentence plain
explanation for every *unfair* training clause.

The gold label (unfair + categories) always comes from the expert-annotated
datasets; the teacher only writes the evidence/explanation text. Allowed
teachers under the hackathon rules are any model used in a supplementary role
(default: Aya Expanse 32B via the Cohere API). Each answer is checked with
``grounding.is_verbatim`` and retried once with feedback; anything still
ungrounded falls back to a template (whole clause as evidence, category
template as explanation) and is marked ``annotation_source="template"``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .data import Clause, read_jsonl
from .grounding import is_verbatim
from .labels import BY_KEY
from .prompts import TEACHER_PROMPT, TEACHER_RETRY_SUFFIX
from .schema import _first_json_object

log = logging.getLogger(__name__)

Annotation = dict  # {"clause_id", "evidence", "explanation", "annotation_source"}
CompletionFn = Callable[[str], str]


def template_annotation(c: Clause) -> Annotation:
    expl = " ".join(BY_KEY[k].plain_template for k in c.categories[:2])
    return {"clause_id": c.id, "evidence": c.text.strip(), "explanation": expl,
            "annotation_source": "template"}


def litellm_completion(model: str, temperature: float = 0.2, max_tokens: int = 300,
                       api_base: str | None = None) -> CompletionFn:
    import litellm  # installed with curatorkit[generation]

    def _call(prompt: str) -> str:
        resp = litellm.completion(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=max_tokens, api_base=api_base,
            num_retries=3,
        )
        return resp.choices[0].message.content or ""
    return _call


def _parse(text: str) -> dict | None:
    blob = _first_json_object(text or "")
    if not blob:
        return None
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    ev, ex = obj.get("evidence"), obj.get("explanation")
    if not isinstance(ev, str) or not isinstance(ex, str) or not ex.strip():
        return None
    return {"evidence": ev.strip(), "explanation": ex.strip()}


def annotate_one(c: Clause, complete: CompletionFn) -> Annotation:
    cats = ", ".join(BY_KEY[k].name for k in c.categories)
    prompt = TEACHER_PROMPT.format(lang=c.lang, clause=c.text, categories=cats)
    last_bad = None
    for _ in range(2):
        p = prompt if last_bad is None else prompt + TEACHER_RETRY_SUFFIX.format(bad=last_bad)
        try:
            got = _parse(complete(p))
        except Exception as e:  # network / provider errors: fall back, never crash the run
            log.warning("teacher error on %s: %s", c.id, e)
            got = None
        if got and is_verbatim(got["evidence"], c.text):
            return {"clause_id": c.id, **got, "annotation_source": "teacher"}
        last_bad = got["evidence"] if got else "(no valid JSON)"
    return template_annotation(c)


def annotate(clauses: Iterable[Clause], complete: CompletionFn | None, cache_path: str | Path,
             concurrency: int = 8) -> dict[str, Annotation]:
    """Annotate unfair clauses, resuming from ``cache_path`` (JSONL)."""
    cache_path = Path(cache_path)
    done: dict[str, Annotation] = {}
    if cache_path.exists():
        done = {a["clause_id"]: a for a in read_jsonl(cache_path)}
    todo = [c for c in clauses if c.unfair and c.id not in done]
    if not todo:
        return done
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        if complete is None:
            for c in todo:
                a = template_annotation(c)
                done[c.id] = a
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
            return done
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = {pool.submit(annotate_one, c, complete): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                a = fut.result()
                done[a["clause_id"]] = a
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
                if i % 100 == 0:
                    f.flush()
                    log.info("annotated %d/%d", i, len(todo))
    return done
