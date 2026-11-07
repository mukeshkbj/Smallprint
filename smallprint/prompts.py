"""Prompts. The same system prompt is used for the zero-shot baseline and the
fine-tuned model, so before/after numbers compare like for like."""

from __future__ import annotations

from .labels import CATEGORIES

_CATEGORY_LINES = "\n".join(f"- {c.key}: {c.definition}" for c in CATEGORIES)

SYSTEM_PROMPT = f"""You are Smallprint. You read one clause from a consumer Terms of Service or privacy policy and flag terms that may be unfair to the consumer.

Categories:
{_CATEGORY_LINES}

Reply with exactly one JSON object and nothing else:
{{"unfair": true|false, "categories": [...], "evidence": "...", "explanation": "..."}}

Rules:
- "unfair" is true only if the clause contains a potentially or clearly unfair term from the list.
- "categories" lists every category that applies; it is [] when "unfair" is false.
- "evidence" is the shortest exact quote, copied word for word from the clause, that shows the unfair term; it is "" when "unfair" is false.
- "explanation" is one plain sentence, in the language of the clause, saying what this means for the consumer; it is "" when "unfair" is false. Do not give legal advice.
- A clause that mentions a topic (for example the governing law) but is fair to the consumer gets "unfair": false."""

FAIR_OUTPUT = {"unfair": False, "categories": [], "evidence": "", "explanation": ""}


def user_message(clause: str) -> str:
    """The user turn is the clause text alone; all instructions live in the system prompt."""
    return clause.strip()


def chat_messages(clause: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message(clause)},
    ]


TEACHER_PROMPT = """You annotate training data for a consumer-protection model.

Clause (language: {lang}):
\"\"\"{clause}\"\"\"

Expert annotators marked this clause as potentially unfair to the consumer, in these categories: {categories}.

Return a JSON object with exactly two keys:
- "evidence": the shortest exact quote from the clause (copied character for character, no ellipses, no added words) that shows the unfair term.
- "explanation": one plain sentence of at most 30 words, written in the same language as the clause, telling a consumer what this means for them. Only state what the clause says. No legal advice.

JSON only."""

TEACHER_RETRY_SUFFIX = """

Your previous evidence was not an exact quote from the clause: {bad!r}
Copy the quote character for character from the clause."""

# Judge prompt for CuratorKIT's hallucination gate. Placeholders are the gate's own:
# {source_text} = the clause, {question} = the clause (instruction), {answer} = target JSON.
GROUNDING_JUDGE_PROMPT = """You check training labels for a consumer-protection model.

Clause:
---
{source_text}
---

Proposed annotation (JSON): {answer}

Score how well the annotation's "explanation" is supported by the clause:
- Every statement in the explanation must follow from the clause text alone.
- The explanation must not add consequences, amounts, deadlines or rights that the clause does not state.
- It may simplify wording; that is fine.

Respond in JSON only:
{{"grounding_score": 0.XX, "supported_claims": ["..."], "unsupported_claims": ["..."], "verdict": "grounded | partially_grounded | hallucinated"}}

1.0 = fully supported, 0.7-0.9 = minor simplification, 0.4-0.6 = adds something not in the clause, 0.0-0.3 = contradicts or invents.

/no_think"""
