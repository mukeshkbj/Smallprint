"""Unfair-clause categories shared by every stage of the pipeline.

The first eight follow LexGLUE UNFAIR-ToS (Lippi et al., 2019) in index order.
``privacy_included`` exists only in the multilingual CLAUDETTE corpus
(Drawzeski et al., 2021), so LexGLUE scores use ``LEXGLUE_KEYS`` and the
multilingual scores use ``ALL_KEYS``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    key: str                 # stable id used in model output
    name: str                # human-readable name
    lexglue_index: int | None
    claudette_code: str      # column name in the multilingual corpus
    definition: str          # one line, used in the system prompt
    plain_template: str      # fallback consumer explanation (English)


CATEGORIES: tuple[Category, ...] = (
    Category(
        "limitation_of_liability", "Limitation of liability", 0, "ltd",
        "the provider limits or excludes its liability for losses or damage",
        "The company limits what it has to pay you if something goes wrong.",
    ),
    Category(
        "unilateral_termination", "Unilateral termination", 1, "ter",
        "the provider can suspend or end the service or your account at its discretion",
        "The company can close your account or stop the service whenever it decides to.",
    ),
    Category(
        "unilateral_change", "Unilateral change", 2, "ch",
        "the provider can change the terms or the service on its own",
        "The company can change these terms or the service without your agreement.",
    ),
    Category(
        "content_removal", "Content removal", 3, "cr",
        "the provider can delete or remove your content at its discretion",
        "The company can delete things you have posted or uploaded.",
    ),
    Category(
        "contract_by_using", "Contract by using", 4, "use",
        "you are bound just by using the service, without actively agreeing",
        "Simply using the service counts as agreeing to these terms.",
    ),
    Category(
        "choice_of_law", "Choice of law", 5, "law",
        "a specific country's or state's law governs the contract",
        "Disputes are decided under a law the company picked, which may not be your local law.",
    ),
    Category(
        "jurisdiction", "Jurisdiction", 6, "j",
        "disputes must be brought in courts the provider chose",
        "You may have to go to a court far away, chosen by the company, to bring a claim.",
    ),
    Category(
        "arbitration", "Arbitration", 7, "a",
        "disputes go to arbitration instead of a public court",
        "Disputes go to a private arbitrator instead of a public court.",
    ),
    Category(
        "privacy_included", "Privacy included", None, "pinc",
        "agreeing to these terms also counts as consenting to the privacy policy",
        "Accepting these terms also counts as accepting how your data is used.",
    ),
)

BY_KEY: dict[str, Category] = {c.key: c for c in CATEGORIES}
ALL_KEYS: tuple[str, ...] = tuple(c.key for c in CATEGORIES)
LEXGLUE_KEYS: tuple[str, ...] = tuple(
    c.key for c in sorted((c for c in CATEGORIES if c.lexglue_index is not None),
                          key=lambda c: c.lexglue_index)
)
LEXGLUE_INDEX_TO_KEY: dict[int, str] = {
    c.lexglue_index: c.key for c in CATEGORIES if c.lexglue_index is not None
}
CLAUDETTE_CODE_TO_KEY: dict[str, str] = {c.claudette_code: c.key for c in CATEGORIES}

# Lenient aliases a model might emit; normalised before scoring.
_ALIASES: dict[str, str] = {
    "liability": "limitation_of_liability",
    "limitation_of_liabilities": "limitation_of_liability",
    "termination": "unilateral_termination",
    "change": "unilateral_change",
    "unilateral_changes": "unilateral_change",
    "removal": "content_removal",
    "contract_by_use": "contract_by_using",
    "law": "choice_of_law",
    "governing_law": "choice_of_law",
    "forum": "jurisdiction",
    "arbitration_clause": "arbitration",
    "privacy": "privacy_included",
}
_ALIASES.update({c.claudette_code: c.key for c in CATEGORIES})
_ALIASES.update({c.name.lower().replace(" ", "_"): c.key for c in CATEGORIES})


def normalise_category(raw: str) -> str | None:
    """Map a raw category string to a canonical key, or None if unknown."""
    if not isinstance(raw, str):
        return None
    k = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if k in BY_KEY:
        return k
    return _ALIASES.get(k)
