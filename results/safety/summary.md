# Safety evaluation

Models: `base` = CohereLabs/tiny-aya-water, `smallprint` = mukeshkbj/smallprint-tiny-aya-water-3b

| | base | smallprint |
|---|---:|---:|
| AuditKIT red team: mean pass rate | 0.856 | 0.856 |
|   prompt_injection pass rate | 0.625 | 0.625 |
|   jailbreak pass rate | 1.000 | 1.000 |
|   encoding pass rate | 0.800 | 0.800 |
|   refusal pass rate | 1.000 | 1.000 |
| AdvBench harmful requests refused (higher is safer) | 1.000 | 0.995 |
| XSTest unsafe prompts refused (higher is safer) | 0.955 | 0.965 |
| XSTest safe prompts answered (higher is better) | 0.856 | 0.904 |
| Refusal labels from the LLM judge (rest: keyword fallback) | 1.000 | 1.000 |
| Unfair clauses flagged, clean | 0.483 | 0.855 |
| Unfair clauses flagged, harmless sentence added | 0.260 | 0.801 |
| Injection 'reviewer_note': flags removed (lower is better) | 1.000 | 0.979 |
| Injection 'override': flags removed (lower is better) | 1.000 | 1.000 |
| Injection 'legal_assurance': flags removed (lower is better) | 0.974 | 0.803 |
| Injection: mean flags removed (lower is better) | 0.991 | 0.927 |

Flags removed = share of the unfair clauses a model flagged clean that it no longer flags once the injection is appended. Red-team pass rates are AuditKIT's built-in detectors; refusal labels come from an LLM judge with a keyword fallback (see the *.responses.jsonl files).
