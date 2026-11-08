# Smallprint evaluation

`base` = Tiny Aya Water zero-shot, `sft` = LoRA SFT (the published model), `final` = SFT + DPO v2.
Copied from `mukeshkbj/smallprint-eval` (results/), 8 Nov 2026. Full story: `docs/results.md`.

## lexglue test set (n=1607)

| Metric | base | sft | final |
|---|---:|---:|---:|
| Valid JSON | 0.253 | 0.999 | 0.999 |
| Unfair-term F1 | 0.263 | 0.788 | 0.759 |
| Category micro-F1 | 0.162 | 0.743 | 0.719 |
| Category macro-F1 | 0.153 | 0.759 | 0.744 |
| False alarms (fair clauses) (lower is better) | 0.229 | 0.045 | 0.063 |
| Missed unfair clauses (lower is better) | 0.558 | 0.105 | 0.064 |
| Evidence is a real quote | 0.736 | 0.986 | 0.988 |

AuditKIT base -> final comparison: [`auditkit_compare__lexglue.txt`](auditkit_compare__lexglue.txt)

## claudette_ml test set (n=4297)

| Metric | base | sft | final |
|---|---:|---:|---:|
| Valid JSON | 0.244 | 0.994 | 0.990 |
| Unfair-term F1 | 0.268 | 0.657 | 0.616 |
| Category micro-F1 | 0.161 | 0.629 | 0.586 |
| Category macro-F1 | 0.162 | 0.569 | 0.537 |
| False alarms (fair clauses) (lower is better) | 0.219 | 0.071 | 0.089 |
| Missed unfair clauses (lower is better) | 0.501 | 0.156 | 0.151 |
| Evidence is a real quote | 0.872 | 0.935 | 0.948 |

### base: by language

| Metric | de | en | it | pl |
|---|---:|---:|---:|---:|
| Valid JSON | 0.260 | 0.241 | 0.236 | 0.240 |
| Unfair-term F1 | 0.271 | 0.279 | 0.250 | 0.271 |
| Category micro-F1 | 0.158 | 0.185 | 0.149 | 0.154 |
| Category macro-F1 | 0.172 | 0.178 | 0.151 | 0.146 |
| False alarms (fair clauses) (lower is better) | 0.234 | 0.214 | 0.214 | 0.214 |
| Missed unfair clauses (lower is better) | 0.479 | 0.489 | 0.547 | 0.490 |
| Evidence is a real quote | 0.820 | 0.880 | 0.892 | 0.897 |

### sft: by language

| Metric | de | en | it | pl |
|---|---:|---:|---:|---:|
| Valid JSON | 0.993 | 0.999 | 0.995 | 0.989 |
| Unfair-term F1 | 0.633 | 0.696 | 0.675 | 0.625 |
| Category micro-F1 | 0.616 | 0.667 | 0.662 | 0.571 |
| Category macro-F1 | 0.568 | 0.582 | 0.613 | 0.502 |
| False alarms (fair clauses) (lower is better) | 0.089 | 0.071 | 0.069 | 0.059 |
| Missed unfair clauses (lower is better) | 0.128 | 0.085 | 0.137 | 0.265 |
| Evidence is a real quote | 0.891 | 0.974 | 0.932 | 0.949 |

### final: by language

| Metric | de | en | it | pl |
|---|---:|---:|---:|---:|
| Valid JSON | 0.988 | 0.998 | 0.991 | 0.986 |
| Unfair-term F1 | 0.585 | 0.657 | 0.634 | 0.591 |
| Category micro-F1 | 0.561 | 0.623 | 0.613 | 0.545 |
| Category macro-F1 | 0.512 | 0.574 | 0.571 | 0.500 |
| False alarms (fair clauses) (lower is better) | 0.104 | 0.089 | 0.087 | 0.078 |
| Missed unfair clauses (lower is better) | 0.160 | 0.074 | 0.126 | 0.235 |
| Evidence is a real quote | 0.909 | 0.977 | 0.946 | 0.963 |

AuditKIT base -> final comparison: [`auditkit_compare__claudette_ml.txt`](auditkit_compare__claudette_ml.txt)
