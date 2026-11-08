# Results (full runs, 7–8 Nov 2026)

Short version: **SFT is the best model.** DPO v1 over-flagged; DPO v2 (label-balanced pairs) got close
but still scored below SFT, so SFT is what `mukeshkbj/smallprint-tiny-aya-water-3b` holds.

## Run 1: base, SFT, DPO v1

HF Job `6aef94790e0338d1bb83d766`, A100 80 GB, 21:13–00:30 UTC. Test sets at their natural class mix:
LexGLUE UNFAIR-ToS test (1,607 clauses, 172 unfair) and multilingual CLAUDETTE test companies
(4,297 clauses, 385 unfair, EN/DE/IT/PL). All three models were scored by AuditKIT on the same clauses.

`base` = Tiny Aya Water, `sft` = base + LoRA SFT adapter, `final` = SFT + DPO v1 (merged).

## LexGLUE test (n=1607)

| Metric | base | sft | final (DPO v1) |
|---|---:|---:|---:|
| Valid JSON | 0.253 | 0.999 | 0.999 |
| Unfair-term F1 | 0.263 | **0.788** | 0.585 |
| Category micro-F1 | 0.162 | **0.743** | 0.535 |
| Category macro-F1 | 0.153 | **0.759** | 0.572 |
| False alarms on fair clauses (lower is better) | 0.229 | **0.045** | 0.163 |
| Missed unfair clauses (lower is better) | 0.558 | 0.105 | **0.023** |
| Evidence is a real quote | 0.736 | **0.986** | 0.985 |

## Multilingual CLAUDETTE test (n=4297)

| Metric | base | sft | final (DPO v1) |
|---|---:|---:|---:|
| Valid JSON | 0.244 | **0.994** | 0.986 |
| Unfair-term F1 | 0.268 | **0.657** | 0.450 |
| Category micro-F1 | 0.161 | **0.629** | 0.396 |
| Category macro-F1 | 0.162 | **0.569** | 0.358 |
| False alarms on fair clauses (lower is better) | 0.219 | **0.071** | 0.199 |
| Missed unfair clauses (lower is better) | 0.501 | 0.156 | **0.122** |
| Evidence is a real quote | 0.872 | 0.935 | **0.952** |

SFT by language (CLAUDETTE), unfair-term F1: DE 0.633, EN 0.696, IT 0.675, PL 0.625.
Missed unfair clauses: DE 0.128, EN 0.085, IT 0.137, PL 0.265 (Polish is the weak spot).

## What went wrong with DPO v1

DPO cut missed unfair clauses (10.5% to 2.3% on LexGLUE) but tripled false alarms on fair clauses
(4.5% to 16.3%). About 90% of real clauses are fair, so F1 fell by 0.2.

Cause: the preference pairs. Pairs whose chosen and rejected answers differ on the fair/unfair
decision move the model's prior. v1 had 400 under-flag pairs and 400 prose pairs (all on unfair
clauses, both rewarding `"unfair": true`) against 400 over-flag pairs (rewarding `"unfair": false`):
about two "flag it" signals for every "leave it" signal. With beta 0.1 and lr 1e-5 the model moved a
long way from SFT (DPO train loss 0.40, reward accuracy ~0.87).

AuditKIT's before/after comparison caught the regression before anything was published.

## Fix for DPO v2

- Decision-moving pairs follow the 1 unfair : 2 fair training mix that SFT is calibrated on:
  under_flag 200, prose_unfair 200 vs over_flag 400, prose_fair 400. fabricated_quote and
  wrong_category (400 each) keep the decision fixed, so they only teach grounding and categories.
- beta 0.1 to 0.2 and learning rate 1e-5 to 5e-6, to stay closer to the SFT model.
- The SFT predictions are re-scored from saved outputs, so base and SFT numbers are identical across runs.

## Run 2: DPO v2 (8 Nov)

HF Job `6af047c00e0338d1bb84a602`, A100, resumed from the Hub (SFT not retrained). 2,242 training pairs:
386 SFT mistakes (out of 3,000 training clauses) plus 2,000 targeted negatives (over_flag 400, prose_fair
400, under_flag 200, prose_unfair 200, fabricated_quote 400, wrong_category 400). β=0.2, lr 5e-6,
1 epoch, 20 minutes.

| Metric | SFT | SFT + DPO v2 | SFT | SFT + DPO v2 |
|---|---:|---:|---:|---:|
| | **LexGLUE** | | **CLAUDETTE** | |
| Valid JSON | 0.999 | 0.999 | 0.994 | 0.990 |
| Unfair-term F1 | **0.788** | 0.759 | **0.657** | 0.616 |
| Category micro-F1 | **0.743** | 0.719 | **0.629** | 0.586 |
| Category macro-F1 | **0.759** | 0.744 | **0.569** | 0.537 |
| False alarms on fair clauses (lower is better) | **0.045** | 0.063 | **0.071** | 0.089 |
| Missed unfair clauses (lower is better) | 0.105 | **0.064** | 0.156 | **0.151** |
| Evidence is a real quote | 0.986 | **0.988** | 0.935 | **0.948** |

Balancing the pairs fixed most of v1's damage (LexGLUE F1 0.585 to 0.759; false alarms 16.3% to 6.3%), and
DPO v2 misses fewer unfair clauses on LexGLUE (6.4% vs 10.5%). It still raises more false alarms than SFT and
scores lower on every F1, in every language (CLAUDETTE unfair-term F1, DE/EN/IT/PL: 0.585 / 0.657 / 0.634 /
0.591). By the promotion rule (mean of unfair-term F1 and category macro-F1 over both test sets) SFT wins,
0.693 to 0.664.

If missing an unfair clause matters more to you than an extra false alarm, DPO v2 is the higher-recall
option on LexGLUE; it is published separately as `mukeshkbj/smallprint-tiny-aya-water-3b-dpo`.

## An evaluation bug worth knowing about

The first scoring of run 2 reported SFT at the baseline's numbers (F1 0.263) and published DPO v2 as the
winner. AuditKIT caches results by a fingerprint of the model, and our two "replay saved predictions"
functions fingerprinted the same, so the SFT replay was served the baseline's cached result.
`run_eval` now gives every call its own model identity (a test reproduces the collision), the SFT results
were restored from the earlier commit of the eval repo, and a repair job re-scored all three models and
published the real winner.
