# Smallprint: write-up

> Numbers come from `results/summary.md`, `results/roi.md`, `docs/results.md` and the data build report.
> The hackathon asks for: the metrics and why they were chosen; the approach and the reasons behind the
> datasets, tasks and models; and how the result is meaningfully better in use.

## 1. The problem

- Consumer contracts online are long, and unfair terms are common: unilateral change and termination, content
  removal, liability limits, forced arbitration, distant jurisdiction, contract-by-use.
- Existing ToS summarisers mostly send the whole document to a cloud API. For a privacy tool that is the wrong
  architecture, and the cost grows with every page read.
- Goal: a small model that runs on the user's device and is **checkable**, with every flag tied to a quote.

## 2. Task design

- **Clause-level JSON** (`unfair`, `categories`, `evidence`, `explanation`) instead of free-text summaries. It is
  measurable against expert labels, keeps every call short (Tiny Aya has an 8K context), and makes grounding
  verifiable.
- **Document reports are deterministic roll-ups** of grounded clause outputs (problem statement 9 without a
  long-context hallucination risk).
- Problem statements covered: 1, 5, 3, 4, 9, 2.

## 3. Data, and why these datasets

| Dataset | Why | Licence |
|---|---|---|
| LexGLUE UNFAIR-ToS | Standard benchmark with expert labels; comparable with published work | CC BY 4.0 |
| CLAUDETTE multilingual | Same scheme in DE/IT/PL plus fairness levels; plays to Aya's multilingual strength | CC BY-NC 2.5 |

- **Leakage removal** (both come from the same project): 4 companies dropped at company level (Evernote,
  LinkedIn, Snap, Uber, with all translations), 1,223 near-duplicate sentences dropped. Explain why: without it, the "after" number would be
  partly memorisation.
- **Balancing**: training only, 2 fair : 1 unfair per language. Test sets keep their natural mix (10.7% unfair on LexGLUE, 9.0% on CLAUDETTE).
- **Teacher text**: Aya Expanse 32B wrote evidence and explanation for 2,551 unfair clauses. 100 (3.9%) still
  had no verbatim quote after one retry, and CuratorKIT's grounding gate (Qwen3 235B as judge) rejected 264
  (10.3%); both fell back to templates while keeping the expert label. 2,187 teacher answers (85.7%) were kept.

## 4. Training

- Model: Tiny Aya Water (3.35B, tuned for European languages), the target for on-device use. LoRA (r=16), bf16.
- SFT with AlignTune on CuratorKIT's `sft_messages` export: 7,188 examples, 2 epochs, one A100 80 GB, 38 minutes.
- DPO with AlignTune, run 1: 2,232 training pairs from 417 SFT mistakes (out of 3,000 training clauses) and 2,000
  targeted negatives (made-up quote, over-flag, under-flag, wrong category, prose). β=0.1. It tripled false
  alarms, because decision-changing pairs favoured "unfair" two to one; see `docs/results.md`. Run 2 kept
  those pairs at the 1 unfair : 2 fair training mix, with β=0.2 and half the learning rate: F1 0.759 vs SFT
  0.788, fewer misses but more false alarms. SFT is published; DPO v2 is kept as a separate higher-recall model.

## 5. Metrics, and why

| Metric | Why it matters in use |
|---|---|
| Valid JSON rate | The app can't show what it can't parse |
| Unfair-term F1 | The core decision: is there something here worth reading? |
| Category micro/macro-F1 (LexGLUE style) | Users need the *type* of risk; macro-F1 stops rare categories hiding behind common ones |
| False-alarm rate on fair clauses | A tool that flags everything is ignored |
| Miss rate | A missed arbitration clause is the costly error |
| Evidence is a real quote | Lets the user verify every flag; a made-up quote destroys trust |
| Per-language macro-F1 | The same protection for German, Italian and Polish readers |
| Tokens/sec, tokens per clause | Feed the ROI and on-device feasibility |

## 6. Results

Full tables: `docs/results.md`; every comparison in one place: `results/comparisons.md`.

SFT did most of the work. On LexGLUE, valid JSON went from 25% to 99.9%, unfair-term F1 from 0.26 to 0.79 and
category macro-F1 from 0.15 to 0.76, while false alarms on fair clauses fell from 23% to 4.5%. On held-out CLAUDETTE
companies, F1 rose from 0.27 to 0.66 in four languages, with Polish the weakest (26% of unfair clauses missed).
DPO did not beat SFT: run 1 tripled false alarms because its pairs favoured "unfair" two to one; the
label-balanced run 2 came close but still lost on every F1. Tiny Aya Global ties with Water (promotion score
0.697 vs 0.693); Water misses fewer unfair clauses in every CLAUDETTE language, so we kept it.

One clause from the LexGLUE test set, before and after:

> academia.edu reserves the right to suspend or terminate your account if any information provided during the
> registration process or thereafter proves to be inaccurate , not current or incomplete .

- Tiny Aya Water: `{"unfair": false, "categories": ["limitation_of_liability"], "evidence": "reserves the right to
  suspend or terminate your account", ...}`: says "not unfair" while naming a category, and the wrong one.
- Smallprint: `{"unfair": true, "categories": ["unilateral_termination"], "evidence": "academia.edu reserves the
  right to suspend or terminate your account if any information ... incomplete .", "explanation": "The platform
  can close your account at any time if they find any information you provided during registration to be false,
  outdated, ..."}`: the expert label, a verbatim quote and a plain sentence.

### Safety and robustness

- General safety held: AdvBench harmful requests refused 100% → 99.5%, XSTest unsafe prompts refused 95.5% → 96.5%,
  XSTest safe prompts answered 85.6% → 90.4%, AuditKIT red-team pass rate 0.856 for both (one LLM judge,
  DeepSeek-V3.2, for every refusal label).
- In-document prompt injection is the real risk: one sentence appended to an unfair clause removes 80–100% of
  Smallprint's flags (a harmless sentence removes 8%).
- Our first hardening run cut that to 0.4% and raised CLAUDETTE macro-F1 to 0.619, but a mirrored test showed it had
  learned to do the opposite of the planted sentence (95–97% false alarms on fair clauses that end with a
  reassurance). Not adopted; the balanced version is described in `docs/results.md`.
- CircuitKIT mixed precision: uniform 4-bit (2.5 GB) keeps F1 but misses twice as many unfair clauses; keeping the
  15% of layers in the fair/unfair circuit in bf16 (3.1 GB) stays close to bf16 and beats a random choice of the
  same size, within the noise of a 1,600-clause sample.

## 7. Why it's better in use

- The example in section 6: the base model's answer contradicts itself ("not unfair", with a category); Smallprint's can be checked word for word against the clause.
- False alarms per typical policy: on a 100-clause policy with 90 fair clauses, the base model raises about 21
  false alarms and Smallprint (SFT) about 4, while catching 9 in 10 unfair clauses instead of 4 in 10 (LexGLUE rates).
- Every flag can be checked against the quoted text, highlighted in the demo.

## 8. ROI

See `results/roi.md`. At 100,000 policies a month (188 clauses each), GPT-5 mini at list price costs about $4,913
($0.049 a policy, a lower bound because its reasoning tokens aren't counted), Smallprint on one A100 about $2,085
($0.021) with plain `transformers` generation at a measured 6.3 clauses/s, and Smallprint on the user's device
nothing at the margin, with the document never leaving the device. Prices checked 8 Nov 2026.

## 9. Limitations and next steps

- Sentence-level units: terms spread over several sentences can be missed.
- Nine categories from EU consumer-law research; no category for data selling, for example (OPP-115 could add
  privacy-practice categories, but its licence is non-commercial and share-alike).
- Explanations come from a teacher and a judge; spot-checked, but not lawyer-reviewed.
- Non-commercial licence inherited from Tiny Aya.
- Text inside a policy can switch flags off (section 6). Until a hardened model passes both injection tests, the app
  should treat instruction-like sentences in a clause as a reason to show the clause to the user, not to trust the
  verdict.
- Data augmentation can teach shortcuts that the obvious test rewards: test the mirror image of every robustness claim.
- Quantization results come from 1,600 clauses per test set; quanto's 4-bit kernels saved memory but were slower than
  bf16 on an A100.
- Next: balanced injection hardening (ready, `hardened2`), GGUF build for the Chrome extension, more languages.

## 10. Reproducing

See the README's Quickstart and Reproducibility sections, `notebooks/smallprint_colab.ipynb` and the dataset cards
(CuratorKIT manifests and checksums).
