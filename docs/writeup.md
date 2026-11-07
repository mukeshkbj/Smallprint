# Smallprint: write-up (skeleton for the HF article or README submission)

> Fill in the [X] values from `results/summary.md`, `results/roi.md` and `data/build_report.json`.
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

- **Leakage removal** (both come from the same project): [X] companies dropped at company level (with all
  translations), [X] near-duplicate sentences dropped. Explain why: without it, the "after" number would be
  partly memorisation.
- **Balancing**: training only, 2 fair : 1 unfair per language. Test sets keep their natural mix ([X]% unfair).
- **Teacher text**: Aya Expanse 32B wrote evidence and explanation for [X] unfair clauses. [X]% passed the
  verbatim check first time, [X]% after one retry. CuratorKIT's grounding gate rejected [X]%; those fell back to
  templates while keeping the expert label.

## 4. Training

- Model: Tiny Aya Water (3.35B, tuned for European languages), the target for on-device use. QLoRA (r=16).
- SFT with AlignTune on CuratorKIT's `sft_messages` export: [X] examples, [X] epochs, [GPU], [X] GPU-hours.
- DPO with AlignTune: [X] pairs, of which [X] were the SFT model's own mistakes and [X] targeted negatives (made-up
  quote, over-flag, under-flag, wrong category, prose). β=0.1.

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

[Paste the tables from `results/summary.md`: LexGLUE, CLAUDETTE all languages, CLAUDETTE by language.]

[3–5 sentences: what improved most, what didn't, one concrete example clause before vs after.]

## 7. Why it's better in use

- [Before/after example: base model output vs Smallprint output on the same clause.]
- [False alarms per typical policy: e.g. "on a [X]-clause policy the base model raises [X] false alarms, Smallprint [X]".]
- Every flag can be checked against the quoted text, highlighted in the demo.

## 8. ROI

[Paste `results/roi.md`. Note the price sources and dates, the measured throughput and GPU, and that on-device
cost is zero at the margin and keeps documents private.]

## 9. Limitations and next steps

- Sentence-level units: terms spread over several sentences can be missed.
- Nine categories from EU consumer-law research; no category for data selling, for example (OPP-115 could add
  privacy-practice categories, but its licence is non-commercial and share-alike).
- Explanations come from a teacher and a judge; spot-checked, but not lawyer-reviewed.
- Non-commercial licence inherited from Tiny Aya.
- Next: GGUF build for the Chrome extension, more languages, SafeTune to check that the domain tuning hasn't weakened refusals.

## 10. Reproducing

See the README's Quickstart and Reproducibility sections, `notebooks/smallprint_colab.ipynb` and the dataset cards
(CuratorKIT manifests and checksums).
