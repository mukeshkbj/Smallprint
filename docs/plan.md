# 7-day plan

Deadline: **Saturday 14 Nov, 23:59 AoE**, which is **Sunday 15 Nov, 11:59 UK time**. Aim to submit on
Saturday evening and keep Sunday morning as buffer. Only the last submission counts, so submit early
and resubmit.

Lexsi office hours: daily 8–9 PM IST = **14:30–15:30 UK** on Discord.

| Day | Goal | Done when |
|---|---|---|
| **Sat 7** | Set up | Repo scaffolded; Tiny Aya licence accepted on HF; HF credits claimed (Discord joined, HF username matches registration); Cohere API key; Discord ticket raised (`docs/discord-ticket.md`); Colab smoke test (`SMOKE_TEST=True`) runs end to end |
| **Sun 8** | Data + baseline | `01_build_data.py --push` with teacher and judge; `build_report.json` looks sane (leakage dropped, few template fallbacks); **base model scored** on both test sets (`05_evaluate.py --skip-final`), so a "before" exists whatever happens next |
| **Mon 9** | SFT | `02_train_sft.py --push`; quick eval on 300 samples per set; read 20 wrong predictions |
| **Tue 10** | DPO | `03_build_dpo.py` + `04_train_dpo.py --push`; full eval with `--with-sft` |
| **Wed 11** | One improvement + demo | Change one thing based on the error analysis (e.g. `fair_per_unfair`, DPO `beta`, more on-policy negatives); re-evaluate; deploy the Space |
| **Thu 12** | ROI + write-up | Prices filled in (with date and source); `06_roi.py`; tokens/sec measured on GPU and, if time, on a laptop (GGUF / llama.cpp); `docs/writeup.md` and `docs/abstract.md` drafted with real numbers |
| **Fri 13** | Polish | README results table; HF collections (model + datasets) public; feedback at office hours; dry-run the submission form |
| **Sat 14** | Submit | Submit by evening UK time; Sunday morning buffer only |

## Fallbacks

- **Training breaks or runs out of GPU time:** SFT alone is a valid "after". DPO is the improvement on top.
- **No Cohere key or credits:** use `--no-teacher` (template explanations and whole-clause evidence). The label metrics are unaffected; only explanation quality drops.
- **Full test sets are too slow:** use `eval.backend: vllm`, or `--max-samples 1000` (uniform sampling keeps the natural class mix). Say so in the write-up.
- **Free T4 is too slow:** Kaggle gives 2×T4 for 30 h/week; the $20 HF credits can rent an L4 for a few hours.
