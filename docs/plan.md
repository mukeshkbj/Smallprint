# 7-day plan

Deadline: **Sunday 15 Nov, 23:59 AoE**, which is **Monday 16 Nov, 11:59 UK time** (moved from 14 Nov; check
hackathon.lexsi.ai/submit). Aim to submit by Saturday evening and keep Sunday as buffer. Only the last submission counts, so submit early
and resubmit.

Lexsi office hours: daily 8–9 PM IST = **14:30–15:30 UK** on Discord.

| Day | Goal | Done when |
|---|---|---|
| **Sat 7** | Set up | Repo scaffolded ✓; Tiny Aya licence accepted ✓; $40 HF credits on the account ✓ (teacher and judge run on them through the HF router); HF write token in Colab secrets; `00_check_providers.py` passes; Discord ticket raised (`docs/discord-ticket.md`); Colab smoke test (`SMOKE_TEST=True`) runs end to end |
| **Sun 8** | Data + baseline | `01_build_data.py --push` with teacher and judge; `build_report.json` looks sane (leakage dropped, few template fallbacks); **base model scored** on both test sets (`05_evaluate.py --skip-final`), so a "before" exists whatever happens next |
| **Mon 9** | SFT | `02_train_sft.py --push`; quick eval on 300 samples per set; read 20 wrong predictions |
| **Tue 10** | DPO | `03_build_dpo.py` + `04_train_dpo.py --push`; full eval with `--with-sft` |
| **Wed 11** | One improvement + demo | Change one thing based on the error analysis (e.g. `fair_per_unfair`, DPO `beta`, more on-policy negatives); re-evaluate; deploy the Space |
| **Thu 12** | ROI + write-up | Prices filled in (with date and source); `06_roi.py`; tokens/sec measured on GPU and, if time, on a laptop (GGUF / llama.cpp); `docs/writeup.md` and `docs/abstract.md` drafted with real numbers |
| **Fri 13** | Polish | README results table; HF collections (model + datasets) public; feedback at office hours; dry-run the submission form |
| **Sat 14** | Submit | Submit by evening UK time; Sunday morning buffer only |

## Fallbacks

- **Training breaks or runs out of GPU time:** SFT alone is a valid "after". DPO is the improvement on top.
- **HF router call fails (402 = out of credits, provider down):** point `teacher`/`judge` at another provider in `configs/smallprint.yaml` (Nebius or Cohere directly), or use `--no-teacher` (template explanations and whole-clause evidence). The label metrics are unaffected; only explanation quality drops.
- **Full test sets are too slow:** use `eval.backend: vllm`, or `--max-samples 1000` (uniform sampling keeps the natural class mix). Say so in the write-up.
- **Free T4 is too slow:** Kaggle gives 2×T4 for 30 h/week; the $20 HF credits can rent an L4 for a few hours.
