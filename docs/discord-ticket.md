# Discord ticket (#report_issues → private channel)

Hi Lexsi team, I'm Mukesh, registered for Track 1. I'd appreciate a quick sanity check on my idea before I
commit the GPU time.

**Smallprint:** Tiny Aya Water fine-tuned to read Terms of Service and privacy policies clause by clause. For
each clause it returns JSON: unfair yes/no, category (the 8 LexGLUE UNFAIR-ToS types plus "privacy included"),
an exact quote as evidence, and a one-line explanation in the clause's language (EN/DE/IT/PL). The aim is
on-device use, so documents don't leave the user's machine.

- Data: LexGLUE UNFAIR-ToS and the multilingual CLAUDETTE corpus, with train/test leakage removed between them.
  A teacher (Aya Expanse 32B) writes the evidence and explanations, and CuratorKIT's grounding gate checks them
  with a different judge model (Qwen3 235B), both called through HF Inference Providers.
- Training: AlignTune QLoRA SFT, then DPO on the SFT model's own mistakes plus targeted negatives.
- Eval: AuditKIT before/after on the LexGLUE test set and held-out CLAUDETTE companies, plus an ROI table.
- Problem statements: 1, 5, 3, 4, 9, 2.

Three questions:
1. Is Tiny Aya Water the right pick here, or would you rather see Tiny Aya Global for a multilingual entry?
2. Is the 32K-context Tiny Aya Base on the allowed list? I'm working clause-level within 8K, so it isn't essential.
3. Anything you'd want to see in the metrics or the write-up that I haven't listed?

Repo: https://github.com/mukeshkbj/smallprint

Thanks!
