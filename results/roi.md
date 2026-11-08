Workload: 100,000 documents/month x 188 clauses = 18,800,000 clauses (15,142.2M input, 563.7M output tokens).

| Option | Monthly cost (USD) | Per document (USD) | Notes |
|---|---:|---:|---|
| OpenAI GPT-5 mini (API) | 4,913.05 | 0.0491 | $0.25/M in, $2.0/M out; list price, developers.openai.com/api/docs/pricing (via developersdigest.tech, verified 17 Oct 2026), checked 8 Nov 2026; documents leave the device |
| Smallprint self-hosted (1x A100 80GB (HF Jobs a100-large)) | 2,085.17 | 0.0209 | 834.1 GPU h at $2.5/h, 188 tok/s measured; huggingface.co/docs/hub/jobs-pricing, checked 8 Nov 2026 |
| Smallprint on device | 0.00 | 0.0000 | runs on the user's laptop or phone; nothing leaves the device |
Assumptions and caveats:
- Workload: 188 clauses per document is the average of the 50 real Terms of Service in LexGLUE UNFAIR-ToS
  (9,414 sentences). 805 input tokens per clause includes the ~416-token system prompt, the same prompt for every option.
- Token counts and speed are measured on our own evaluation run (1,607 LexGLUE clauses), using the merged
  3.35B model on one A100 with plain `transformers` generation, batch 32: 6.3 clauses/s. A serving engine
  such as vLLM would be faster; we did not measure it, so we don't count it.
- The API figure is a lower bound: GPT-5 mini is a reasoning model and bills its reasoning tokens as output,
  which we don't count, and we assume it answers in as few tokens as Smallprint. Prompt caching of the system
  prompt would cut its input cost; batch pricing halves it.
- On device: no marginal cost and no document leaves the device; the cost is the user's own hardware.
