"""Inference backends. Each one maps a list of clause texts to a list of raw
replies, so the same object works in the demo, in DPO negative mining and as
an AuditKIT model (AuditKIT accepts any ``list[str] -> list[str]`` callable).

* ``HFClauseModel``      transformers (+ optional LoRA adapter, 4-bit) - Colab / Space
* ``VLLMClauseModel``    vLLM offline engine - fastest for full test sets
* ``LiteLLMClauseModel`` any hosted model (closed-model reference for the ROI table)
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from .prompts import SYSTEM_PROMPT, chat_messages
from .schema import _first_json_object

DEFAULT_MAX_NEW_TOKENS = 192


def json_complete(texts: Sequence[str]) -> list[bool]:
    """Per sequence: has a complete first JSON object been generated yet? Scoring only ever
    reads the first object, so generation can stop there (base models tend to ramble on)."""
    return [_first_json_object(t) is not None for t in texts]


def _stop_after_json(tok, prompt_len: int, every: int = 4):
    """A transformers StoppingCriteria that ends each sequence once its JSON object closes."""
    import torch
    from transformers import StoppingCriteria

    class StopAfterJSON(StoppingCriteria):
        def __init__(self):
            self.step = 0

        def __call__(self, input_ids, scores, **kwargs):
            self.step += 1
            if self.step % every:
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            texts = tok.batch_decode(input_ids[:, prompt_len:], skip_special_tokens=True)
            return torch.tensor(json_complete(texts), dtype=torch.bool, device=input_ids.device)

    return StopAfterJSON()


def native_bf16() -> bool:
    """True only for GPUs with native bf16 (Ampere and newer). torch.cuda.is_bf16_supported()
    also returns True on a T4 through slow emulation, which is not what we want."""
    import torch
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # older torch without the argument
        return torch.cuda.get_device_capability()[0] >= 8


class _TimedMixin:
    """Tracks generated tokens and wall time for the throughput/ROI numbers."""

    def _reset_stats(self) -> None:
        self.stats = {"calls": 0, "clauses": 0, "output_tokens": 0, "input_tokens": 0, "seconds": 0.0}

    @property
    def tokens_per_second(self) -> float:
        s = self.stats
        return s["output_tokens"] / s["seconds"] if s["seconds"] else 0.0


class HFClauseModel(_TimedMixin):
    def __init__(self, model_id: str, adapter_id: str | None = None, load_in_4bit: bool = False,
                 system_prompt: str = SYSTEM_PROMPT, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                 batch_size: int = 8, device_map: str | None = "auto", token: str | None = None,
                 dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.system_prompt, self.max_new_tokens, self.batch_size = system_prompt, max_new_tokens, batch_size
        self.tok = AutoTokenizer.from_pretrained(model_id, token=token)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw: dict = {"device_map": device_map, "token": token}
        if dtype is not None and not load_in_4bit:
            kw["dtype"] = dtype
        elif load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if native_bf16() else torch.float16,
            )
        else:
            # Tiny Aya ships as bf16. GPUs without native bf16 (T4, V100) emulate it very slowly,
            # so load fp16 there instead.
            if torch.cuda.is_available():
                kw["dtype"] = torch.bfloat16 if native_bf16() else torch.float16
            else:
                kw["dtype"] = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        if adapter_id:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_id, token=token)
        self.model.eval()
        self.name = f"hf:{model_id}" + (f"+{adapter_id}" if adapter_id else "")
        self._reset_stats()

    def __call__(self, clauses: Sequence[str]) -> list[str]:
        import torch

        outs: list[str] = []
        t0 = time.perf_counter()
        for i in range(0, len(clauses), self.batch_size):
            batch = clauses[i : i + self.batch_size]
            prompts = [
                self.tok.apply_chat_template(chat_messages(c, self.system_prompt), tokenize=False,
                                             add_generation_prompt=True)
                for c in batch
            ]
            enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            from transformers import StoppingCriteriaList
            stop = StoppingCriteriaList([_stop_after_json(self.tok, enc["input_ids"].shape[1])])
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id, stopping_criteria=stop)
            new = gen[:, enc["input_ids"].shape[1]:]
            for row in new:
                self.stats["output_tokens"] += int((row != self.tok.pad_token_id).sum())
            self.stats["input_tokens"] += int(enc["attention_mask"].sum())
            outs.extend(self.tok.batch_decode(new, skip_special_tokens=True))
            if (i // self.batch_size) % 5 == 4:  # progress for long eval runs
                el = time.perf_counter() - t0
                print(f"  {len(outs)}/{len(clauses)} clauses, {len(outs) / el:.1f}/s", flush=True)
        self.stats["seconds"] += time.perf_counter() - t0
        self.stats["calls"] += 1
        self.stats["clauses"] += len(clauses)
        return outs


class VLLMClauseModel(_TimedMixin):
    def __init__(self, model_id: str, adapter_path: str | None = None, system_prompt: str = SYSTEM_PROMPT,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.85):
        from vllm import LLM, SamplingParams

        self.llm = LLM(model=model_id, max_model_len=max_model_len, enable_lora=bool(adapter_path),
                       gpu_memory_utilization=gpu_memory_utilization)
        self.params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        self.system_prompt = system_prompt
        self.lora = None
        if adapter_path:
            from vllm.lora.request import LoRARequest
            self.lora = LoRARequest("smallprint", 1, adapter_path)
        self.name = f"vllm:{model_id}" + (f"+{adapter_path}" if adapter_path else "")
        self._reset_stats()

    def __call__(self, clauses: Sequence[str]) -> list[str]:
        t0 = time.perf_counter()
        convs = [chat_messages(c, self.system_prompt) for c in clauses]
        res = self.llm.chat(convs, self.params, lora_request=self.lora, use_tqdm=False)
        outs = [r.outputs[0].text for r in res]
        self.stats["output_tokens"] += sum(len(r.outputs[0].token_ids) for r in res)
        self.stats["input_tokens"] += sum(len(r.prompt_token_ids or []) for r in res)
        self.stats["seconds"] += time.perf_counter() - t0
        self.stats["calls"] += 1
        self.stats["clauses"] += len(clauses)
        return outs


class LiteLLMClauseModel(_TimedMixin):
    """A hosted model with the same prompt, for the closed-model reference column."""

    def __init__(self, model: str, system_prompt: str = SYSTEM_PROMPT,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, concurrency: int = 8):
        self.model, self.system_prompt = model, system_prompt
        self.max_new_tokens, self.concurrency = max_new_tokens, concurrency
        self.name = f"litellm:{model}"
        self._reset_stats()

    def __call__(self, clauses: Sequence[str]) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor

        import litellm

        def one(c: str) -> tuple[str, int, int]:
            r = litellm.completion(model=self.model, messages=chat_messages(c, self.system_prompt),
                                   temperature=0.0, max_tokens=self.max_new_tokens, num_retries=3)
            u = getattr(r, "usage", None)
            return (r.choices[0].message.content or "",
                    getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(self.concurrency) as pool:
            res = list(pool.map(one, clauses))
        self.stats["seconds"] += time.perf_counter() - t0
        self.stats["input_tokens"] += sum(r[1] for r in res)
        self.stats["output_tokens"] += sum(r[2] for r in res)
        self.stats["calls"] += 1
        self.stats["clauses"] += len(clauses)
        return [r[0] for r in res]
