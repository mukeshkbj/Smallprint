"""The teacher and judge must receive the configured endpoint and key (HF router by default)."""

import sys
import types

import pytest

from smallprint.config import load_config
from smallprint.teacher import litellm_completion, resolve_api_key


def test_default_config_uses_hf_router():
    cfg = load_config()
    for section in ("teacher", "judge"):
        s = cfg[section]
        assert s["api_base"] == "https://router.huggingface.co/v1" and s["api_key_env"] == "HF_TOKEN"
        assert s["model"].startswith("openai/")
    assert cfg["teacher"]["model"].split("/")[1] != cfg["judge"]["model"].split("/")[1]  # different families


def test_resolve_api_key(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    assert resolve_api_key({"api_key_env": "HF_TOKEN", "model": "m"}) == "hf_test"
    assert resolve_api_key({"model": "m"}) is None
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(SystemExit):
        resolve_api_key({"api_key_env": "HF_TOKEN", "model": "m"})


def test_litellm_completion_forwards_endpoint(monkeypatch):
    seen = {}

    def fake_completion(**kw):
        seen.update(kw)
        msg = types.SimpleNamespace(content='{"evidence": "x", "explanation": "y"}')
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=fake_completion))
    fn = litellm_completion("openai/CohereLabs/aya-expanse-32b:cohere",
                            api_base="https://router.huggingface.co/v1", api_key="hf_test")
    assert fn("hello").startswith("{")
    assert seen["model"] == "openai/CohereLabs/aya-expanse-32b:cohere"
    assert seen["api_base"] == "https://router.huggingface.co/v1" and seen["api_key"] == "hf_test"


def test_grounding_gate_passes_judge_key(monkeypatch, tmp_path):
    from smallprint import curate

    captured = {}

    class FakeConfig:
        def __init__(self, **kw):
            captured.update(kw)

    class FakeCurator:
        def __init__(self, cfg):
            pass

        def run(self):
            return types.SimpleNamespace(passed=[], rejected=[])

    monkeypatch.setattr(curate, "_curator", lambda: (FakeCurator, FakeConfig))
    curate.grounding_gate(tmp_path / "in.jsonl", tmp_path / "out", "openai/Qwen/Qwen3-235B-A22B-Instruct-2507",
                          0.7, "https://router.huggingface.co/v1", 4, "hf_test")
    assert captured["llm_api_key"] == "hf_test"
    assert captured["judge_llm_api_base"] == captured["llm_api_base"] == "https://router.huggingface.co/v1"
    assert captured["judge_llm_model"] == "openai/Qwen/Qwen3-235B-A22B-Instruct-2507"
    assert captured["hallucination_threshold"] == 0.7
