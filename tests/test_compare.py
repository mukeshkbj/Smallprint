import importlib.util
import json
from pathlib import Path

from smallprint.metrics import HEADLINE

spec = importlib.util.spec_from_file_location("compare", Path(__file__).parents[1] / "scripts" / "14_compare.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def _write(folder: Path, run: str, test_set: str, value: float, langs=()):
    d = {k: value for k, _ in HEADLINE}
    if langs:
        d["per_language"] = {lang: {k: value for k, _ in HEADLINE} for lang in langs}
    (folder / f"{run}__{test_set}.json").write_text(json.dumps(d))


def test_compare_pending_and_full(tmp_path, monkeypatch):
    water, glob = tmp_path / "w", tmp_path / "g"
    water.mkdir(), glob.mkdir()
    for run, v in (("base", 0.2), ("sft", 0.8), ("final", 0.7)):
        _write(water, run, "lexglue", v)
        _write(water, run, "claudette_ml", v, langs=("de", "en"))
    out = tmp_path / "c.md"
    monkeypatch.setattr("sys.argv", ["x", "--water", str(water), "--variant", f"global={glob}", "--out", str(out)])
    compare.main()
    text = out.read_text()
    assert "Pending: only water" in text and "Pending: run scripts/12_safety_eval.py" in text
    assert "| SFT | 0.800 |" in text

    for run, v in (("base", 0.3), ("sft", 0.75)):
        _write(glob, run, "lexglue", v)
        _write(glob, run, "claudette_ml", v, langs=("de", "en"))
    (water / "safety").mkdir()
    (water / "safety" / "summary.md").write_text("# Safety evaluation\n\n| a | b |\n")
    compare.main()
    text = out.read_text()
    assert "| Global | 0.300 | 0.750 |" in text
    assert "| DE | 0.200 | 0.800 | 0.300 | 0.750 |" in text
    assert "## 3. Safety degradation" in text and "| a | b |" in text
