#!/usr/bin/env python
"""Step 6: ROI table (closed API vs self-hosted vs on device).

Token counts per clause and throughput come from your eval run
(results/usage.json). Prices come from configs/smallprint.yaml -> roi, which
you fill in from today's price pages (with the source and date).

Usage:
  python scripts/06_roi.py --measure policies/*.txt     # clauses per real policy
  python scripts/06_roi.py --clauses-per-doc 180 --tps 900 --closed-run reference__lexglue
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smallprint.config import load_config  # noqa: E402
from smallprint.document import split_clauses  # noqa: E402
from smallprint.roi import ClosedAPI, SelfHosted, Workload, monthly, to_markdown  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--measure", nargs="*", help="plain-text policies to count clauses in")
    ap.add_argument("--clauses-per-doc", type=float)
    ap.add_argument("--run", default="final__lexglue", help="usage key for token counts and throughput")
    ap.add_argument("--tps", type=float, help="override measured output tokens/sec (e.g. batched vLLM)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    r = cfg["roi"]

    if args.measure:
        counts = [len(split_clauses(Path(p).read_text(encoding="utf-8"))) for p in args.measure]
        print(f"clauses per document over {len(counts)} files: mean {sum(counts)/len(counts):.1f}, "
              f"min {min(counts)}, max {max(counts)}")
        return

    usage = json.loads((Path(cfg["eval"]["out_dir"]) / "usage.json").read_text())[args.run]
    cpd = args.clauses_per_doc or r.get("clauses_per_document")
    missing = [k for k in ("closed_api_name", "closed_api_usd_per_million_input", "closed_api_usd_per_million_output",
                           "closed_api_price_source", "gpu_name", "usd_per_gpu_hour", "gpu_price_source")
               if r.get(k) in (None, "")]
    if not cpd:
        missing.append("clauses_per_document (or --clauses-per-doc)")
    if missing:
        raise SystemExit("Fill these in configs/smallprint.yaml -> roi first: " + ", ".join(missing))

    w = Workload(r["documents_per_month"], cpd, usage["input_tokens_per_clause"], usage["output_tokens_per_clause"])
    api = ClosedAPI(r["closed_api_name"], r["closed_api_usd_per_million_input"],
                    r["closed_api_usd_per_million_output"], r["closed_api_price_source"])
    gpu = SelfHosted(r["gpu_name"], r["usd_per_gpu_hour"], args.tps or usage["output_tokens_per_second"],
                     r["gpu_price_source"])
    md = to_markdown(w, api, gpu, monthly(w, api, gpu))
    out = Path(cfg["eval"]["out_dir"]) / "roi.md"
    out.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
