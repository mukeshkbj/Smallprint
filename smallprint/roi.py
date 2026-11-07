"""ROI comparison: a closed-model API vs self-hosting vs on-device Smallprint.

Every price is an input. Nothing is hard-coded, so the table only ever shows
prices you looked up yourself (put the date and source in the config).
Token counts and throughput should come from your own eval run
(results/*.json -> "usage"), not guesses.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Workload:
    documents_per_month: int
    clauses_per_document: float     # measured with document.split_clauses on real policies
    input_tokens_per_clause: float  # includes the system prompt
    output_tokens_per_clause: float


@dataclass
class ClosedAPI:
    name: str
    usd_per_million_input: float
    usd_per_million_output: float
    price_source: str               # URL + date you checked


@dataclass
class SelfHosted:
    gpu_name: str
    usd_per_gpu_hour: float
    output_tokens_per_second: float  # measured, batched, on that GPU
    price_source: str


def monthly(workload: Workload, api: ClosedAPI, gpu: SelfHosted) -> dict:
    clauses = workload.documents_per_month * workload.clauses_per_document
    tin = clauses * workload.input_tokens_per_clause
    tout = clauses * workload.output_tokens_per_clause
    api_cost = tin / 1e6 * api.usd_per_million_input + tout / 1e6 * api.usd_per_million_output
    gpu_hours = tout / gpu.output_tokens_per_second / 3600 if gpu.output_tokens_per_second else float("inf")
    gpu_cost = gpu_hours * gpu.usd_per_gpu_hour
    return {
        "clauses_per_month": clauses,
        "input_tokens": tin, "output_tokens": tout,
        "closed_api_usd": api_cost,
        "self_hosted_gpu_hours": gpu_hours, "self_hosted_usd": gpu_cost,
        "on_device_usd": 0.0,
        "closed_api_usd_per_document": api_cost / workload.documents_per_month if workload.documents_per_month else 0.0,
        "self_hosted_usd_per_document": gpu_cost / workload.documents_per_month if workload.documents_per_month else 0.0,
    }


def to_markdown(workload: Workload, api: ClosedAPI, gpu: SelfHosted, r: dict) -> str:
    return "\n".join([
        f"Workload: {workload.documents_per_month:,} documents/month x {workload.clauses_per_document:.0f} clauses "
        f"= {r['clauses_per_month']:,.0f} clauses ({r['input_tokens']/1e6:,.1f}M input, {r['output_tokens']/1e6:,.1f}M output tokens).",
        "",
        "| Option | Monthly cost (USD) | Per document (USD) | Notes |",
        "|---|---:|---:|---|",
        f"| {api.name} (API) | {r['closed_api_usd']:,.2f} | {r['closed_api_usd_per_document']:.4f} | "
        f"${api.usd_per_million_input}/M in, ${api.usd_per_million_output}/M out; {api.price_source}; documents leave the device |",
        f"| Smallprint self-hosted ({gpu.gpu_name}) | {r['self_hosted_usd']:,.2f} | {r['self_hosted_usd_per_document']:.4f} | "
        f"{r['self_hosted_gpu_hours']:,.1f} GPU h at ${gpu.usd_per_gpu_hour}/h, {gpu.output_tokens_per_second:,.0f} tok/s measured; {gpu.price_source} |",
        "| Smallprint on device | 0.00 | 0.0000 | runs on the user's laptop or phone; nothing leaves the device |",
    ])
