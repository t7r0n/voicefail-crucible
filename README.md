# Voice Agent Reliability Lab

A local-first replay and release-gating prototype for voice-agent failure modes, with deterministic traces, evidence-backed reports, and an offline dashboard.

## Features

- Synthetic voice-call traces with latency, interruption, silence, noise, sentiment, and outcome signals.
- Replay-case compilation for recurring failure modes and release regression gates.
- Evidence verification for every generated claim before dashboard export.

## Run Locally

```bash
uv sync
uv run app init-demo
uv run app ingest fixtures/
uv run app analyze
uv run app verify
uv run app dashboard
uv run app benchmark
uv run app export-demo-pack
uv run pytest -q
uv run ruff check .
```

## Outputs

- `outputs/dashboard.html`
- `outputs/decision_report.md`
- `outputs/evidence_graph.mmd`
- `outputs/risk_or_quality_report.csv`
- `outputs/benchmark.md`
- `outputs/demo_pack.md`

## Data Policy

This project runs fully locally on deterministic synthetic fixtures. It does not require external APIs, credentials, private datasets, network access, or production systems.
