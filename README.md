# Voice Agent Reliability Lab

A local-first replay and release-gating prototype for voice-agent failure modes, with deterministic traces, evidence-backed reports, and an offline dashboard.

`hamming-ai-voice-reliability-lab` favors explicit fixtures, deterministic checks, and reviewable artifacts over hidden services or live data.

## Thesis

Voice Agent Chaos Lab: Production Failure Replay and Regression Firewall.

## Primitives

- Synthetic voice-call traces with latency, interruption, silence, noise, sentiment, and outcome signals.
- Replay-case compilation for recurring failure modes and release regression gates.
- Evidence verification for every generated claim before dashboard export.

## Reproduce locally

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

## Review packet

- `outputs/dashboard.html`
- `outputs/decision_report.md`
- `outputs/evidence_graph.mmd`
- `outputs/risk_or_quality_report.csv`
- `outputs/benchmark.md`
- `outputs/demo_pack.md`

## Confidence checks

```bash
uv run ruff check .
uv run pytest -q
uv run app verify
```

## Data limits

`Voice Agent Reliability Lab` is built for local reproduction: deterministic inputs enter the run, deterministic evidence comes out, and private data stays outside the repo.
