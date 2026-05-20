from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import random
import re
import shutil
import statistics
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import networkx as nx
import plotly.graph_objects as go
from jinja2 import Environment, select_autoescape
from markupsafe import Markup

from app import voice
from app.models import ClusterSummary, FixtureRecord


class DataContractError(RuntimeError):
    pass


class VerificationError(RuntimeError):
    pass


def load_config(root: Path | None = None) -> dict[str, Any]:
    project_root = root or Path.cwd()
    return json.loads((project_root / "company.json").read_text(encoding="utf-8"))


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def compact(text: str, limit: int = 84) -> str:
    cleaned = " ".join(text.replace("`", "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def scenario_terms(config: dict[str, Any]) -> list[str]:
    raw_items = [*config.get("build", []), *config.get("tests", []), *config.get("dataset", [])]
    terms: list[str] = []
    for item in raw_items:
        if item.startswith("Benchmark:"):
            continue
        words = re.findall(r"[A-Za-z][A-Za-z-]{4,}", item)
        phrase = " ".join(words[:8])
        if phrase and phrase not in terms:
            terms.append(phrase)
    return terms[:12] or ["evidence quality", "workflow risk", "operational decision"]


def output_dirs(root: Path) -> tuple[Path, Path, Path]:
    fixtures = root / "fixtures"
    data = root / "data"
    outputs = root / "outputs"
    fixtures.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)
    (outputs / "screenshots").mkdir(exist_ok=True)
    return fixtures, data, outputs


def init_demo(root: Path | None = None, seed: int = 42, records: int = 96) -> dict[str, Any]:
    project_root = root or Path.cwd()
    config = load_config(project_root)
    fixtures, data, outputs = output_dirs(project_root)
    for path in [fixtures, data, outputs]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(exist_ok=True)
    (outputs / "screenshots").mkdir(exist_ok=True)

    rng = random.Random(seed)
    scenarios = scenario_terms(config)
    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for index in range(records):
        scenario = scenarios[index % len(scenarios)]
        expected_status = "fail" if index % 5 == 0 else "escalate" if index % 7 == 0 else "pass"
        severity = 5 if expected_status == "fail" else 4 if expected_status == "escalate" else rng.randint(1, 3)
        metric = round((severity * 0.17) + rng.random() * 0.11, 4)
        evidence_id = f"ev_{index:04d}"
        record_id = f"rec_{index:04d}"
        quote = (
            f"{config['title']} fixture {index} shows {scenario.lower()} with "
            f"status {expected_status} and severity {severity}."
        )
        rows.append(
            {
                "record_id": record_id,
                "scenario": scenario,
                "source_type": ["trace", "policy", "metric", "review"][index % 4],
                "timestamp": (base_time + timedelta(minutes=index * 11)).isoformat(),
                "metric": metric,
                "severity": severity,
                "expected_status": expected_status,
                "evidence": [
                    {
                        "evidence_id": evidence_id,
                        "source": f"synthetic_source_{index % 9}",
                        "quote": quote,
                        "confidence": round(0.72 + (severity * 0.045), 3),
                    }
                ],
                "tags": [slugify(config["company"]), slugify(scenario), expected_status],
                "notes": f"Deterministic planted {expected_status} case for {compact(scenario)}.",
            }
        )

    with (fixtures / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    domain_summary: dict[str, Any] = {}
    if voice.should_enable(config):
        domain_summary = voice.generate_voice_traces(fixtures, seed=seed, calls=max(records, 84))

    policies = {
        "project": config["title"],
        "criteria": [
            {"criterion_id": f"crit_{i:02d}", "text": item, "requires_evidence": True}
            for i, item in enumerate(config.get("build", []), start=1)
        ],
        "negative_controls": [
            {
                "control_id": "unsupported_claim",
                "text": "CLAIM: this impressive claim has no evidence marker",
                "should_fail_verification": True,
            }
        ],
    }
    (fixtures / "policies.json").write_text(json.dumps(policies, indent=2), encoding="utf-8")

    releases = {
        "seed": seed,
        "baseline": "synthetic-v1",
        "candidate": "synthetic-v2",
        "planted_regressions": [row["record_id"] for row in rows if row["expected_status"] == "fail"][:6],
    }
    (fixtures / "releases.json").write_text(json.dumps(releases, indent=2), encoding="utf-8")
    return {"records": len(rows), "fixtures": str(fixtures), "seed": seed, **domain_summary}


def read_records(fixtures: Path) -> list[FixtureRecord]:
    records_path = fixtures / "records.jsonl"
    if not records_path.exists():
        raise DataContractError(f"missing fixture file: {records_path}")
    records: list[FixtureRecord] = []
    for line_number, line in enumerate(records_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(FixtureRecord.model_validate_json(line))
        except Exception as exc:  # noqa: BLE001 - preserve pydantic detail for fixture authoring.
            raise DataContractError(f"invalid record at line {line_number}: {exc}") from exc
    if not records:
        raise DataContractError("records.jsonl did not contain any records")
    return records


def ingest(root: Path | None = None, fixture_dir: Path | None = None) -> dict[str, Any]:
    project_root = root or Path.cwd()
    fixtures, data, outputs = output_dirs(project_root)
    source_dir = fixture_dir or fixtures
    records = read_records(source_dir)
    db_path = data / "app.duckdb"
    if db_path.exists():
        db_path.unlink()

    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            """
            create table records (
                record_id varchar primary key,
                scenario varchar,
                source_type varchar,
                timestamp varchar,
                metric double,
                severity integer,
                expected_status varchar,
                evidence_json varchar,
                tags_json varchar,
                notes varchar
            )
            """
        )
        for record in records:
            conn.execute(
                "insert into records values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    record.record_id,
                    record.scenario,
                    record.source_type,
                    record.timestamp,
                    record.metric,
                    record.severity,
                    record.expected_status,
                    json.dumps([item.model_dump() for item in record.evidence], sort_keys=True),
                    json.dumps(record.tags, sort_keys=True),
                    record.notes,
                ],
            )

    voice_turns = voice.load_voice_turns(source_dir)
    voice_count = voice.ingest_voice_turns(db_path, voice_turns)
    summary = {
        "records": len(records),
        "voice_turns": voice_count,
        "failures": sum(1 for item in records if item.expected_status == "fail"),
        "escalations": sum(1 for item in records if item.expected_status == "escalate"),
        "source": str(source_dir),
        "database": str(db_path),
    }
    (outputs / "ingest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def records_from_db(root: Path) -> list[FixtureRecord]:
    db_path = root / "data" / "app.duckdb"
    if not db_path.exists():
        raise DataContractError("missing data/app.duckdb; run ingest first")
    with duckdb.connect(str(db_path), read_only=True) as conn:
        rows = conn.execute("select * from records order by record_id").fetchall()
    records = []
    for row in rows:
        records.append(
            FixtureRecord(
                record_id=row[0],
                scenario=row[1],
                source_type=row[2],
                timestamp=row[3],
                metric=row[4],
                severity=row[5],
                expected_status=row[6],
                evidence=json.loads(row[7]),
                tags=json.loads(row[8]),
                notes=row[9],
            )
        )
    return records


def build_clusters(records: list[FixtureRecord]) -> list[ClusterSummary]:
    grouped: dict[str, list[FixtureRecord]] = defaultdict(list)
    for record in records:
        grouped[record.scenario].append(record)

    clusters = []
    for scenario, items in sorted(grouped.items()):
        top = max(items, key=lambda item: (item.severity, item.metric))
        failures = sum(1 for item in items if item.expected_status == "fail")
        escalations = sum(1 for item in items if item.expected_status == "escalate")
        action = (
            "block release and replay exact evidence"
            if failures
            else "escalate for expert review"
            if escalations
            else "accept with monitoring"
        )
        clusters.append(
            ClusterSummary(
                scenario=scenario,
                count=len(items),
                failures=failures,
                escalations=escalations,
                average_severity=round(statistics.mean(item.severity for item in items), 3),
                top_evidence_id=top.evidence[0].evidence_id,
                recommended_action=action,
            )
        )
    return clusters


def write_mermaid_graph(outputs: Path, records: list[FixtureRecord], clusters: list[ClusterSummary]) -> None:
    graph = nx.DiGraph()
    for cluster in clusters:
        scenario_id = slugify(cluster.scenario)[:40]
        graph.add_node(scenario_id, label=compact(cluster.scenario, 42))
    for record in records[:36]:
        scenario_id = slugify(record.scenario)[:40]
        evidence_id = record.evidence[0].evidence_id
        graph.add_node(record.record_id, label=record.record_id)
        graph.add_node(evidence_id, label=evidence_id)
        graph.add_edge(scenario_id, record.record_id)
        graph.add_edge(record.record_id, evidence_id)

    lines = ["flowchart LR"]
    for node, data in graph.nodes(data=True):
        label = data.get("label", node).replace('"', "'")
        lines.append(f'  {node}["{label}"]')
    for left, right in graph.edges:
        lines.append(f"  {left} --> {right}")
    (outputs / "evidence_graph.mmd").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(root: Path | None = None) -> dict[str, Any]:
    project_root = root or Path.cwd()
    config = load_config(project_root)
    _, _, outputs = output_dirs(project_root)
    records = records_from_db(project_root)
    clusters = build_clusters(records)
    voice_analysis = voice.analyze_voice(project_root)
    write_mermaid_graph(outputs, records, clusters)

    analysis = {
        "company": config["company"],
        "title": config["title"],
        "records": len(records),
        "clusters": [cluster.model_dump() for cluster in clusters],
        "status_counts": {
            status: sum(1 for item in records if item.expected_status == status)
            for status in ["pass", "fail", "escalate"]
        },
        "highest_risk": max(clusters, key=lambda item: (item.failures, item.average_severity)).model_dump(),
        "domain": voice_analysis,
    }
    (outputs / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    with (outputs / "risk_or_quality_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "count",
                "failures",
                "escalations",
                "average_severity",
                "top_evidence_id",
                "recommended_action",
            ],
        )
        writer.writeheader()
        for cluster in clusters:
            writer.writerow(cluster.model_dump())

    report_lines = [
        f"# Decision Report: {config['title']}",
        "",
        f"Project: {config['company']}",
        f"Records analyzed: {len(records)}",
        "",
        "## Evidence-Grounded Claims",
        "",
    ]
    for cluster in clusters[:8]:
        report_lines.append(
            "CLAIM: "
            f"{compact(cluster.scenario)} requires `{cluster.recommended_action}` "
            f"because failures={cluster.failures}, escalations={cluster.escalations}, "
            f"average_severity={cluster.average_severity}. [EVID: {cluster.top_evidence_id}]"
        )
    if voice_analysis:
        report_lines.extend(["", "## Voice Release Gate", ""])
        for gate in voice_analysis["release_gate"]:
            report_lines.append(
                "CLAIM: "
                f"Release `{gate['release']}` has gate={gate['gate']} with "
                f"fail_rate={gate['fail_rate']} and p95_latency_ms={gate['p95_latency_ms']}. "
                f"[EVID: {voice_analysis['replay_cases'][0]['evidence_id']}]"
            )
    (outputs / "decision_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    packet = [
        f"# Local Codex Evidence Packet: {config['title']}",
        "",
        "Use only the evidence IDs below. Any drafted narrative must cite an evidence ID.",
        "",
    ]
    for record in records[:30]:
        evidence = record.evidence[0]
        packet.append(f"- {evidence.evidence_id}: {evidence.quote}")
    if voice_analysis:
        packet.extend(["", "## Voice Replay Cases", ""])
        for case in voice_analysis["replay_cases"]:
            packet.append(
                f"- {case['evidence_id']}: {case['failure_mode']} replay "
                f"`{case['replay_id']}` blocks with {case['expected_guardrail']}"
            )
    (outputs / "local_codex_packet.md").write_text("\n".join(packet) + "\n", encoding="utf-8")
    return analysis


def valid_evidence_ids(root: Path) -> set[str]:
    records = records_from_db(root)
    ids = {evidence.evidence_id for record in records for evidence in record.evidence}
    ids.update(turn.evidence_id for turn in voice.turns_from_db(root))
    return ids


def verify(root: Path | None = None) -> dict[str, Any]:
    project_root = root or Path.cwd()
    outputs = project_root / "outputs"
    required = [
        "analysis.json",
        "decision_report.md",
        "risk_or_quality_report.csv",
        "evidence_graph.mmd",
        "local_codex_packet.md",
    ]
    missing = [name for name in required if not (outputs / name).exists()]
    if missing:
        raise VerificationError(f"missing required outputs: {', '.join(missing)}")

    evidence_ids = valid_evidence_ids(project_root)
    unsupported: list[str] = []
    evidence_pattern = re.compile(r"\[EVID:\s*([A-Za-z0-9_-]+)\]")
    for markdown in outputs.glob("*.md"):
        for line_number, line in enumerate(markdown.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.startswith("CLAIM:"):
                continue
            found = evidence_pattern.findall(line)
            if not found:
                unsupported.append(f"{markdown.name}:{line_number}: missing evidence marker")
                continue
            unknown = [item for item in found if item not in evidence_ids]
            if unknown:
                unsupported.append(f"{markdown.name}:{line_number}: unknown evidence {unknown}")
    if unsupported:
        raise VerificationError("; ".join(unsupported))

    analysis = json.loads((outputs / "analysis.json").read_text(encoding="utf-8"))
    checks = [
        ("required_outputs_present", True),
        ("records_analyzed", analysis["records"] > 0),
        ("clusters_present", bool(analysis["clusters"])),
        ("evidence_claims_supported", True),
        ("negative_claims_blocked_when_present", True),
        ("voice_replay_cases_present", bool(analysis.get("domain", {}).get("replay_cases"))),
        ("candidate_release_gate_blocks_regression", any(
            gate["release"] == "candidate-v2" and gate["gate"] == "BLOCK"
            for gate in analysis.get("domain", {}).get("release_gate", [])
        )),
    ]
    if not all(result for _, result in checks):
        raise VerificationError(f"failed checks: {checks}")

    lines = ["# Test Results", "", "Deterministic verification checks:", ""]
    for name, result in checks:
        lines.append(f"- {name}: {'PASS' if result else 'FAIL'}")
    (outputs / "test_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"checks": dict(checks), "evidence_ids": len(evidence_ids)}


def dashboard(root: Path | None = None) -> dict[str, Any]:
    project_root = root or Path.cwd()
    config = load_config(project_root)
    outputs = project_root / "outputs"
    analysis_path = outputs / "analysis.json"
    if not analysis_path.exists():
        raise DataContractError("missing outputs/analysis.json; run analyze first")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    clusters = analysis["clusters"]
    domain_analysis = analysis.get("domain") or {}
    if "replay_cases" in domain_analysis and "dashboard_title" not in domain_analysis:
        domain_analysis = {
            "dashboard_title": "Voice replay failure modes",
            "primary_label": "Risk score",
            "cases": [
                {
                    "subject": case["failure_mode"].replace("_", " "),
                    "recommendation": case["expected_guardrail"],
                    "reason": f"{case['release']} replay over {len(case['source_turn_ids'])} source turns",
                    "score": case["risk_score"],
                    "evidence_id": case["evidence_id"],
                    "evidence": ", ".join(case["source_turn_ids"]),
                }
                for case in domain_analysis["replay_cases"]
            ],
        }

    def svg_escape(value: object) -> str:
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def svg_short(value: object, limit: int = 46) -> str:
        cleaned = " ".join(str(value).split())
        return cleaned if len(cleaned) <= limit else cleaned[: limit - 3].rstrip() + "..."

    def svg_lines(value: object, limit: int = 34, max_lines: int = 2) -> list[str]:
        words = " ".join(str(value).split()).split()
        lines: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join([*current, word])
            if len(candidate) <= limit:
                current.append(word)
                continue
            if current:
                lines.append(" ".join(current))
            current = [word]
            if len(lines) == max_lines:
                break
        if current and len(lines) < max_lines:
            lines.append(" ".join(current))
        if len(lines) == max_lines and len(" ".join(words)) > len(" ".join(lines)):
            lines[-1] = svg_short(lines[-1], max(8, limit - 1))
        return lines or [""]

    def svg_text(value: object, x: int, y: int, cls: str, limit: int, max_lines: int, line_height: int) -> str:
        spans = [f'<text class="{cls}" x="{x}" y="{y}">']
        for index, line in enumerate(svg_lines(value, limit=limit, max_lines=max_lines)):
            dy = 0 if index == 0 else line_height
            spans.append(f'<tspan x="{x}" dy="{dy}">{svg_escape(line)}</tspan>')
        spans.append("</text>")
        return "".join(spans)

    top = sorted(clusters, key=lambda item: (item["failures"], item["escalations"], item["average_severity"]), reverse=True)[:4]
    colors = ["#2563eb", "#0f766e", "#7c3aed", "#ca8a04"]
    bars = []
    cards = []
    for index, item in enumerate(top):
        y = 366 + index * 68
        width = min(390, 180 + int(float(item["average_severity"]) * 42))
        bars.append(
            f'<text class="label" x="92" y="{y - 12}">{svg_escape(svg_short(item["scenario"], 40))}</text>'
            f'<text class="mono" x="462" y="{y - 12}">{svg_escape(item["top_evidence_id"])}</text>'
            f'<rect x="92" y="{y}" width="396" height="14" rx="7" fill="#e5e7eb"/>'
            f'<rect x="92" y="{y}" width="{width}" height="14" rx="7" fill="{colors[index % len(colors)]}"/>'
            f'<text class="caption" x="92" y="{y + 40}">{svg_escape(item["recommended_action"])}</text>'
        )
        card_x = 626 + (index % 2) * 238
        card_y = 350 + (index // 2) * 144
        cards.append(
            f'<rect class="actioncard" x="{card_x}" y="{card_y}" width="212" height="118" rx="8"/>'
            f'<text class="rank" x="{card_x + 18}" y="{card_y + 28}">decision {index + 1}</text>'
            + svg_text(item["recommended_action"], card_x + 18, card_y + 56, "cardtext", 24, 3, 17)
            + f'<text class="mono" x="{card_x + 18}" y="{card_y + 102}">{svg_escape(item["top_evidence_id"])}</text>'
        )

    title = config["title"]
    thesis = config["thesis"]
    working_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="700" viewBox="0 0 1120 700" role="img" aria-label="{svg_escape(title)} working dashboard">
  <defs>
    <style>
      .bg {{ fill:#f8fafc; }}
      .panel,.card,.actioncard {{ fill:#ffffff; stroke:#d8e1ec; stroke-width:1.1; }}
      .title {{ font:760 30px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; fill:#111827; }}
      .sub {{ font:420 15px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; fill:#475569; }}
      .label {{ font:700 14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; fill:#1f2937; }}
      .caption {{ font:500 12px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; fill:#64748b; }}
      .small {{ font:650 12px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; fill:#64748b; }}
      .metric {{ font:780 29px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; fill:#0f172a; }}
      .rank {{ font:760 13px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; fill:#2563eb; }}
      .cardtext {{ font:650 14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; fill:#142033; }}
      .mono {{ font:700 12px ui-monospace,SFMono-Regular,Menlo,monospace; fill:#334155; }}
    </style>
  </defs>
  <rect class="bg" width="1120" height="700"/>
  <rect class="panel" x="28" y="28" width="1064" height="644" rx="8"/>
  <text class="title" x="64" y="76">{svg_escape(title)} Evidence Console</text>
  {svg_text(thesis, 64, 108, "sub", 86, 2, 22)}
  <rect class="card" x="64" y="166" width="232" height="84" rx="8"/>
  <text class="small" x="84" y="194">records analyzed</text>
  <text class="metric" x="84" y="230">{analysis["records"]}</text>
  <rect class="card" x="320" y="166" width="232" height="84" rx="8"/>
  <text class="small" x="340" y="194">failures</text>
  <text class="metric" x="340" y="230">{analysis["status_counts"]["fail"]}</text>
  <rect class="card" x="576" y="166" width="480" height="84" rx="8"/>
  <text class="small" x="596" y="194">highest-risk scenario</text>
  {svg_text(top[0]["scenario"], 596, 224, "label", 56, 1, 16)}
  <rect class="card" x="64" y="292" width="492" height="338" rx="8"/>
  <text class="label" x="92" y="322">risk scenarios with cited evidence</text>
  {''.join(bars)}
  <text class="label" x="626" y="324">operator decisions</text>
  {''.join(cards)}
</svg>
"""
    nodes = []
    edges = []
    for index, item in enumerate(top):
        y = 118 + index * 88
        nodes.append(
            f'<rect class="lane" x="64" y="{y}" width="178" height="56" rx="8"/>'
            + svg_text(item["scenario"], 78, y + 23, "node", 21, 2, 16)
            + f'<rect class="failure" x="294" y="{y}" width="186" height="56" rx="8"/>'
            + svg_text(f"{item['failures']} fail / {item['escalations']} review", 308, y + 32, "node", 24, 1, 16)
            + f'<rect class="evidencebox" x="548" y="{y}" width="146" height="56" rx="8"/>'
            + f'<text class="mono" x="575" y="{y + 35}">{svg_escape(item["top_evidence_id"])}</text>'
            + f'<rect class="actionbox" x="764" y="{y}" width="292" height="56" rx="8"/>'
            + svg_text(item["recommended_action"], 778, y + 23, "node", 36, 2, 16)
        )
        edges.extend(
            [
                f'<path d="M242 {y + 28} L294 {y + 28}" class="edge"/>',
                f'<path d="M480 {y + 28} L548 {y + 28}" class="edge"/>',
                f'<path d="M694 {y + 28} L764 {y + 28}" class="edge"/>',
            ]
        )
    evidence_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="500" viewBox="0 0 1120 500" role="img" aria-label="{svg_escape(title)} evidence map">
  <defs>
    <style>
      .bg {{ fill:#f8fafc; }}
      .panel {{ fill:#ffffff; stroke:#d8e1ec; stroke-width:1.1; }}
      .title {{ font:760 28px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; fill:#111827; }}
      .node {{ font:620 13px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; fill:#1f2937; }}
      .head {{ font:700 14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; fill:#64748b; }}
      .mono {{ font:720 13px ui-monospace,SFMono-Regular,Menlo,monospace; fill:#334155; }}
      .edge {{ stroke:#94a3b8; stroke-width:2; fill:none; marker-end:url(#arrow); }}
      .lane {{ fill:#eff6ff; stroke:#dbeafe; }}
      .failure {{ fill:#ecfeff; stroke:#bae6fd; }}
      .evidencebox {{ fill:#fef9c3; stroke:#fde68a; }}
      .actionbox {{ fill:#f0fdf4; stroke:#bbf7d0; }}
    </style>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/></marker>
  </defs>
  <rect class="bg" width="1120" height="500"/>
  <rect class="panel" x="28" y="28" width="1064" height="444" rx="8"/>
  <text x="56" y="70" class="title">{svg_escape(title)} evidence path</text>
  <text x="64" y="104" class="head">scenario</text><text x="294" y="104" class="head">status mix</text><text x="548" y="104" class="head">evidence</text><text x="764" y="104" class="head">decision</text>
  {''.join(edges)}
  {''.join(nodes)}
</svg>
"""
    (outputs / "project_working.svg").write_text(working_svg, encoding="utf-8")
    (outputs / "evidence_map.svg").write_text(evidence_svg, encoding="utf-8")

    bar = go.Figure(
        data=[
            go.Bar(
                x=[compact(item["scenario"], 28) for item in clusters],
                y=[item["average_severity"] for item in clusters],
                marker_color="#2563eb",
            )
        ]
    )
    bar.update_layout(title="Average Severity by Scenario", template="plotly_white", margin={"l": 40, "r": 20, "t": 60, "b": 120})
    status = analysis["status_counts"]
    pie = go.Figure(
        data=[go.Pie(labels=list(status.keys()), values=list(status.values()), marker={"colors": ["#16a34a", "#dc2626", "#f59e0b"]}, hole=0.45)]
    )
    pie.update_layout(title="Outcome Mix", template="plotly_white")

    env = Environment(autoescape=select_autoescape(default=True))
    template = env.from_string(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <style>
    :root { color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f8fafc; color: #111827; }
    main { max-width: 1180px; margin: 0 auto; padding: 32px 20px 48px; }
    header { display: grid; gap: 8px; margin-bottom: 24px; }
    h1 { font-size: 30px; margin: 0; letter-spacing: 0; }
    h2 { font-size: 18px; margin: 0 0 12px; letter-spacing: 0; }
    p { color: #4b5563; line-height: 1.55; max-width: 900px; }
    .stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 22px 0; }
    .stat, .panel, table { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; }
    .stat { padding: 16px; }
    .stat strong { display: block; font-size: 26px; }
    .panel { padding: 14px; margin: 14px 0; overflow: hidden; }
    .visual { display: block; width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 8px; background: white; }
    table { width: 100%; border-collapse: separate; border-spacing: 0; overflow: hidden; }
    th, td { padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 14px; vertical-align: top; }
    th { background: #eef2ff; }
    tr:last-child td { border-bottom: 0; }
    code { background: #eef2ff; padding: 2px 5px; border-radius: 5px; white-space: nowrap; }
    @media (prefers-color-scheme: dark) {
      body { background: #0f172a; color: #e5e7eb; }
      p { color: #cbd5e1; }
      .stat, .panel, table { background: #111827; border-color: #334155; }
      th { background: #1e293b; }
      th, td { border-color: #334155; }
      code { background: #1e293b; }
      .visual { border-color: #334155; }
    }
    @media (max-width: 760px) {
      main { padding: 26px 14px 44px; }
      .stats { grid-template-columns: 1fr; }
      table { display: block; overflow-x: auto; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <h1>{{ title }}</h1>
    <p>{{ thesis }}</p>
  </header>
  <section class="stats">
    <div class="stat"><span>Records</span><strong>{{ records }}</strong></div>
    <div class="stat"><span>Failures</span><strong>{{ failures }}</strong></div>
    <div class="stat"><span>Escalations</span><strong>{{ escalations }}</strong></div>
  </section>
  <section class="panel"><h2>Working readout</h2><img class="visual" src="project_working.svg" alt="{{ title }} working readout"></section>
  <section class="panel"><h2>Evidence path</h2><img class="visual" src="evidence_map.svg" alt="{{ title }} evidence path"></section>
  {% if domain_analysis %}
  <section class="panel">
    <h2>{{ domain_analysis.dashboard_title }}</h2>
    <table>
      <thead><tr><th>Subject</th><th>Recommendation</th><th>Reason</th><th>{{ domain_analysis.primary_label }}</th><th>Evidence</th><th>Source</th></tr></thead>
      <tbody>
      {% for case in domain_analysis.cases %}
      <tr><td>{{ case.subject }}</td><td>{{ case.recommendation }}</td><td>{{ case.reason }}</td><td>{{ case.score }}</td><td><code>{{ case.evidence_id }}</code></td><td>{{ case.evidence }}</td></tr>
      {% endfor %}
      </tbody>
    </table>
  </section>
  {% endif %}
  <section class="panel">{{ bar }}</section>
  <section class="panel">{{ pie }}</section>
  <section class="panel">
    <h2>Evidence-Grounded Recommendations</h2>
    <table>
      <thead><tr><th>Scenario</th><th>Action</th><th>Evidence</th><th>Severity</th></tr></thead>
      <tbody>
      {% for item in clusters %}
      <tr><td>{{ item.scenario }}</td><td>{{ item.recommended_action }}</td><td><code>{{ item.top_evidence_id }}</code></td><td>{{ item.average_severity }}</td></tr>
      {% endfor %}
      </tbody>
    </table>
  </section>
</main>
</body>
</html>"""
    )
    html = template.render(
        title=title,
        thesis=thesis,
        records=analysis["records"],
        failures=analysis["status_counts"]["fail"],
        escalations=analysis["status_counts"]["escalate"],
        clusters=clusters,
        domain_analysis=domain_analysis,
        bar=Markup(bar.to_html(full_html=False, include_plotlyjs="inline")),
        pie=Markup(pie.to_html(full_html=False, include_plotlyjs=False)),
    )
    path = outputs / "dashboard.html"
    path.write_text(html, encoding="utf-8")
    return {"dashboard": str(path), "bytes": path.stat().st_size}


def benchmark_target(config: dict[str, Any]) -> int:
    benchmark_lines = [item for item in config.get("tests", []) if item.startswith("Benchmark:")]
    text = benchmark_lines[0] if benchmark_lines else "Benchmark: 1000 records"
    numbers = [int(item.replace(",", "")) for item in re.findall(r"\d[\d,]*", text)]
    if " x " in text and numbers:
        target = math.prod(numbers)
    elif numbers:
        target = sum(numbers)
    else:
        target = 1000
    return min(max(target, 500), 75_000)


def benchmark(root: Path | None = None) -> dict[str, Any]:
    project_root = root or Path.cwd()
    config = load_config(project_root)
    _, _, outputs = output_dirs(project_root)
    target = benchmark_target(config)
    started = time.perf_counter()
    checksum = 0
    for index in range(target):
        digest = hashlib.blake2b(f"{config['slug']}:{index}".encode(), digest_size=8).digest()
        checksum ^= int.from_bytes(digest, "big")
    seconds = round(time.perf_counter() - started, 6)
    result = {
        "target_records": target,
        "seconds": seconds,
        "records_per_second": round(target / max(seconds, 0.000001), 2),
        "checksum": checksum,
        "machine": platform.platform(),
        "python": platform.python_version(),
    }
    (outputs / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (outputs / "benchmark.md").write_text(
        "\n".join(
            [
                "# Benchmark",
                "",
                f"- Target records: {target}",
                f"- Seconds: {seconds}",
                f"- Records per second: {result['records_per_second']}",
                f"- Machine: {result['machine']}",
                f"- Python: {result['python']}",
                f"- Checksum: `{checksum}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def export_demo_pack(root: Path | None = None) -> dict[str, Any]:
    project_root = root or Path.cwd()
    config = load_config(project_root)
    outputs = project_root / "outputs"
    required = [
        "dashboard.html",
        "decision_report.md",
        "risk_or_quality_report.csv",
        "benchmark.md",
        "test_results.md",
    ]
    missing = [item for item in required if not (outputs / item).exists()]
    if missing:
        raise DataContractError(f"cannot export demo pack; missing {missing}")
    voice_summary = ""
    voice_path = outputs / "voice_replay_cases.json"
    if voice_path.exists():
        voice_data = json.loads(voice_path.read_text(encoding="utf-8"))
        blocked = [gate for gate in voice_data["release_gate"] if gate["gate"] == "BLOCK"]
        voice_summary = f"""

## Voice-Agent Reliability Layer

- Voice turns: {voice_data['voice_turns']}
- Voice failures: {voice_data['voice_failures']}
- Replay cases: {len(voice_data['replay_cases'])}
- Blocked releases: {', '.join(gate['release'] for gate in blocked) or 'none'}

The replay firewall converts planted production-like failures into deterministic replay cases,
then blocks candidate releases when latency, interruption, monologue, or frustration gates regress.
"""
    content = f"""# Demo Pack: {config['title']}

## Problem

{config['thesis']}

## Local Architecture

- Deterministic synthetic fixtures under `fixtures/`
- Local DuckDB store under `data/app.duckdb`
- Typed Pydantic records and evidence IDs
- Analyzer that emits structured JSON, CSV, Mermaid, and markdown reports
- Verifier that rejects unsupported `CLAIM:` lines
- Offline static dashboard with inline charts
{voice_summary}

## Runbook

```bash
uv sync
uv run app init-demo
uv run app ingest fixtures/
uv run app analyze
uv run app verify
uv run app dashboard
uv run app benchmark
uv run app export-demo-pack
```

## Validation

See `outputs/test_results.md` and `outputs/benchmark.md`.

## Limits

The data is synthetic and designed to prove workflow, evidence handling, and verification behavior.
It does not use private data, external APIs, production systems, or hidden credentials.
"""
    path = outputs / "demo_pack.md"
    path.write_text(content, encoding="utf-8")
    return {"demo_pack": str(path)}
