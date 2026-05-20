from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from app import core


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def reset_runtime() -> None:
    for name in ["fixtures", "data", "outputs"]:
        path = PROJECT_ROOT / name
        if path.exists():
            shutil.rmtree(path)


def test_end_to_end_flow() -> None:
    reset_runtime()
    created = core.init_demo(PROJECT_ROOT, seed=123, records=64)
    assert created["records"] == 64
    ingest = core.ingest(PROJECT_ROOT)
    assert ingest["records"] == 64
    analysis = core.analyze(PROJECT_ROOT)
    assert analysis["clusters"]
    verification = core.verify(PROJECT_ROOT)
    assert verification["checks"]["evidence_claims_supported"] is True
    dash = core.dashboard(PROJECT_ROOT)
    assert Path(dash["dashboard"]).exists()
    bench = core.benchmark(PROJECT_ROOT)
    assert bench["target_records"] >= 500
    pack = core.export_demo_pack(PROJECT_ROOT)
    assert Path(pack["demo_pack"]).exists()


def test_malformed_fixture_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "fixtures"
    bad.mkdir()
    (bad / "records.jsonl").write_text(json.dumps({"record_id": "missing_fields"}) + "\n")
    with pytest.raises(core.DataContractError):
        core.ingest(PROJECT_ROOT, bad)


def test_unsupported_claim_rejected() -> None:
    reset_runtime()
    core.init_demo(PROJECT_ROOT, seed=321, records=32)
    core.ingest(PROJECT_ROOT)
    core.analyze(PROJECT_ROOT)
    (PROJECT_ROOT / "outputs" / "unsupported_claim.md").write_text(
        "CLAIM: this should be rejected because it has no evidence marker\n",
        encoding="utf-8",
    )
    with pytest.raises(core.VerificationError):
        core.verify(PROJECT_ROOT)


def test_unknown_evidence_rejected() -> None:
    reset_runtime()
    core.init_demo(PROJECT_ROOT, seed=222, records=32)
    core.ingest(PROJECT_ROOT)
    core.analyze(PROJECT_ROOT)
    (PROJECT_ROOT / "outputs" / "unknown_evidence.md").write_text(
        "CLAIM: this cites nonexistent evidence. [EVID: ev_missing]\n",
        encoding="utf-8",
    )
    with pytest.raises(core.VerificationError):
        core.verify(PROJECT_ROOT)


def test_cli_ingest_accepts_documented_positional_argument() -> None:
    reset_runtime()
    first = subprocess.run(
        ["uv", "run", "app", "init-demo"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "records" in first.stdout
    second = subprocess.run(
        ["uv", "run", "app", "ingest", "fixtures/"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "records" in second.stdout


def test_voice_specific_replay_firewall_outputs() -> None:
    reset_runtime()
    created = core.init_demo(PROJECT_ROOT, seed=77, records=96)
    assert created["voice_turns"] >= 96
    ingest = core.ingest(PROJECT_ROOT)
    assert ingest["voice_turns"] >= 96
    analysis = core.analyze(PROJECT_ROOT)
    domain = analysis["domain"]
    assert domain["voice_failures"] > 0
    assert {case["failure_mode"] for case in domain["replay_cases"]} >= {
        "barge_in_ignored",
        "latency_spike",
        "compliance_disclosure_omitted",
        "caller_frustration_ignored",
        "asr_ambiguity",
        "repetitive_monologue",
    }
    assert any(gate["release"] == "candidate-v2" and gate["gate"] == "BLOCK" for gate in domain["release_gate"])
    assert (PROJECT_ROOT / "outputs" / "voice_replay_cases.csv").exists()


def test_voice_evidence_ids_are_accepted_by_verifier() -> None:
    reset_runtime()
    core.init_demo(PROJECT_ROOT, seed=88, records=96)
    core.ingest(PROJECT_ROOT)
    core.analyze(PROJECT_ROOT)
    voice_data = json.loads((PROJECT_ROOT / "outputs" / "voice_replay_cases.json").read_text())
    evidence_id = voice_data["replay_cases"][0]["evidence_id"]
    (PROJECT_ROOT / "outputs" / "voice_claim.md").write_text(
        f"CLAIM: voice replay case is evidence backed. [EVID: {evidence_id}]\n",
        encoding="utf-8",
    )
    verification = core.verify(PROJECT_ROOT)
    assert verification["checks"]["voice_replay_cases_present"] is True


def test_dashboard_escapes_fixture_html() -> None:
    reset_runtime()
    core.init_demo(PROJECT_ROOT, seed=99, records=24)
    records_path = PROJECT_ROOT / "fixtures" / "records.jsonl"
    first_line, *rest = records_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(first_line)
    payload["scenario"] = "<script>alert('xss')</script>"
    records_path.write_text("\n".join([json.dumps(payload), *rest]) + "\n", encoding="utf-8")
    core.ingest(PROJECT_ROOT)
    core.analyze(PROJECT_ROOT)
    dashboard = Path(core.dashboard(PROJECT_ROOT)["dashboard"]).read_text(encoding="utf-8")
    assert "<script>alert('xss')</script>" not in dashboard
    assert "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;" in dashboard


def test_visual_svgs_are_bounded_and_regenerated() -> None:
    reset_runtime()
    core.init_demo(PROJECT_ROOT, seed=88, records=40)
    core.ingest(PROJECT_ROOT)
    core.analyze(PROJECT_ROOT)
    core.verify(PROJECT_ROOT)
    core.dashboard(PROJECT_ROOT)

    import xml.etree.ElementTree as ET

    for name in ["project_working.svg", "evidence_map.svg"]:
        svg_path = PROJECT_ROOT / "outputs" / name
        assert svg_path.exists()
        root = ET.fromstring(svg_path.read_text(encoding="utf-8"))
        _, _, width, height = [float(item) for item in root.attrib["viewBox"].split()]
        for rect in root.findall(".//{http://www.w3.org/2000/svg}rect"):
            x = float(rect.attrib.get("x", 0))
            y = float(rect.attrib.get("y", 0))
            w = float(rect.attrib.get("width", 0))
            h = float(rect.attrib.get("height", 0))
            assert x >= 0 and y >= 0
            assert x + w <= width
            assert y + h <= height


def test_dashboard_has_single_domain_panel_after_visual_pass() -> None:
    reset_runtime()
    core.init_demo(PROJECT_ROOT, seed=89, records=40)
    core.ingest(PROJECT_ROOT)
    analysis = core.analyze(PROJECT_ROOT)
    core.verify(PROJECT_ROOT)
    dashboard_path = Path(core.dashboard(PROJECT_ROOT)["dashboard"])
    dashboard = dashboard_path.read_text(encoding="utf-8")
    domain_title = analysis["domain"].get("dashboard_title", "Voice replay failure modes")
    assert dashboard.count(domain_title) == 1
    assert "project_working.svg" in dashboard
    assert "evidence_map.svg" in dashboard
