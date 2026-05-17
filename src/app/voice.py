from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import duckdb
from pydantic import BaseModel, Field


class VoiceTurn(BaseModel):
    turn_id: str
    call_id: str
    release: str
    speaker: str
    intent: str
    transcript: str
    started_ms: int
    duration_ms: int
    latency_ms: int
    silence_ms: int
    overlap_ms: int
    noise_db: float
    sentiment_delta: float = Field(ge=-1.0, le=1.0)
    outcome: str
    failure_mode: str
    evidence_id: str


class ReplayCase(BaseModel):
    replay_id: str
    failure_mode: str
    release: str
    source_turn_ids: list[str]
    expected_guardrail: str
    risk_score: float
    evidence_id: str


@dataclass(frozen=True)
class FailureRule:
    mode: str
    intent: str
    transcript: str
    guardrail: str


FAILURE_RULES = [
    FailureRule(
        "barge_in_ignored",
        "cancel appointment",
        "I am trying to interrupt because the date is wrong",
        "agent must stop monologue within one turn after overlap_ms >= 350",
    ),
    FailureRule(
        "latency_spike",
        "billing dispute",
        "Why did it take so long to answer that question?",
        "p95 latency must stay below 1400 ms for billing disputes",
    ),
    FailureRule(
        "compliance_disclosure_omitted",
        "insurance eligibility",
        "Before we continue I need the required disclosure",
        "regulated flows must include disclosure before collecting details",
    ),
    FailureRule(
        "caller_frustration_ignored",
        "technical support",
        "I have already said this three times and I am frustrated",
        "negative sentiment shift below -0.35 must trigger empathy and escalation",
    ),
    FailureRule(
        "asr_ambiguity",
        "pharmacy refill",
        "No, I said fifteen milligrams, not fifty",
        "ambiguous dosage or numeric entity must trigger confirmation",
    ),
    FailureRule(
        "repetitive_monologue",
        "loan servicing",
        "You are repeating yourself and not answering my question",
        "agent response longer than 35 seconds must be summarized or interrupted",
    ),
]


def should_enable(config: dict[str, Any]) -> bool:
    return config.get("slug") == "hamming-ai-voice-reliability-lab"


def generate_voice_traces(fixtures: Path, seed: int, calls: int = 84) -> dict[str, Any]:
    rng = random.Random(seed)
    turns: list[VoiceTurn] = []
    for index in range(calls):
        rule = FAILURE_RULES[index % len(FAILURE_RULES)]
        release = "candidate-v2" if index % 4 in {0, 1} else "baseline-v1"
        is_failure = index % 5 == 0 or (release == "candidate-v2" and index % 9 == 0)
        failure_mode = rule.mode if is_failure else "none"
        latency_ms = rng.randint(420, 1180) + (900 if rule.mode == "latency_spike" and is_failure else 0)
        overlap_ms = rng.randint(0, 180) + (460 if rule.mode == "barge_in_ignored" and is_failure else 0)
        silence_ms = rng.randint(80, 850) + (1400 if rule.mode == "asr_ambiguity" and is_failure else 0)
        duration_ms = rng.randint(2400, 9000) + (
            31000 if rule.mode == "repetitive_monologue" and is_failure else 0
        )
        sentiment_delta = round(rng.uniform(-0.18, 0.12), 3)
        if rule.mode == "caller_frustration_ignored" and is_failure:
            sentiment_delta = round(rng.uniform(-0.82, -0.42), 3)
        turn = VoiceTurn(
            turn_id=f"turn_{index:04d}",
            call_id=f"call_{index // 3:04d}",
            release=release,
            speaker="caller",
            intent=rule.intent,
            transcript=rule.transcript if is_failure else f"Resolved {rule.intent} cleanly",
            started_ms=index * 12_000,
            duration_ms=duration_ms,
            latency_ms=latency_ms,
            silence_ms=silence_ms,
            overlap_ms=overlap_ms,
            noise_db=round(rng.uniform(28.0, 62.0), 2),
            sentiment_delta=sentiment_delta,
            outcome="failed" if is_failure else "resolved",
            failure_mode=failure_mode,
            evidence_id=f"voice_ev_{index:04d}",
        )
        turns.append(turn)

    path = fixtures / "voice_traces.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for turn in turns:
            handle.write(turn.model_dump_json() + "\n")
    return {"voice_turns": len(turns), "voice_failures": sum(t.failure_mode != "none" for t in turns)}


def load_voice_turns(fixtures: Path) -> list[VoiceTurn]:
    path = fixtures / "voice_traces.jsonl"
    if not path.exists():
        return []
    return [VoiceTurn.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()]


def ingest_voice_turns(db_path: Path, turns: Iterable[VoiceTurn]) -> int:
    turns = list(turns)
    if not turns:
        return 0
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            """
            create table if not exists voice_turns (
                turn_id varchar primary key,
                call_id varchar,
                release varchar,
                speaker varchar,
                intent varchar,
                transcript varchar,
                started_ms integer,
                duration_ms integer,
                latency_ms integer,
                silence_ms integer,
                overlap_ms integer,
                noise_db double,
                sentiment_delta double,
                outcome varchar,
                failure_mode varchar,
                evidence_id varchar
            )
            """
        )
        for turn in turns:
            conn.execute(
                "insert into voice_turns values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    turn.turn_id,
                    turn.call_id,
                    turn.release,
                    turn.speaker,
                    turn.intent,
                    turn.transcript,
                    turn.started_ms,
                    turn.duration_ms,
                    turn.latency_ms,
                    turn.silence_ms,
                    turn.overlap_ms,
                    turn.noise_db,
                    turn.sentiment_delta,
                    turn.outcome,
                    turn.failure_mode,
                    turn.evidence_id,
                ],
            )
    return len(turns)


def turns_from_db(root: Path) -> list[VoiceTurn]:
    db_path = root / "data" / "app.duckdb"
    with duckdb.connect(str(db_path), read_only=True) as conn:
        exists = conn.execute(
            "select count(*) from information_schema.tables where table_name = 'voice_turns'"
        ).fetchone()[0]
        if not exists:
            return []
        rows = conn.execute("select * from voice_turns order by turn_id").fetchall()
    return [
        VoiceTurn(
            turn_id=row[0],
            call_id=row[1],
            release=row[2],
            speaker=row[3],
            intent=row[4],
            transcript=row[5],
            started_ms=row[6],
            duration_ms=row[7],
            latency_ms=row[8],
            silence_ms=row[9],
            overlap_ms=row[10],
            noise_db=row[11],
            sentiment_delta=row[12],
            outcome=row[13],
            failure_mode=row[14],
            evidence_id=row[15],
        )
        for row in rows
    ]


def replay_cases(turns: list[VoiceTurn]) -> list[ReplayCase]:
    grouped: dict[str, list[VoiceTurn]] = defaultdict(list)
    for turn in turns:
        if turn.failure_mode != "none":
            grouped[turn.failure_mode].append(turn)

    cases: list[ReplayCase] = []
    rule_by_mode = {rule.mode: rule for rule in FAILURE_RULES}
    for mode, failures in sorted(grouped.items()):
        top = max(
            failures,
            key=lambda turn: (
                turn.latency_ms / 1400,
                turn.overlap_ms / 350,
                abs(turn.sentiment_delta),
                turn.duration_ms / 35_000,
            ),
        )
        risk_score = min(
            100.0,
            round(
                (top.latency_ms / 40)
                + (top.overlap_ms / 9)
                + (top.silence_ms / 45)
                + abs(top.sentiment_delta) * 35
                + (top.duration_ms / 900),
                2,
            ),
        )
        cases.append(
            ReplayCase(
                replay_id=f"replay_{mode}",
                failure_mode=mode,
                release=top.release,
                source_turn_ids=[turn.turn_id for turn in failures[:5]],
                expected_guardrail=rule_by_mode[mode].guardrail,
                risk_score=risk_score,
                evidence_id=top.evidence_id,
            )
        )
    return cases


def analyze_voice(root: Path) -> dict[str, Any]:
    turns = turns_from_db(root)
    if not turns:
        return {}
    failures = [turn for turn in turns if turn.failure_mode != "none"]
    cases = replay_cases(turns)
    by_release: dict[str, list[VoiceTurn]] = defaultdict(list)
    for turn in turns:
        by_release[turn.release].append(turn)
    release_gate = []
    for release, items in sorted(by_release.items()):
        fail_rate = sum(1 for item in items if item.failure_mode != "none") / len(items)
        p95_latency = sorted(item.latency_ms for item in items)[int(len(items) * 0.95) - 1]
        barge_in_violations = sum(1 for item in items if item.overlap_ms >= 350)
        monologues = sum(1 for item in items if item.duration_ms >= 35_000)
        gate_pass = fail_rate <= 0.14 and p95_latency < 1500 and barge_in_violations <= 1
        release_gate.append(
            {
                "release": release,
                "turns": len(items),
                "fail_rate": round(fail_rate, 4),
                "p95_latency_ms": p95_latency,
                "barge_in_violations": barge_in_violations,
                "monologue_violations": monologues,
                "gate": "PASS" if gate_pass else "BLOCK",
            }
        )
    result = {
        "voice_turns": len(turns),
        "voice_failures": len(failures),
        "failure_modes": {
            mode: sum(1 for turn in failures if turn.failure_mode == mode)
            for mode in sorted({turn.failure_mode for turn in failures})
        },
        "replay_cases": [case.model_dump() for case in cases],
        "release_gate": release_gate,
        "latency_p95_ms": sorted(turn.latency_ms for turn in turns)[int(len(turns) * 0.95) - 1],
        "frustration_events": sum(1 for turn in turns if turn.sentiment_delta <= -0.35),
    }
    outputs = root / "outputs"
    (outputs / "voice_replay_cases.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (outputs / "voice_replay_cases.csv").open("w", encoding="utf-8") as handle:
        handle.write("replay_id,failure_mode,release,risk_score,evidence_id,expected_guardrail\n")
        for case in cases:
            handle.write(
                f"{case.replay_id},{case.failure_mode},{case.release},{case.risk_score},"
                f"{case.evidence_id},\"{case.expected_guardrail}\"\n"
            )
    return result
