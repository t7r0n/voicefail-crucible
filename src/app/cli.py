from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from app import core


console = Console()
cli = typer.Typer(no_args_is_help=True)


def print_result(result: dict) -> None:
    console.print_json(json.dumps(result, indent=2))


@cli.command("init-demo")
def init_demo(seed: int = 42, records: int = 96) -> None:
    print_result(core.init_demo(Path.cwd(), seed=seed, records=records))


@cli.command()
def ingest(path: Path = typer.Argument(Path("fixtures"))) -> None:
    print_result(core.ingest(Path.cwd(), path))


@cli.command()
def analyze() -> None:
    print_result(core.analyze(Path.cwd()))


@cli.command()
def verify() -> None:
    try:
        print_result(core.verify(Path.cwd()))
    except core.VerificationError as exc:
        console.print(f"[red]verification failed:[/red] {exc}")
        raise typer.Exit(1) from exc


@cli.command()
def dashboard() -> None:
    print_result(core.dashboard(Path.cwd()))


@cli.command()
def benchmark() -> None:
    print_result(core.benchmark(Path.cwd()))


@cli.command("export-demo-pack")
def export_demo_pack() -> None:
    print_result(core.export_demo_pack(Path.cwd()))


def main() -> None:
    cli()
