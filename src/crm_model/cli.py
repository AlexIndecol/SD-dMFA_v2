from __future__ import annotations
from pathlib import Path
import typer

from .run import run_project

app = typer.Typer(add_completion=False)

@app.command()
def run(
    scenario: str = typer.Option("baseline", help="Scenario id (used for output folder naming)."),
    dry_run: bool = typer.Option(True, help="If true, only ingest/validate/snapshot and compute placeholder indicators.")
):
    root = Path(__file__).resolve().parents[2]
    out_dir = run_project(root=root, scenario_id=scenario, dry_run=dry_run)
    typer.echo(f"Wrote outputs to: {out_dir}")

if __name__ == "__main__":
    app()
