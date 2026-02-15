from __future__ import annotations
from pathlib import Path

from .run import run_project

try:
    import typer
except Exception:
    typer = None

def _run_impl(scenario: str = "baseline", dry_run: bool = True) -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = run_project(root=root, scenario_id=scenario, dry_run=dry_run)
    print(f"Wrote outputs to: {out_dir}")

if typer is not None:
    app = typer.Typer(add_completion=False)

    @app.command()
    def run(
        scenario: str = typer.Option("baseline", help="Scenario id (used for output folder naming)."),
        dry_run: bool = typer.Option(True, help="If true, only ingest/validate/snapshot and compute placeholder indicators.")
    ):
        _run_impl(scenario=scenario, dry_run=dry_run)

if __name__ == "__main__":
    if typer is not None:
        app()
    else:
        import argparse
        parser = argparse.ArgumentParser(description="CRM SD-dMFA runner")
        parser.add_argument("command", nargs="?", default="run")
        parser.add_argument("--scenario", default="baseline")
        parser.add_argument("--dry-run", default="true")
        args = parser.parse_args()
        if args.command != "run":
            parser.error("Only 'run' is supported.")
        dry = str(args.dry_run).lower() in {"1", "true", "yes", "y"}
        _run_impl(scenario=args.scenario, dry_run=dry)
