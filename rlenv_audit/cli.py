"""Thin CLI wrapper over ``rlenv_audit.audit()``.

Parses args, calls the library, renders the scorecard. No audit logic lives here.
"""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.table import Table

from rlenv_audit import __version__
from rlenv_audit.adapters.verifiers import EnvLoadError
from rlenv_audit.checks import REGISTRY
from rlenv_audit.core import audit


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


@click.group()
@click.version_option(__version__, prog_name="rlenv-audit")
def main() -> None:
    """RLEnv_audit — pytest for RL environments."""


@main.command()
@click.argument("env_id")
@click.option("--only", default=None, help="comma-separated checks to run (exclusively).")
@click.option("--skip", default=None, help="comma-separated checks to exclude.")
@click.option("--json", "json_path", default=None, help="also write the JSON report here.")
@click.option("--model", default=None, help="reference model for the distribution check.")
@click.option(
    "--report/--no-report",
    default=True,
    help="write report.json in the cwd (default: on).",
)
def run(
    env_id: str,
    only: str | None,
    skip: str | None,
    json_path: str | None,
    model: str | None,
    report: bool,
) -> None:
    """Run the audit battery against ENV_ID and print a scorecard."""
    console = Console()
    config: dict = {}
    if model:
        config["model"] = model

    try:
        scorecard = audit(
            env_id,
            only=_split_csv(only),
            skip=_split_csv(skip),
            config=config,
        )
    except EnvLoadError as exc:
        console.print(f"[bold red]error:[/] {exc}")
        sys.exit(2)
    except KeyError as exc:
        # unknown check name from --only/--skip
        console.print(f"[bold red]error:[/] {exc.args[0] if exc.args else exc}")
        sys.exit(2)

    console.print()
    scorecard.to_terminal(console)

    written: list[str] = []
    if report:
        scorecard.write_json("report.json")
        written.append("report.json")
    if json_path:
        scorecard.write_json(json_path)
        written.append(json_path)
    if written:
        console.print(f"\nreport written to: {', '.join(written)}")

    # Non-zero exit if the env failed the audit — useful in CI.
    sys.exit(1 if scorecard.grade == "FAIL" else 0)


@main.command(name="list-checks")
def list_checks() -> None:
    """List available checks and what each one needs."""
    console = Console()
    table = Table(title="RLEnv_audit checks", title_style="bold")
    table.add_column("check", style="bold")
    table.add_column("needs")
    table.add_column("description")
    for spec in REGISTRY.values():
        table.add_row(spec.name, spec.needs(), spec.description)
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    main()
