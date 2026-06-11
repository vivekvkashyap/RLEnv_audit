"""CLI for env_audit's mechanical tools.

The audit *checks* are skills run by an agent (see ``skills/``). These commands
are the deterministic helpers those skills shell out to: ``inspect`` an env,
``score`` agent-written completions, run shared ``rollouts``, and render a
``scorecard``. Each emits JSON (``--out`` / stdout) so a skill can read it back.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from rlenv_audit import __version__
from rlenv_audit.tools import (
    build_scorecard,
    inspect_env,
    render_scorecard,
    run_rollouts,
    score_completions,
)


def _emit(obj: dict, out: str | None) -> None:
    text = json.dumps(obj, indent=2, default=str)
    if out:
        with open(out, "w") as f:
            f.write(text)
        click.echo(f"written to {out}")
    else:
        click.echo(text)


@click.group()
@click.version_option(__version__, prog_name="env_audit")
def main() -> None:
    """env_audit — skill-based auditing for RL environments (mechanical tools)."""


@main.command()
@click.argument("env_id")
@click.option("-n", "--samples", default=20, help="dataset rows to sample.")
@click.option("--out", default=None, help="write JSON here instead of stdout.")
def inspect(env_id: str, samples: int, out: str | None) -> None:
    """Load ENV_ID and dump a structured description (load status, reward source,
    dataset samples, system prompt). Used by the integrity, problem-alignment,
    reward-design and contamination checks."""
    _emit(inspect_env(env_id, n=samples), out)


@main.command()
@click.argument("env_id")
@click.argument("completions_json", type=click.Path(exists=True))
@click.option("--out", default=None, help="write JSON here instead of stdout.")
def score(env_id: str, completions_json: str, out: str | None) -> None:
    """Score COMPLETIONS_JSON ([{prompt_index, label, text}, ...]) through ENV_ID's
    reward function. Used by the reward-design skill."""
    with open(completions_json) as f:
        completions = json.load(f)
    if isinstance(completions, dict):
        completions = completions.get("completions", [])
    _emit(score_completions(env_id, completions), out)


@main.command()
@click.argument("env_id")
@click.option("--endpoint", default=None, help="OpenAI-compatible base URL.")
@click.option("--api-key", default=None, help="API key (else OPENAI_API_KEY, else EMPTY for vLLM).")
@click.option("--model", default=None, help="model name (else first model served by the endpoint).")
@click.option("-n", "--samples", default=20, help="tasks (examples) to roll out.")
@click.option("-k", "--rollouts", "k", default=8, help="rollouts per task.")
@click.option("--max-tokens", default=1024, help="max tokens per generation (0 = model default).")
@click.option("--temperature", default=None, type=float, help="sampling temperature (else env default).")
@click.option("--max-concurrent", default=8, help="concurrent vf-eval requests.")
@click.option("--dummy", is_flag=True, help="fake rollouts without an endpoint.")
@click.option("--out", default=None, help="write JSON here (also cached by default).")
def rollouts(env_id, endpoint, api_key, model, samples, k, max_tokens, temperature, max_concurrent, dummy, out) -> None:
    """Generate K rollouts over N tasks once via verifiers' own vf-eval engine,
    score+time them, and cache to JSON. The latency and rollout-quality skills
    share this single cache. Requires the user's model endpoint to be up
    (vf-eval is a client; it does not start a model)."""
    result = run_rollouts(
        env_id, endpoint=endpoint, api_key=api_key, model=model,
        n_samples=samples, k=k, max_tokens=(max_tokens or None),
        temperature=temperature, max_concurrent=max_concurrent,
        dummy=dummy, cache_path=out,
    )
    click.echo(json.dumps({k2: v for k2, v in result.items() if k2 != "samples"}, indent=2, default=str))
    if "error" in result:
        sys.exit(1)


@main.command()
@click.argument("results_json", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="emit the computed scorecard as JSON.")
def scorecard(results_json: str, as_json: bool) -> None:
    """Render a scorecard from RESULTS_JSON ({env_id, checks:[{name,status,score,
    justification}]}). The orchestrator skill writes this after running the checks."""
    with open(results_json) as f:
        data = json.load(f)
    if as_json:
        click.echo(json.dumps(build_scorecard(data), indent=2, default=str))
    else:
        render_scorecard(data)
    if build_scorecard(data)["grade"] == "FAIL":
        sys.exit(1)


def _bundled_skills_dir() -> Path | None:
    """Locate the skill files shipped with this installation: inside the package
    for a wheel install (force-included at build time), beside it for an
    editable install / git checkout."""
    here = Path(__file__).resolve()
    for candidate in (here.parent / "skills", here.parents[1] / "skills"):
        if candidate.is_dir() and any(candidate.glob("*/SKILL.md")):
            return candidate
    return None


@main.command("install-skills")
@click.option(
    "--target", default=None,
    help="directory to install the skills into (default: ~/.claude/skills).",
)
def install_skills(target: str | None) -> None:
    """Copy the bundled audit skills (the six checks + the env-audit
    orchestrator) into the agent's skills directory, so asking Claude Code to
    "audit <env>" just works."""
    import shutil

    src = _bundled_skills_dir()
    if src is None:
        raise click.ClickException(
            "bundled skills not found in this installation — "
            "reinstall rlenv-audit or run from a git checkout"
        )
    dst_root = Path(target).expanduser() if target else Path.home() / ".claude" / "skills"
    dst_root.mkdir(parents=True, exist_ok=True)
    installed = []
    for skill_dir in sorted(p.parent for p in src.glob("*/SKILL.md")):
        shutil.copytree(skill_dir, dst_root / skill_dir.name, dirs_exist_ok=True)
        installed.append(skill_dir.name)
    click.echo(f"installed {len(installed)} skills to {dst_root}:")
    for name in installed:
        click.echo(f"  - {name}")
    click.echo('\nNow ask your agent: "Audit <env-id>" (e.g. primeintellect/gsm8k).')


if __name__ == "__main__":  # pragma: no cover
    main()
