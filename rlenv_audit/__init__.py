"""env_audit — a skill-based auditing system for RL environments.

The audit *checks* are skill files (``skills/``) executed by an agent (Claude
Code / Codex). This package is the deterministic tool layer those skills call:

    from rlenv_audit import load_handle
    from rlenv_audit.tools import inspect_env, score_completions, run_rollouts

The same tools are exposed on the ``env_audit`` / ``rlenv-audit`` CLI.
"""

from importlib.metadata import PackageNotFoundError, version

from rlenv_audit.adapters.verifiers import EnvHandle, EnvLoadError, load_handle

try:
    __version__ = version("rlenv-audit")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0+unknown"

__all__ = ["load_handle", "EnvHandle", "EnvLoadError", "__version__"]
