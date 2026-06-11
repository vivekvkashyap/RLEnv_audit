"""env_audit — a skill-based auditing system for RL environments.

The audit *checks* are skill files (``skills/``) executed by an agent (Claude
Code / Codex). This package is the deterministic tool layer those skills call:

    from rlenv_audit import load_handle
    from rlenv_audit.tools import inspect_env, score_completions, run_rollouts

The same tools are exposed on the ``env_audit`` / ``rlenv-audit`` CLI.
"""

from rlenv_audit.adapters.verifiers import EnvHandle, EnvLoadError, load_handle

__version__ = "0.3.3"

__all__ = ["load_handle", "EnvHandle", "EnvLoadError", "__version__"]
