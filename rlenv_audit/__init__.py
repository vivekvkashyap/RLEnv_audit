"""RLEnv_audit — pytest for RL environments.

Public API (library-first; the CLI is a thin wrapper over this):

    import rlenv_audit
    scorecard = rlenv_audit.audit(env)   # env: a loaded verifiers Environment

The full surface (`audit`, `CheckResult`, `CheckStatus`, `Scorecard`) is wired up
in later commits. This commit ships the project scaffold only.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
