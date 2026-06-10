"""The check registry.

Each check is an independent function `check_x(handle, config) -> CheckResult`.
The registry (populated in later commits) maps a check name to its function plus
metadata about what it needs (GPU, Docker) so the CLI can list and filter them.
"""
