"""Docker-isolated execution for the exploits check.

The exploits check submits deliberately hostile completions to a reward function.
For code/agentic environments the rubric *executes* the submitted code, so this
must never run on the host. ``run_scoring`` ships the scoring into a locked-down
container: no network, capped CPU/memory, read-only mounts.

Design notes (learned the hard way on the target box):

* The Docker daemon does **not** see the host's ``/tmp`` (systemd PrivateTmp), so
  the scratch dir lives under ``~/.cache/rlenv_audit`` — a path the daemon mounts
  fine.
* We don't install anything in the container. We mount the *existing* project
  venv + its uv-managed CPython + the HF cache at their real paths and run the
  venv's interpreter directly. The container image only provides a glibc
  userspace.
* The dataset is read on the host and the chosen task is passed in via the
  payload, so the sandbox only needs to *score*, not hit the network.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_IMAGE = "python:3.11-slim"
_RESULT_MARKER = "SBX_RESULT="


class SandboxError(Exception):
    """Raised when the sandbox cannot run (Docker down, image missing, timeout)."""


def _error_tail(stderr: str) -> str:
    """Pull a useful one-liner out of a container's stderr for a SKIP message —
    the last real error line, ignoring noisy warnings."""
    lines = [
        ln.strip()
        for ln in stderr.splitlines()
        if ln.strip() and "Warning" not in ln and not ln.startswith("Map")
    ]
    if not lines:
        return "(no stderr)"
    # Prefer the last line mentioning an error/exception, else just the tail.
    for ln in reversed(lines):
        if any(tok in ln for tok in ("Error", "error", "Exception", "Could not", "not found")):
            return ln[:240]
    return lines[-1][:240]


def docker_available() -> tuple[bool, str]:
    """Return ``(ok, message)``. ``ok`` is False (with a reason) if Docker can't
    be used — the exploits check turns this into a clean SKIP."""
    try:
        import docker
    except Exception as exc:  # pragma: no cover
        return False, f"docker SDK not importable: {exc}"
    try:
        client = docker.from_env()
        client.ping()
        return True, "docker available"
    except Exception as exc:
        return False, f"docker daemon not reachable: {exc}"


def _mounts() -> dict[str, dict[str, str]]:
    """Build the bind-mount set (host_path -> docker volume spec), derived from
    the running interpreter so it works regardless of where the venv lives."""
    import rlenv_audit

    volumes: dict[str, dict[str, str]] = {}

    def add(path: str, mode: str = "ro") -> None:
        rp = os.path.realpath(path)
        if rp and os.path.exists(rp):
            volumes[rp] = {"bind": rp, "mode": mode}

    # Project source (holds the editable rlenv_audit package + usually the venv).
    project_root = os.path.realpath(str(Path(rlenv_audit.__file__).resolve().parent.parent))
    add(project_root)
    # The venv — only as a separate mount if it lives OUTSIDE the project tree.
    # When it's inside (the common case), the project mount already covers it; a
    # nested bind here breaks the venv's pyvenv.cfg detection in the container.
    venv = os.path.realpath(sys.prefix)
    if not venv.startswith(project_root + os.sep):
        add(venv)
    # The uv-managed CPython the venv's python symlinks into (so the symlink
    # resolves inside the container).
    python_home = str(Path(os.path.realpath(sys.executable)).parent.parent)
    add(python_home)
    # HuggingFace cache so cached datasets load offline. Mounted read-WRITE:
    # many envs (re)build/cache their dataset at load time and fail on a
    # read-only cache. The container is still network-isolated, resource-capped
    # and ephemeral, so the only effect is normal dataset caching.
    hf_cache = Path.home() / ".cache" / "huggingface"
    if hf_cache.exists():
        add(str(hf_cache), mode="rw")

    return volumes


def _scratch_base() -> Path:
    base = Path.home() / ".cache" / "rlenv_audit"
    base.mkdir(parents=True, exist_ok=True)
    return base


def run_scoring(
    env_id: str,
    task: dict[str, Any],
    cheats: list[dict[str, str]],
    *,
    env_args: dict[str, Any] | None = None,
    image: str = DEFAULT_IMAGE,
    timeout_s: int = 300,
    mem_limit: str = "2g",
    cpus: float = 2.0,
) -> dict[str, dict]:
    """Score ``cheats`` against ``env_id`` inside Docker; return ``{label: {...}}``.

    Each result dict carries ``reward`` (float) or ``error`` (str). Raises
    ``SandboxError`` on any infrastructure failure so the caller can SKIP.
    """
    import docker

    volumes = _mounts()
    scratch = Path(tempfile.mkdtemp(prefix="sbx-", dir=str(_scratch_base())))
    try:
        payload = {
            "env_id": env_id,
            "env_args": env_args or {},
            "task": task,
            "cheats": cheats,
        }
        (scratch / "payload.json").write_text(json.dumps(payload))
        volumes[str(scratch)] = {"bind": "/sbx", "mode": "ro"}

        runner = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "_sandbox_runner.py"
        )
        # Invoke the venv's python *symlink* (not its realpath): running the
        # symlink is what triggers pyvenv.cfg detection so the venv's
        # site-packages (verifiers, the env, rlenv_audit) are importable.
        venv_python = sys.executable

        client = docker.from_env()
        try:
            container = client.containers.run(
                image,
                command=[venv_python, runner, "/sbx/payload.json"],
                volumes=volumes,
                environment={
                    "HOME": os.path.expanduser("~"),
                    "HF_HUB_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                network_mode="none",
                mem_limit=mem_limit,
                nano_cpus=int(cpus * 1e9),
                working_dir="/sbx",
                detach=True,
            )
        except docker.errors.ImageNotFound as exc:
            raise SandboxError(
                f"docker image '{image}' not present and cannot pull offline: {exc}"
            ) from exc
        except Exception as exc:
            raise SandboxError(f"failed to start sandbox container: {exc}") from exc

        try:
            result = container.wait(timeout=timeout_s)
            logs = container.logs(stdout=True, stderr=False).decode("utf-8", "replace")
            errs = container.logs(stdout=False, stderr=True).decode("utf-8", "replace")
            status = result.get("StatusCode", 1)
        except Exception as exc:
            try:
                container.kill()
            except Exception:
                pass
            raise SandboxError(f"sandbox execution failed/timed out: {exc}") from exc
        finally:
            try:
                container.remove(force=True)
            except Exception:
                pass

        for line in logs.splitlines():
            if line.startswith(_RESULT_MARKER):
                return json.loads(line[len(_RESULT_MARKER):])
        raise SandboxError(
            f"env failed to load/score in the sandbox (exit {status}): {_error_tail(errs)}"
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
