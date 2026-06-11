"""In-sandbox scoring runner — executes INSIDE the Docker container, never on the
host. It is invoked as ``python _sandbox_runner.py <payload.json>``.

The payload (written by ``sandbox.py`` on the host) names the environment and a
list of untrusted completions. This script loads the env, scores each completion
through its rubric, and prints a single ``SBX_RESULT=<json>`` line to stdout. It
runs with no network; if a completion is hostile code that the rubric executes,
it executes here, contained — that is the whole point of the sandbox.
"""

import json
import sys
import warnings

warnings.filterwarnings("ignore")


def main() -> None:
    payload = json.load(open(sys.argv[1]))
    from rlenv_audit.adapters.verifiers import load_handle

    handle = load_handle(payload["env_id"], payload.get("env_args") or {})
    task = payload["task"]
    columns = task.get("columns") or task.get("raw") or {}
    results: dict[str, dict] = {}
    for cheat in payload["cheats"]:
        try:
            reward, metrics = handle.score(
                cheat["text"],
                task["prompt"],
                task.get("answer", ""),
                task.get("info") or {},
                columns,
            )
            results[cheat["label"]] = {"reward": reward, "metrics": metrics}
        except Exception as exc:  # a cheat that crashes scoring is not an exploit
            results[cheat["label"]] = {"error": str(exc)}

    try:
        handle.teardown()
    except Exception:
        pass

    print("SBX_RESULT=" + json.dumps(results))


if __name__ == "__main__":
    main()
