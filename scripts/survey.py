"""Batch-inspect a list of Hub environments and aggregate the results.

The full audit is agent-driven (skills), so the deterministic part a script can
batch is the tool layer: install each env and run `rlenv-audit inspect` in an
isolated subprocess (with timeouts) so a single hang, crash, or dependency
blow-up can't take down the whole run. Produces survey.json + a printed summary
("N envs: X load cleanly, Y fail to load, ...") — the fastest way to surface
env-variability bugs in the adapter/tools before pointing an agent at an env.

Usage:
    python scripts/survey.py [env_id ...]          # defaults to CURATED below
    python scripts/survey.py --venv .venv311 aime2024 math500 ...

Env ids may be bare (`aime2024`) or Hub-qualified (`primeintellect/aime2024`).
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# A diverse, mostly-CPU starter set spanning env shapes: single-turn math,
# text/logic transforms, multi-turn games, reasoning, MCQ.
CURATED = [
    "aime2024", "aime2025", "math500", "hendrycks-math", "intellect-math",
    "skywork-math", "deepscaler-math", "acereason-math", "math-env", "minif2f",
    "reverse-text", "unscramble", "lisanbench", "ascii-tree", "logic-env",
    "pydantic-adherence", "verbatim-copy", "graphwalks",
    "wordle", "mastermind", "alphabet-sort", "sentence-repeater",
    "reasoning-gym-env", "mmlu-pro", "ifeval", "misguided-attn",
    "thematic-generalization", "simple-bench",
]

INSTALL_TIMEOUT = 240
INSPECT_TIMEOUT = 180


def run(cmd, timeout):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"


def main(argv):
    venv = ".venv311"
    envs = []
    i = 0
    while i < len(argv):
        if argv[i] == "--venv":
            venv = argv[i + 1]
            i += 2
        else:
            envs.append(argv[i])
            i += 1
    envs = envs or CURATED

    vf = os.path.join(ROOT, venv, "bin", "vf-install")
    audit = os.path.join(ROOT, venv, "bin", "rlenv-audit")
    outdir = os.path.join(ROOT, "survey_reports")
    os.makedirs(outdir, exist_ok=True)

    results = []
    for n, env in enumerate(envs, 1):
        name = env.split("/")[-1]
        hub_id = env if "/" in env else f"primeintellect/{name}"
        print(f"\n[{n}/{len(envs)}] === {name} ===", flush=True)

        t0 = time.time()
        run([vf, hub_id], INSTALL_TIMEOUT)  # vf-install returns 0 even on failure
        out = os.path.join(outdir, f"{name}.json")
        if os.path.exists(out):
            os.remove(out)
        code, so, se = run(
            [audit, "inspect", name, "-n", "5", "--out", out],
            INSPECT_TIMEOUT,
        )
        elapsed = round(time.time() - t0, 1)

        if os.path.exists(out):
            info = json.load(open(out))
            if info.get("loaded"):
                row = {
                    "env": name, "status": "loaded",
                    "env_type": info.get("env_type"),
                    "reward_funcs": [f["name"] for f in info.get("reward_funcs", [])],
                    "dataset_size": info.get("dataset_size"),
                    "has_system_prompt": bool(info.get("system_prompt")),
                    "n_samples": len(info.get("sample", [])),
                    "elapsed_s": elapsed,
                }
            else:
                row = {"env": name, "status": "load_failed",
                       "reason": str(info.get("error", ""))[:200], "elapsed_s": elapsed}
        else:
            reason = (se or so).strip().splitlines()
            row = {"env": name, "status": "inspect_crashed",
                   "reason": (reason[-1][:200] if reason else "unknown"), "elapsed_s": elapsed}
        results.append(row)
        print(f"  -> {row['status']} ({elapsed}s)", flush=True)

    summary = {"n": len(results), "results": results}
    with open(os.path.join(ROOT, "survey.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n\n================ SURVEY SUMMARY ================")
    print(f"{'env':24s} {'status':16s} detail")
    for r in results:
        if r["status"] == "loaded":
            detail = (f"{r['env_type']} rewards={','.join(r['reward_funcs'])[:40]} "
                      f"ds={r['dataset_size']}")
        else:
            detail = r.get("reason", "")[:70]
        print(f"{r['env']:24s} {r['status']:16s} {detail}")
    n_ok = sum(1 for r in results if r["status"] == "loaded")
    print(f"\n{n_ok}/{len(results)} loaded cleanly. survey.json written.")


if __name__ == "__main__":
    main(sys.argv[1:])
