"""Batch-audit a list of Hub environments and aggregate the results.

Each env is installed and audited in an isolated subprocess (with timeouts) so a
single hang, crash, or dependency blow-up can't take down the whole run. Produces
a survey.json + a printed summary table. This is the harness behind the survey
("I audited N environments; X failed determinism, Y are exploitable, …") and the
fastest way to surface env-variability bugs in the checks.

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
AUDIT_TIMEOUT = 360


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
        # Skip the hardware/endpoint-dependent checks so the batch stays CPU-only.
        code, so, se = run(
            [audit, "run", name, "--skip", "distribution,rollouts",
             "--json", out, "--no-report"],
            AUDIT_TIMEOUT,
        )
        elapsed = round(time.time() - t0, 1)

        if os.path.exists(out):
            rep = json.load(open(out))
            checks = {c["check_name"]: c["status"] for c in rep["checks"]}
            row = {"env": name, "status": "audited", "grade": rep["grade"],
                   "checks": checks, "elapsed_s": elapsed}
            # surface anything interesting
            row["findings"] = {
                c["check_name"]: c["summary"]
                for c in rep["checks"] if c["status"] in ("FAIL", "WARN")
            }
        else:
            reason = (se or so).strip().splitlines()
            row = {"env": name, "status": "load_failed", "grade": None,
                   "reason": (reason[-1][:200] if reason else "unknown"), "elapsed_s": elapsed}
        results.append(row)
        print(f"  -> {row['status']} grade={row.get('grade')} ({elapsed}s)", flush=True)

    summary = {"n": len(results), "results": results}
    with open(os.path.join(ROOT, "survey.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n\n================ SURVEY SUMMARY ================")
    print(f"{'env':24s} {'status':12s} {'grade':6s} checks")
    for r in results:
        if r["status"] == "audited":
            cks = " ".join(f"{k[:4]}:{v}" for k, v in r["checks"].items())
            print(f"{r['env']:24s} {'audited':12s} {str(r['grade']):6s} {cks}")
        else:
            print(f"{r['env']:24s} {r['status']:12s} {'-':6s} {r.get('reason','')[:60]}")
    print("\nsurvey.json written.")


if __name__ == "__main__":
    main(sys.argv[1:])
