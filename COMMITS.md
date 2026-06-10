# COMMITS — build plan for RLEnv_audit

Built commit-by-commit; each commit builds and is self-contained. Tick a box when
the commit lands. Milestone is commit 4: the moment `rlenv-audit run gsm8k --only
determinism` prints `determinism PASS` on a real Hub env, the architecture is
proven.

- [x] **1. chore: scaffold** — `pyproject.toml` (verifiers==0.1.14, click, rich,
  docker), `.gitignore`, README stub, empty package dirs, `DESIGN.md`,
  `COMMITS.md`. Dedicated venv (`uv`, py3.10); `verifiers==0.1.14` + editable
  install; `vf-install gsm8k -r` to get the reference env.
- [x] **2. feat(base+report): data model** — `checks/base.py` (`CheckStatus`,
  `CheckResult`), `report.py` (`Scorecard.to_terminal` / `to_json`).
- [x] **3. feat(adapter): verifiers EnvHandle** — load env; normalize
  RubricGroup-aware rubric / parser / dataset; synchronous `score()` over the
  async path; `teardown()`. Graceful load failures.
- [x] **4. feat(determinism) + CLI** — determinism check + `audit()` orchestrator
  + `cli.py` (`run`, `list-checks`). **Milestone proven on gsm8k.**
- [x] **5. feat(sandbox+exploits)** — `sandbox.py` Docker isolation + exploits
  check running cheat patterns inside it; SKIP cleanly if Docker is down.
- [x] **6. feat(parser)** — parser-robustness check.
- [x] **7. feat(contamination)** — n-gram overlap vs cached eval sets.
- [x] **8. feat(latency)** — timing check.
- [x] **9. feat(distribution)** — GPU/vLLM check; SKIP-degrading when absent.
- [x] **10. docs+tests** — README, sample scorecard, what each check means; real
  tests in `tests/`.
- [ ] **11. push** — `git push -u origin main` after confirming the remote +
  stored credentials work.

## Guardrails (apply throughout)

- Library-first: every check callable programmatically; CLI is a thin shell.
- Keep `verifiers` pinned to `==0.1.14`; never pull `verifiers-rl`/torch/vLLM as
  hard deps.
- Fail gracefully: env won't load / no vLLM / Docker down → clear SKIP or error,
  never a traceback dump.
- No plugin system, no config framework, no multi-format support. Six checks,
  one format, that's v0.
