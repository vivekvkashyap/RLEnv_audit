---
name: env-audit-repair
description: Repair an audited RL environment from its scorecard feedback — only when the user explicitly asks to rewrite/repair/fix the env. Applies mechanical fixes to a local copy under rlenv_audit_repairs/ (never the installed package, never the Hub), triages design-level findings into recommendations, and validates the repaired copy by re-running the cheap checks. Publishing the result is the user's job.
---

# env-repair — rewrite the environment from audit feedback

**Runs only on an explicit user request** — "rewrite it", "repair the env",
"apply the feedback". Never as an automatic part of an audit: the orchestrator
may *offer* it once after a WARN/FAIL grade, but it starts only on the user's
yes. This skill is not a seventh check — it produces no score row; it consumes
the report the six checks wrote.

**Contract: everything lands in a local folder.**
All changes go to `rlenv_audit_repairs/<account>__<name>/` in the working
directory — a *copy* of the env source. Never edit the installed package in
site-packages, never reinstall over the original, never publish anything.
Pushing the repaired env to the Hub (or upstream as a patch) is the user's
decision and the user's command.

## Guardrails — read before editing

- **Fix causes, not scores.** The goal is a better *training* environment, not
  a better scorecard. Never loosen a reward, widen a parser, or pad a system
  prompt *because it would score higher* — only because the feedback names a
  real defect it cures. If a fix's only justification is "the check would pass",
  don't make it.
- **Reward changes are loud.** Any edit to a reward function changes the
  training objective itself. Flag every one prominently in the summary, state
  what behavior it now rewards differently, and keep it minimal. When a reward
  fix could plausibly alter task semantics, propose it instead of applying it.
- **Triage every finding** before touching anything:
  - **Mechanical — apply**: parser too strict / missing fallback, reward
    crashing on edge inputs (empty, long, malformed), missing or ambiguous
    system prompt / format instruction, missing partial credit where the
    feedback calls for it, unreachable termination / missing `max_turns`,
    undeclared runtime dependencies, broken imports / convention slips.
  - **Design-level — recommend, never apply**: misaligned dataset or task
    (tests the wrong thing), contaminated rows (data dedup is the author's
    call), difficulty recalibration, changing the interaction shape
    (single-turn ↔ tool use), anything that rewrites *what is being trained*.
- **Minimal diffs.** Each fix is the smallest change that cures the named
  defect, with a one-line rationale. No refactors, no style passes, no
  improvements the report didn't ask for.

## Steps

1. **Inputs.** Read `rlenv_audit_reports/<account>__<name>/report.json` (and
   `report.md`). If no report exists, stop: "Run the audit first — repair works
   from its feedback." Also read `/tmp/envaudit_inspect.json` for `module_file`
   and the reward sources.

2. **Get the source.** If the user has the env source locally (ask if unclear),
   copy *that* into `rlenv_audit_repairs/<account>__<name>/`. Otherwise copy the
   installed package's source (the directory of `module_file`, plus its
   `pyproject.toml` if present) into the repair folder, preserving the package
   layout so the module imports under the same name. Work only inside this
   folder from here on.

3. **Triage.** Walk the report's feedback and per-check justifications; sort
   every finding into *mechanical* or *design-level* per the guardrails. List
   the triage to the user before editing.

4. **Apply the mechanical fixes.** One finding at a time, smallest possible
   diff, each with a rationale tied to the report line it cures.

5. **Write `REPAIRS.md`** in the repair folder: the date, the report it worked
   from, a table of applied fixes (finding → file → what changed → why), the
   design-level recommendations left for the author, and any reward-function
   changes called out in their own section.

6. **Validate against the repaired copy — without touching the install.** Put
   the repair folder first on the import path and re-run the cheap checks:
   `PYTHONPATH=rlenv_audit_repairs/<account>__<name> rlenv-audit inspect <env> ...`
   then re-score the reward-design probes the same way. Confirm each applied
   fix actually cures its finding (the crash no longer crashes, the formerly
   zero-scored correct answer now scores). If a fix doesn't validate, revert it
   and record that in `REPAIRS.md`.

7. **Hand back.** Tell the user: the repair folder path, the fixes applied
   (rewards flagged), the recommendations not applied and why, the validation
   results, and the two follow-ups that are theirs: run a **fresh full audit**
   against the repaired copy to get the before/after scorecard, and publish to
   the Hub themselves if satisfied.

## Output

No score row. End with a short summary block:

```
repaired:  rlenv_audit_repairs/<account>__<name>/   (REPAIRS.md inside)
applied:   <n> mechanical fixes (<m> touched a reward function — review these)
deferred:  <k> design-level recommendations
validated: <which findings were confirmed cured against the repaired copy>
next:      re-audit the repaired copy; publish is your call
```
