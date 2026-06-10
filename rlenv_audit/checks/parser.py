"""parser check — does the answer parser survive cosmetic reformatting?

A model rarely emits the answer in exactly the canonical shape. If the parser
drops the reward over a stray space, a trailing period, or the answer being
stated twice, correct rollouts get scored wrong and the gradient is corrupted.

We don't know an env's answer format a priori, so we first *discover* a canonical
wrapper that the parser extracts the gold answer from (boxed, hashed, plain, …),
then apply a battery of harmless perturbations to it and check the parser still
recovers the answer. Score = fraction of perturbations handled.
"""

from __future__ import annotations

from rlenv_audit.adapters.verifiers import EnvHandle
from rlenv_audit.checks.base import CheckResult, CheckStatus

# Below this fraction of perturbations handled, the parser is flagged brittle.
_WARN_THRESHOLD = 0.8


def _as_completion(text: str):
    return [{"role": "assistant", "content": text}]


def _norm(s) -> str:
    return str(s).strip() if s is not None else ""


def _candidate_wrappers(ans: str) -> list[tuple[str, str]]:
    """Plausible canonical formats an env might expect the answer in."""
    return [
        ("plain", ans),
        ("boxed", f"\\boxed{{{ans}}}"),
        ("hashed", f"#### {ans}"),
        ("stated", f"The answer is {ans}."),
        ("answer_tag", f"<answer>{ans}</answer>"),
        ("think_then", f"<think>reasoning</think>\n\\boxed{{{ans}}}"),
    ]


def _perturbations(canonical: str, ans: str) -> list[tuple[str, str]]:
    """(label, perturbed text) — each should still parse to the gold answer."""
    perts = [
        ("canonical", canonical),
        ("surrounding_whitespace", f"   \n  {canonical}  \n   "),
        ("trailing_punctuation", f"{canonical}."),
        ("trailing_newlines", f"{canonical}\n\n"),
        ("leading_reasoning", f"Let me work through this step by step. {canonical}"),
        ("stated_twice", f"{canonical} So, again: {canonical}"),
    ]
    # Casing only tests the parser's structural tokens (e.g. \BOXED vs \boxed).
    # If the answer itself has letters, recasing would change the answer's
    # content, not just its format — that's not a parser bug, so skip it.
    if not any(c.isalpha() for c in ans):
        perts.append(("uppercased", canonical.upper()))
        perts.append(("lowercased", canonical.lower()))
    return perts


def check_parser(handle: EnvHandle, config: dict) -> CheckResult:
    if handle.parser is None:
        return CheckResult("parser", CheckStatus.SKIP, "environment has no parser")

    n_prompts = int(config.get("parser_prompts", 3))
    threshold = float(config.get("parser_threshold", _WARN_THRESHOLD))
    rows = handle.dataset(n=n_prompts)
    if not rows:
        return CheckResult("parser", CheckStatus.SKIP, "environment exposes no dataset")

    parser = handle.parser

    # A pass-through/identity parser returns the whole message verbatim — answer
    # extraction then lives in the reward function, not the parser. Perturbing
    # the text would "fail" trivially and mislead, so don't test it as a parser.
    sentinel = "qx7 sentinel answer zz9 trailing words"
    try:
        passthrough = _norm(parser.parse_answer(_as_completion(sentinel))) == sentinel
    except Exception:
        passthrough = False
    if passthrough:
        return CheckResult(
            "parser", CheckStatus.SKIP,
            "env uses a pass-through parser (answer extraction is in the reward function, "
            "not the parser) — nothing parser-specific to perturb",
        )

    per_perturbation_pass: dict[str, int] = {}
    per_perturbation_total: dict[str, int] = {}
    findings: list[dict] = []
    rows_tested = 0

    for i, row in enumerate(rows):
        ans = _norm(row["answer"])
        if not ans:
            continue

        # Discover a canonical wrapper this parser actually extracts the answer
        # from. First ask the parser to format the answer itself (XMLParser etc.),
        # then fall back to generic format guesses.
        canonical = None
        canon_label = None
        from_parser = handle.canonical_answer(ans)
        candidates = ([("parser_format", from_parser)] if from_parser else []) + _candidate_wrappers(ans)
        for label, wrapper in candidates:
            try:
                got = parser.parse_answer(_as_completion(wrapper))
            except Exception:
                got = None
            if _norm(got) == ans:
                canonical, canon_label = wrapper, label
                break

        if canonical is None:
            findings.append({"prompt": i, "note": "no canonical format parsed cleanly"})
            continue

        rows_tested += 1
        for label, text in _perturbations(canonical, ans):
            try:
                got = parser.parse_answer(_as_completion(text))
                ok = _norm(got) == ans
            except Exception:
                got, ok = None, False
            per_perturbation_total[label] = per_perturbation_total.get(label, 0) + 1
            per_perturbation_pass[label] = per_perturbation_pass.get(label, 0) + int(ok)
            findings.append(
                {
                    "prompt": i,
                    "canonical_format": canon_label,
                    "perturbation": label,
                    "extracted": _norm(got),
                    "expected": ans,
                    "handled": ok,
                }
            )

    if rows_tested == 0:
        return CheckResult(
            "parser", CheckStatus.SKIP,
            "could not construct any answer the parser extracts (unknown format)",
            details={"findings": findings},
        )

    total = sum(per_perturbation_total.values())
    passed = sum(per_perturbation_pass.values())
    score = passed / total if total else 0.0
    weak = sorted(
        label for label, tot in per_perturbation_total.items()
        if per_perturbation_pass.get(label, 0) < tot
    )
    details = {
        "rows_tested": rows_tested,
        "perturbation_pass_rate": {
            label: f"{per_perturbation_pass.get(label,0)}/{tot}"
            for label, tot in per_perturbation_total.items()
        },
        "weak_perturbations": weak,
        "findings": findings,
    }

    if score < threshold:
        return CheckResult(
            "parser", CheckStatus.WARN,
            f"parser handled {passed}/{total} perturbations ({score:.0%}); "
            f"brittle on: {', '.join(weak) if weak else '—'}",
            score=score, details=details,
        )
    return CheckResult(
        "parser", CheckStatus.PASS,
        f"parser handled {passed}/{total} perturbations ({score:.0%})"
        + (f"; minor gaps: {', '.join(weak)}" if weak else ""),
        score=score, details=details,
    )


from rlenv_audit.checks import CheckSpec  # noqa: E402

SPEC = CheckSpec(
    name="parser",
    func=check_parser,
    description="answer parser still extracts under whitespace/format/casing perturbations",
    needs_gpu=False,
    needs_docker=False,
)
