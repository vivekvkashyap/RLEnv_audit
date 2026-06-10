"""contamination check — does the env's dataset overlap popular eval sets?

If training tasks overlap a benchmark (AIME, MATH-500, GSM8K, HumanEval,
LiveCodeBench), then "improvement" measured on that benchmark is partly
memorization, not learning. We n-gram each dataset question and look for high
containment against the reference benchmarks.

Reference sets are loaded via HuggingFace and cached locally by the `datasets`
library. Sets that can't be fetched (offline, script-based loaders) are reported
as unavailable, not fatal — the check only SKIPs if *no* reference set loads.
"""

from __future__ import annotations

import re

from rlenv_audit.adapters.verifiers import EnvHandle
from rlenv_audit.checks.base import CheckResult, CheckStatus

# (name, hf_id, config, split, question_field)
REFERENCE_SETS: list[tuple[str, str, str | None, str, str]] = [
    ("gsm8k", "openai/gsm8k", "main", "test", "question"),
    ("math500", "HuggingFaceH4/MATH-500", None, "test", "problem"),
    ("aime2024", "Maxwell-Jia/AIME_2024", None, "train", "Problem"),
    ("humaneval", "openai/openai_humaneval", None, "test", "prompt"),
    # Script-based loader; usually unavailable on modern `datasets` — degrades
    # to "unavailable" rather than failing the check.
    ("livecodebench", "livecodebench/code_generation_lite", "release_v1", "test", "question_content"),
]

_DEFAULT_N = 10          # word-level shingle size
# Containment of a question's informative shingles in a single reference item.
# Set high (0.8): true duplicates sit near 1.0, while same-template-different-
# instance collisions (e.g. two quadratics sharing boilerplate phrasing but
# different numbers) land lower and are correctly NOT flagged.
_DEFAULT_THRESHOLD = 0.8
_DEFAULT_MAX_REF = 1000   # cap problems loaded per reference set
_DEFAULT_MAX_DATASET = 500  # cap env questions scanned


def _normalize(text: str) -> list[str]:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text))
    return text.split()


def _shingles(words: list[str], n: int) -> set[str]:
    if len(words) < n:
        # Too short to n-gram — treat the whole thing as one shingle (exact match).
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _question_text(row: dict) -> str:
    """Best-effort extraction of the human task text from a normalized env row."""
    raw = row.get("raw", {})
    for key in ("question", "problem", "prompt", "query", "text"):
        if isinstance(raw.get(key), str) and raw[key].strip():
            return raw[key]
    # Fall back to the chat prompt's user turns.
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        parts = [
            m.get("content", "")
            for m in prompt
            if isinstance(m, dict) and m.get("role") in ("user", "system")
        ]
        return "\n".join(p for p in parts if isinstance(p, str))
    if isinstance(prompt, str):
        return prompt
    return ""


def _load_reference_index(n: int, max_ref: int, sets: list[str] | None):
    """Build ``shingle -> (set_name, ref_idx, excerpt)`` plus a document-frequency
    map over the reference sets.

    ``df[shingle]`` = number of distinct reference items containing it. A shingle
    in many items is shared boilerplate (problem templates, instruction prefixes),
    not evidence of contamination, so the matcher later drops high-df shingles.

    Returns ``(index, df, loaded, unavailable, truncated, total_items)``.
    """
    from datasets import load_dataset

    index: dict[str, tuple[str, int, str]] = {}
    df: dict[str, int] = {}
    loaded: dict[str, int] = {}
    unavailable: dict[str, str] = {}
    truncated: list[str] = []
    total_items = 0

    for name, hf_id, config, split, field in REFERENCE_SETS:
        if sets and name not in sets:
            continue
        try:
            ds = (
                load_dataset(hf_id, config, split=split, trust_remote_code=True)
                if config
                else load_dataset(hf_id, split=split, trust_remote_code=True)
            )
        except Exception as exc:
            unavailable[name] = f"{type(exc).__name__}: {exc}"[:160]
            continue

        if field not in ds.column_names:
            unavailable[name] = f"field '{field}' missing"
            continue

        if len(ds) > max_ref:
            truncated.append(f"{name} ({len(ds)}->{max_ref})")
            ds = ds.select(range(max_ref))

        loaded[name] = len(ds)
        total_items += len(ds)
        for idx, text in enumerate(ds[field]):
            sh = _shingles(_normalize(text), n)  # a set: each shingle once per item
            excerpt = " ".join(str(text).split())[:120]
            for s in sh:
                index.setdefault(s, (name, idx, excerpt))
                df[s] = df.get(s, 0) + 1

    return index, df, loaded, unavailable, truncated, total_items


def check_contamination(handle: EnvHandle, config: dict) -> CheckResult:
    n = int(config.get("contamination_n", _DEFAULT_N))
    threshold = float(config.get("contamination_threshold", _DEFAULT_THRESHOLD))
    max_ref = int(config.get("contamination_max_ref", _DEFAULT_MAX_REF))
    max_dataset = int(config.get("contamination_max_dataset", _DEFAULT_MAX_DATASET))
    only_sets = config.get("contamination_sets")

    rows = handle.dataset(n=max_dataset)
    if not rows:
        return CheckResult("contamination", CheckStatus.SKIP, "environment exposes no dataset")

    try:
        index, df, loaded, unavailable, truncated, total_items = _load_reference_index(
            n, max_ref, only_sets
        )
    except Exception as exc:
        return CheckResult(
            "contamination", CheckStatus.SKIP,
            f"could not load any reference eval set: {exc}",
        )

    if not loaded:
        return CheckResult(
            "contamination", CheckStatus.SKIP,
            f"no reference eval sets available ({', '.join(unavailable) or 'none'})",
            details={"unavailable": unavailable},
        )

    # A shingle appearing in this many reference items is boilerplate (shared
    # template / instruction text), not a fingerprint of a specific problem.
    df_cutoff = max(2, int(config.get("contamination_df_cutoff", 0.01 * total_items)))

    matches: list[dict] = []
    scanned = 0
    for i, row in enumerate(rows):
        text = _question_text(row)
        sh = _shingles(_normalize(text), n)
        if not sh:
            continue
        scanned += 1

        # Keep only informative (non-boilerplate) shingles, then tally shared
        # ones per reference item; containment is measured over those.
        informative = [s for s in sh if df.get(s, 0) <= df_cutoff]
        if not informative:
            continue
        per_ref: dict[tuple[str, int], int] = {}
        excerpt_of: dict[tuple[str, int], str] = {}
        for s in informative:
            hit = index.get(s)
            if hit:
                key = (hit[0], hit[1])
                per_ref[key] = per_ref.get(key, 0) + 1
                excerpt_of[key] = hit[2]
        if not per_ref:
            continue
        (best_set, best_idx), shared = max(per_ref.items(), key=lambda kv: kv[1])
        containment = shared / len(informative)
        # Need both enough informative overlap AND high containment. The
        # min-shingle floor (capped at the question's length) blocks single
        # coincidental matches without missing short exact duplicates.
        min_shared = min(3, len(informative))
        if shared >= min_shared and containment >= threshold:
            matches.append(
                {
                    "dataset_index": i,
                    "containment": round(containment, 3),
                    "matched_set": best_set,
                    "matched_ref_index": best_idx,
                    "question_excerpt": " ".join(text.split())[:120],
                    "matched_excerpt": excerpt_of[(best_set, best_idx)],
                }
            )

    details = {
        "reference_sets_loaded": loaded,
        "reference_sets_unavailable": unavailable,
        "reference_sets_truncated": truncated,
        "questions_scanned": scanned,
        "shingle_n": n,
        "containment_threshold": threshold,
        "boilerplate_df_cutoff": df_cutoff,
        "matches": matches,
    }
    avail_note = f"checked {scanned} questions vs {', '.join(loaded)}"
    if unavailable:
        avail_note += f" (unavailable: {', '.join(unavailable)})"

    if matches:
        by_set: dict[str, int] = {}
        for m in matches:
            by_set[m["matched_set"]] = by_set.get(m["matched_set"], 0) + 1
        breakdown = ", ".join(f"{k}:{v}" for k, v in by_set.items())
        details["recommendations"] = [
            f"Dataset overlaps known eval sets ({breakdown}). If this is a TRAINING "
            "env, remove the overlapping items so eval scores stay honest. If it's "
            "an EVAL env, this is expected — just don't train on it "
            "(REWARD_DESIGN.md §contamination)."
        ]
        return CheckResult(
            "contamination", CheckStatus.FAIL,
            f"{len(matches)} dataset items overlap eval sets ({breakdown}); {avail_note}",
            score=0.0, details=details,
        )
    return CheckResult(
        "contamination", CheckStatus.PASS,
        f"no overlap found; {avail_note}",
        score=1.0, details=details,
    )


from rlenv_audit.checks.base import CheckSpec  # noqa: E402

SPEC = CheckSpec(
    name="contamination",
    func=check_contamination,
    description="dataset tasks don't overlap popular eval sets (AIME/MATH/GSM8K/HumanEval/LCB)",
    needs_gpu=False,
    needs_docker=False,
)
