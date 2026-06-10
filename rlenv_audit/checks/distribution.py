"""distribution check — what does the reward distribution look like under a real
policy?

A healthy RL environment gives a spread of rewards a small model can climb. We
generate rollouts with a compact reference model (Qwen-style ~1.5B via vLLM),
score them, and histogram the rewards. We WARN on degenerate shapes that produce
no learning signal:

* **all-zero** — too hard or broken; the gradient is flat at the bottom.
* **all-one** — trivial; the gradient is flat at the top.
* **empty-rewarded** — an empty/blank response already earns reward.

Needs a GPU + vLLM. If either is missing (e.g. the box's CUDA is too old), the
check degrades to SKIP — it never fails the audit over missing hardware.
"""

from __future__ import annotations

from rlenv_audit.adapters.verifiers import EnvHandle, ScoringError
from rlenv_audit.checks.base import CheckResult, CheckStatus

_DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
_DEFAULT_N = 32


def _vllm_available() -> tuple[bool, str]:
    import importlib.util

    if importlib.util.find_spec("vllm") is None:
        return False, "vLLM not installed (install rlenv-audit[gpu])"
    return True, "ok"


def _gpu_available() -> tuple[bool, str]:
    try:
        import torch
    except Exception:
        return False, "torch not installed"
    try:
        if not torch.cuda.is_available():
            return False, "no CUDA GPU available"
    except Exception as exc:
        return False, f"CUDA check failed: {exc}"
    return True, "ok"


def _histogram(rewards: list[float], bins: int = 10) -> dict[str, int]:
    lo, hi = min(rewards), max(rewards)
    if hi <= lo:
        return {f"{lo:.3g}": len(rewards)}
    width = (hi - lo) / bins
    hist: dict[str, int] = {}
    for r in rewards:
        idx = min(bins - 1, int((r - lo) / width))
        label = f"[{lo+idx*width:.2g}, {lo+(idx+1)*width:.2g})"
        hist[label] = hist.get(label, 0) + 1
    return hist


def check_distribution(handle: EnvHandle, config: dict) -> CheckResult:
    ok, msg = _vllm_available()
    if not ok:
        return CheckResult("distribution", CheckStatus.SKIP, msg)
    ok, msg = _gpu_available()
    if not ok:
        return CheckResult("distribution", CheckStatus.SKIP, f"GPU required: {msg}")

    model = config.get("model") or _DEFAULT_MODEL
    n = int(config.get("distribution_n", _DEFAULT_N))
    rows = handle.dataset(n=n)
    if not rows:
        return CheckResult("distribution", CheckStatus.SKIP, "environment exposes no dataset")

    # Everything below touches the GPU/vLLM and the network; any failure degrades
    # to SKIP rather than failing the audit.
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(model=model, gpu_memory_utilization=float(config.get("gpu_mem_util", 0.6)))
        sp = SamplingParams(
            temperature=float(config.get("temperature", 0.7)),
            max_tokens=int(config.get("max_tokens", 1024)),
        )
        chats = [r["prompt"] for r in rows]
        outputs = llm.chat(chats, sp)
        completions = [o.outputs[0].text for o in outputs]
    except Exception as exc:
        return CheckResult(
            "distribution", CheckStatus.SKIP,
            f"could not run reference rollouts with vLLM ({model}): {exc}",
        )

    rewards: list[float] = []
    for r, text in zip(rows, completions):
        try:
            reward, _ = handle.score(text, r["prompt"], r["answer"], r["info"], r.get("raw"))
        except ScoringError:
            continue
        rewards.append(reward)

    if not rewards:
        return CheckResult(
            "distribution", CheckStatus.SKIP, "no rollouts could be scored",
        )

    # Does an empty response already earn reward?
    empty_reward = None
    try:
        empty_reward, _ = handle.score(
            "", rows[0]["prompt"], rows[0]["answer"], rows[0]["info"], rows[0].get("raw")
        )
    except ScoringError:
        pass

    rmin, rmax = min(rewards), max(rewards)
    pass_rate = sum(1 for x in rewards if x > 0) / len(rewards)
    details = {
        "model": model,
        "n_rollouts": len(rewards),
        "pass_rate": round(pass_rate, 3),
        "reward_min": rmin,
        "reward_max": rmax,
        "reward_mean": round(sum(rewards) / len(rewards), 4),
        "histogram": _histogram(rewards),
        "empty_response_reward": empty_reward,
    }

    warnings: list[str] = []
    if rmax <= 0:
        warnings.append("all rewards zero (too hard / broken — no gradient)")
    elif rmin >= rmax:
        warnings.append("all rewards identical (no gradient)")
    if empty_reward is not None and empty_reward > 0:
        warnings.append(f"empty response earns reward ({empty_reward})")

    if warnings:
        details["recommendations"] = [
            "Reward distribution is degenerate for the reference model "
            f"({'; '.join(warnings)}). Adjust task difficulty so a mix of rollouts "
            "pass and fail — an all-pass or all-fail batch gives zero gradient "
            "(REWARD_DESIGN.md §difficulty-curriculum)."
        ]
        return CheckResult(
            "distribution", CheckStatus.WARN,
            f"degenerate reward distribution: {'; '.join(warnings)} "
            f"(pass-rate {pass_rate:.0%}, n={len(rewards)})",
            score=pass_rate, details=details,
        )
    return CheckResult(
        "distribution", CheckStatus.PASS,
        f"reward spread healthy (pass-rate {pass_rate:.0%}, "
        f"range {rmin:.2g}-{rmax:.2g}, n={len(rewards)})",
        score=pass_rate, details=details,
    )


from rlenv_audit.checks import CheckSpec  # noqa: E402

SPEC = CheckSpec(
    name="distribution",
    func=check_distribution,
    description="reward distribution under a small reference policy isn't degenerate (all-0/all-1)",
    needs_gpu=True,
    needs_docker=False,
)
