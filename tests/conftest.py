"""Shared fixtures. The adapter/tools tests are integration tests: they load the
real `gsm8k` Hub environment (install with `vf-install gsm8k -r`) and SKIP
gracefully if it isn't available."""

import pytest

from rlenv_audit.adapters.verifiers import EnvLoadError, load_handle


@pytest.fixture(scope="session")
def gsm8k_handle():
    try:
        handle = load_handle("gsm8k")
    except EnvLoadError as exc:
        pytest.skip(f"gsm8k env not installed ({exc}); run `vf-install gsm8k -r`")
    yield handle
    handle.teardown()
