import multiprocessing as mp
from unittest.mock import patch

import pytest

from birdnet.core.start_method import resolve_start_method
from birdnet.globals import ENV_VAR_START_METHOD


def test_env_var_wins_over_global(monkeypatch: pytest.MonkeyPatch) -> None:
  # the autouse conftest fixture has fixed the global to "spawn"
  other = next(m for m in mp.get_all_start_methods() if m != "spawn")
  monkeypatch.setenv(ENV_VAR_START_METHOD, other)
  assert resolve_start_method() == other


def test_env_var_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv(ENV_VAR_START_METHOD, "teleport")
  with pytest.raises(ValueError, match=ENV_VAR_START_METHOD):
    resolve_start_method()


def test_explicitly_fixed_global_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delenv(ENV_VAR_START_METHOD, raising=False)
  # the autouse conftest fixture has fixed the global to "spawn"
  assert mp.get_start_method() == "spawn"
  assert resolve_start_method() == "spawn"


def test_fork_opt_in_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delenv(ENV_VAR_START_METHOD, raising=False)
  if "fork" not in mp.get_all_start_methods():
    pytest.skip("fork not available on this platform")
  mp.set_start_method("fork", force=True)
  try:
    assert resolve_start_method() == "fork"
  finally:
    mp.set_start_method("spawn", force=True)


def test_unset_global_falls_back_to_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delenv(ENV_VAR_START_METHOD, raising=False)
  # The global default cannot be un-fixed through a public API, so simulate a
  # fresh interpreter where no method has been chosen yet.
  with patch("birdnet.core.start_method.mp.get_start_method", return_value=None):
    assert resolve_start_method() == "spawn"


def test_resolver_does_not_fix_the_global_default() -> None:
  # resolve_start_method must read with allow_none=True; verify it passes the
  # flag instead of fixing the platform default as a side effect.
  with patch(
    "birdnet.core.start_method.mp.get_start_method", return_value=None
  ) as get_sm:
    resolve_start_method()
  get_sm.assert_called_once_with(allow_none=True)
