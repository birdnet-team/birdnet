import pytest

from birdnet_tests import conftest
from birdnet_tests.conftest import (
  LOAD_MODEL_TIMEOUT_S,
  WATCHDOG_IDLE_S,
  WATCHDOG_MARGIN_S,
  _watchdog_deadline,
)

_SHORT = 600.0


def test_logfinish_keeps_the_deadline_of_a_still_running_test(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The failure this guards: a sibling's finish killing a live download."""
  armed: list[float] = []
  # Swap the live state out; the hooks read these as module globals.
  monkeypatch.setattr(conftest, "_watchdog_file", None)
  monkeypatch.setattr(conftest, "_watchdog_in_flight", {})
  monkeypatch.setattr(conftest, "_watchdog_arm", armed.append)
  monkeypatch.setattr(
    conftest,
    "_watchdog_test_timeouts",
    {"short": _SHORT, "download": LOAD_MODEL_TIMEOUT_S},
  )

  conftest.pytest_runtest_logstart("download", ())
  conftest.pytest_runtest_logstart("short", ())
  conftest.pytest_runtest_logfinish("short", ())

  assert armed[-1] == LOAD_MODEL_TIMEOUT_S + WATCHDOG_MARGIN_S

  conftest.pytest_runtest_logfinish("download", ())

  assert armed[-1] == WATCHDOG_IDLE_S


def test_nothing_running_uses_the_idle_deadline() -> None:
  assert _watchdog_deadline({}) == WATCHDOG_IDLE_S


def test_single_test_gets_its_own_timeout_plus_margin() -> None:
  deadline = _watchdog_deadline({"a": _SHORT})

  assert deadline == _SHORT + WATCHDOG_MARGIN_S


def test_concurrent_tests_use_the_longest_timeout() -> None:
  deadline = _watchdog_deadline({"short": _SHORT, "download": LOAD_MODEL_TIMEOUT_S})

  assert deadline == LOAD_MODEL_TIMEOUT_S + WATCHDOG_MARGIN_S


def test_a_finished_sibling_does_not_shorten_a_running_download() -> None:
  """The controller sees every worker's events; xdist runs tests concurrently."""
  in_flight = {"short": _SHORT, "download": LOAD_MODEL_TIMEOUT_S}

  in_flight.pop("short")

  assert _watchdog_deadline(in_flight) == LOAD_MODEL_TIMEOUT_S + WATCHDOG_MARGIN_S


def test_deadline_returns_to_idle_once_the_last_test_finishes() -> None:
  in_flight = {"download": LOAD_MODEL_TIMEOUT_S}

  in_flight.pop("download")

  assert _watchdog_deadline(in_flight) == WATCHDOG_IDLE_S
