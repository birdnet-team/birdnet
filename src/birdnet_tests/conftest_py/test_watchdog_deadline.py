import pytest

from birdnet_tests import conftest
from birdnet_tests.conftest import (
  LOAD_MODEL_TIMEOUT_S,
  WATCHDOG_IDLE_S,
  WATCHDOG_MARGIN_S,
  _watchdog_deadline,
)

_NOW = 1000.0
_SHORT = 600.0


def test_nothing_running_uses_the_idle_deadline() -> None:
  assert _watchdog_deadline({}, _NOW) == WATCHDOG_IDLE_S


def test_single_test_gets_its_remaining_time_plus_margin() -> None:
  deadline = _watchdog_deadline({"a": _NOW + _SHORT}, _NOW)

  assert deadline == _SHORT + WATCHDOG_MARGIN_S


def test_concurrent_tests_use_the_latest_deadline() -> None:
  in_flight = {"short": _NOW + _SHORT, "download": _NOW + LOAD_MODEL_TIMEOUT_S}

  deadline = _watchdog_deadline(in_flight, _NOW)

  assert deadline == LOAD_MODEL_TIMEOUT_S + WATCHDOG_MARGIN_S


def test_a_finished_sibling_does_not_shorten_a_running_download() -> None:
  """The controller sees every worker's events; xdist runs tests concurrently."""
  in_flight = {"short": _NOW + _SHORT, "download": _NOW + LOAD_MODEL_TIMEOUT_S}

  in_flight.pop("short")

  assert _watchdog_deadline(in_flight, _NOW) == LOAD_MODEL_TIMEOUT_S + WATCHDOG_MARGIN_S


def test_a_sibling_event_does_not_extend_a_running_download() -> None:
  """Re-arming mid-test must not hand the test a fresh full budget."""
  in_flight = {"download": _NOW + LOAD_MODEL_TIMEOUT_S}

  elapsed = 500.0
  deadline = _watchdog_deadline(in_flight, _NOW + elapsed)

  assert deadline == LOAD_MODEL_TIMEOUT_S - elapsed + WATCHDOG_MARGIN_S


def test_an_overdue_test_arms_only_the_margin() -> None:
  deadline = _watchdog_deadline({"wedged": _NOW - 10}, _NOW)

  assert deadline == WATCHDOG_MARGIN_S


def test_deadline_returns_to_idle_once_the_last_test_finishes() -> None:
  in_flight = {"download": _NOW + LOAD_MODEL_TIMEOUT_S}

  in_flight.pop("download")

  assert _watchdog_deadline(in_flight, _NOW) == WATCHDOG_IDLE_S


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

  assert armed[-1] == pytest.approx(LOAD_MODEL_TIMEOUT_S + WATCHDOG_MARGIN_S, abs=1)

  conftest.pytest_runtest_logfinish("download", ())

  assert armed[-1] == WATCHDOG_IDLE_S


def test_controller_uses_the_fallback_when_the_nodeid_is_unknown(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Only workers collect, so every controller lookup misses the timeout map."""
  armed: list[float] = []
  monkeypatch.setattr(conftest, "_watchdog_file", None)
  monkeypatch.setattr(conftest, "_watchdog_in_flight", {})
  monkeypatch.setattr(conftest, "_watchdog_arm", armed.append)
  monkeypatch.setattr(conftest, "_watchdog_test_timeouts", {})
  monkeypatch.setattr(conftest, "_watchdog_fallback_s", LOAD_MODEL_TIMEOUT_S)

  conftest.pytest_runtest_logstart("some/unknown/test.py::test_x", ())

  assert armed[-1] == pytest.approx(LOAD_MODEL_TIMEOUT_S + WATCHDOG_MARGIN_S, abs=1)


def test_a_crashed_test_is_dropped_from_the_in_flight_set(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """xdist reports a killed worker's test through handlecrashitem, not logfinish."""
  armed: list[float] = []
  in_flight: dict[str, float] = {}
  monkeypatch.setattr(conftest, "_watchdog_file", None)
  monkeypatch.setattr(conftest, "_watchdog_in_flight", in_flight)
  monkeypatch.setattr(conftest, "_watchdog_arm", armed.append)
  monkeypatch.setattr(conftest, "_watchdog_test_timeouts", {"doomed": _SHORT})

  conftest.pytest_runtest_logstart("doomed", ())
  conftest.pytest_handlecrashitem("doomed", report=None, sched=None)  # type: ignore[arg-type]

  assert in_flight == {}
  assert armed[-1] == WATCHDOG_IDLE_S
