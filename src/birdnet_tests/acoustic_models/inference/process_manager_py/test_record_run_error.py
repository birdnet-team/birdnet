import threading

import pytest

from birdnet.acoustic.inference.process_manager import ProcessManager

pytestmark = pytest.mark.no_tf


class _StubManager:
  """Only what `record_run_error` touches."""

  def __init__(self) -> None:
    self._run_error: BaseException | None = None
    self._run_error_lock = threading.Lock()

  record_run_error = ProcessManager.record_run_error
  run_error = ProcessManager.run_error


def test_the_first_error_is_kept() -> None:
  """Once one thread fails the run, the others fail as a consequence."""
  manager = _StubManager()
  first = ValueError("root cause")
  manager.record_run_error(first)
  manager.record_run_error(RuntimeError("consequence"))
  assert manager.run_error is first
