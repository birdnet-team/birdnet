"""A run whose result rows do not match the rows the producers wrote fails.

Each producer counts the rows it published into the ring; the consumer counts
the rows that came back. Any difference fails the run with both numbers rather
than return a result with segments missing.
"""

from __future__ import annotations

import ctypes
import logging
import threading
from types import SimpleNamespace

import pytest

from birdnet.acoustic.inference.process_manager import (
  PipelineThreadError,
  ProcessManager,
)

pytestmark = pytest.mark.no_tf


class _StubManager:
  """Only what `check_all_rows_arrived` touches."""

  def __init__(self, *rows_written: int) -> None:
    self._logger = logging.getLogger(f"{__name__}.stub")
    self._run_error: BaseException | None = None
    self._run_error_lock = threading.Lock()
    self.cancel_event = threading.Event()
    self._res = SimpleNamespace(
      producer_resources=SimpleNamespace(
        rows_written=[ctypes.c_uint64(n) for n in rows_written]
      ),
      processing_resources=SimpleNamespace(cancel_event=self.cancel_event),
    )

  check_all_rows_arrived = ProcessManager.check_all_rows_arrived
  record_run_error = ProcessManager.record_run_error
  run_error = ProcessManager.run_error


def test_rows_short_fail_the_run_with_both_numbers() -> None:
  manager = _StubManager(5, 7)

  manager.check_all_rows_arrived(10)

  assert manager.cancel_event.is_set()
  error = manager.run_error
  assert isinstance(error, PipelineThreadError), error
  assert str(error).startswith(
    "Results are missing: 12 segments went to the workers, but results for "
    "only 10 came back."
  ), error


def test_every_row_back_leaves_the_run_alone() -> None:
  manager = _StubManager(5, 7)

  manager.check_all_rows_arrived(12)

  assert not manager.cancel_event.is_set()
  assert manager.run_error is None


def test_rows_beyond_the_producers_count_fail_the_run_too() -> None:
  manager = _StubManager(5, 7)

  manager.check_all_rows_arrived(13)

  assert manager.cancel_event.is_set()
  error = manager.run_error
  assert isinstance(error, PipelineThreadError), error
  assert "12 segments went to the workers, but results for 13 came back" in str(
    error
  ), error
