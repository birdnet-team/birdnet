"""A reused session's next run starts with every signal of the last run cleared.

The tracker process and the dispatcher thread each set their finish signal once
per run, and the parent's waits return as soon as a signal is set. One carried
over from the last run ends the wait at once, so the run proceeds while the
tracker is still working. Pinned per signal here: end to end, an instant
progress callback masks a stale signal by timing.
"""

from types import SimpleNamespace

import pytest

from birdnet.acoustic.inference.core.sync import read_failure, write_failure
from birdnet.acoustic.inference.resources import StatisticsResources

pytestmark = pytest.mark.no_tf


def _create(show_stats: str | None, *, with_callback: bool) -> StatisticsResources:
  config = SimpleNamespace(
    start_method="spawn",
    output_conf=SimpleNamespace(
      show_stats=show_stats,
      progress_callback=(lambda stats: None) if with_callback else None,
    ),
  )
  return StatisticsResources.create(
    session_id="session",
    conf=config,  # type: ignore[arg-type]
    benchmark_dir_name="unused",
  )


def test_reset_clears_every_signal_of_the_last_run() -> None:
  resources = _create("progress", with_callback=True)
  assert resources.perf_res_finish_signal is not None
  assert resources.perf_res_start_signal is not None
  assert resources.perf_failure_slot is not None
  assert resources.callback_finish_signal is not None
  assert resources.callback_last_update_signal is not None
  assert resources.callback_busy_signal is not None

  resources.perf_res_finish_signal.set()
  resources.perf_res_start_signal.release()
  resources.perf_res_start_signal.release()
  write_failure(resources.perf_failure_slot, RuntimeError("left over"))
  resources.callback_finish_signal.set()
  resources.callback_last_update_signal.set()
  resources.callback_busy_signal.set()
  resources.save_end_time()

  resources.reset()

  assert not resources.perf_res_finish_signal.is_set(), "tracker finish signal"
  assert not resources.perf_res_start_signal.acquire(block=False), (
    "tracker start permits"
  )
  assert read_failure(resources.perf_failure_slot) is None, "tracker failure slot"
  assert not resources.callback_finish_signal.is_set(), "dispatcher finish signal"
  assert not resources.callback_last_update_signal.is_set(), (
    "dispatcher last-update signal"
  )
  assert not resources.callback_busy_signal.is_set(), "dispatcher busy signal"
  assert resources.stop is None
  assert resources.end_timepoint is None


def test_reset_without_statistics_has_nothing_to_clear() -> None:
  resources = _create(None, with_callback=False)
  assert resources.perf_res_finish_signal is None
  assert resources.callback_finish_signal is None
  resources.save_end_time()

  resources.reset()

  assert resources.stop is None
