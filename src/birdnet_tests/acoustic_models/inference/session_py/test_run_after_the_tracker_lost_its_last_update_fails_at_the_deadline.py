"""A run whose tracker finished but whose last progress update is lost fails in time.

With a progress callback, the dispatcher thread in the parent ends only when
it reads the tracker's last update, the one flagged finished. The tracker
hands that update to its queue's feeder thread and then sets its finish
signal, so a tracker killed after the signal and before the feeder wrote the
update leaves the signal set and the update unsent. The liveness check skips
a signalled child on purpose, so nothing ended the parent's wait for the
dispatcher and `run()` never returned.

The deadline is for that update, not for the callback the dispatcher calls
with it or with any update before it: a callback slower than the deadline
holds the run without failing it. A callback that raises a `BaseException`
that is not an `Exception` fails the run at once with that exception named,
and still ends the dispatcher thread.

The lost update is simulated rather than timed: the dispatcher's
`get_last_stats` is patched in the parent to drop any update flagged
finished, which leaves the dispatcher polling an empty queue exactly as the
lost update would. The run and the exit each execute on a daemon thread so
that a hang fails the test instead of wedging it; on failure the session's
remaining children are killed rather than joined.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Event
from pathlib import Path
from queue import Empty

import pytest

from birdnet.acoustic.inference import process_manager
from birdnet.acoustic.inference.core.perf_tracker import (
  AcousticProgressStats,
  ProgressDispatcher,
)
from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticFilePredictionResult,
)
from birdnet.acoustic.inference.process_manager import ProcessManager
from birdnet.acoustic.inference.session import AcousticPredictionSession
from birdnet_tests.fake_acoustic_backend import (
  fake_predict_session,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf

_LOGGER = logging.getLogger(__name__)

# The promised-message deadline, patched down from 30 s: right for a session,
# wrong for a test that has to wait it out twice.
_DEADLINE_S = 5.0
# How much later than the deadline run() may return: the way to the deadline is
# polled once a second, and a loaded runner adds its own delay.
_MARGIN_S = 10.0
# The test's own bound on run() and __exit__. Below the unpatched 30 s on
# purpose: a wait that ignores the patched deadline fails here instead of
# passing late.
_STEP_BOUND_S = 25.0
# Lets the tracker log its finish and settle into its idle wait before the kill;
# a writer killed inside a log write keeps the logging queue's lock.
_SETTLE_S = 0.3
# Long enough for the tracker to send ordinary updates before the last one:
# 20 single-segment batches of 250 ms on 2 workers, one update per second.
_N_SEGMENTS = 20
# Past the deadline, so a wait that still bounds the callback fails the run.
_SLOW_CALLBACK_S = _DEADLINE_S + 2.0
# For a run that has to be cut short: 120 single-segment batches of 250 ms on 2
# workers sleep for at least 15 s however loaded the machine is, so a run()
# that returns sooner did not compute to its end.
_LONG_RUN_SEGMENTS = 120
_LONG_RUN_MIN_S = 15.0
# For the runs that must fail at once: beyond the bound on run(), so a wait
# that only gives up at the deadline fails that bound.
_LONG_DEADLINE_S = 60.0
# How soon run() fails once the callback has raised: one 1 s poll of the cancel
# event and the way out of run().
_NOTICE_S = 2.0


class _Step:
  """Runs `step` on a daemon thread, so a hang fails the test instead of wedging it."""

  def __init__(self, step: Callable[[], object], name: str) -> None:
    self.done = threading.Event()
    self.error: BaseException | None = None
    self.result: object = None
    self.finished_at: float | None = None

    def call() -> None:
      try:
        self.result = step()
      except BaseException as e:  # noqa: BLE001
        self.error = e
      finally:
        self.finished_at = time.monotonic()
        self.done.set()

    threading.Thread(target=call, name=name, daemon=True).start()


@dataclass
class _Entered:
  session: AcousticPredictionSession
  manager: ProcessManager
  tracker: BaseProcess
  tracker_finished: Event
  audio: Path


@contextmanager
def _entered_session(
  tmp_path: Path,
  progress_callback: Callable[[AcousticProgressStats], None],
  n_segments: int = _N_SEGMENTS,
) -> Iterator[_Entered]:
  """A progress session, left on a `_Step` after the body; killed on failure."""
  audio = write_marked_audio(tmp_path / "in.wav", n_segments)
  session: AcousticPredictionSession = fake_predict_session(
    tmp_path,
    n_workers=2,
    show_stats="progress",
    seconds_per_batch=0.25,
    progress_callback=progress_callback,
  )
  session.__enter__()
  manager = session._process_manager  # noqa: SLF001
  assert manager is not None
  tracker = manager._perf_tracker_process  # noqa: SLF001
  assert tracker is not None
  tracker_finished = session._resources.stats_resources.perf_res_finish_signal  # noqa: SLF001
  assert tracker_finished is not None
  children = [
    *(manager._producer_processes or []),  # noqa: SLF001
    *(manager._worker_processes or []),  # noqa: SLF001
    tracker,
  ]

  left = False
  try:
    yield _Entered(session, manager, tracker, tracker_finished, audio)

    exit_ = _Step(lambda: session.__exit__(None, None, None), "__exit__")
    assert exit_.done.wait(_STEP_BOUND_S), (
      f"__exit__ did not return within {_STEP_BOUND_S:.0f} s"
    )
    assert exit_.error is None, exit_.error
    left = True
  finally:
    if not left:
      for child in children:
        child.kill()


def _drop_finished_updates(monkeypatch: pytest.MonkeyPatch) -> None:
  """Make the dispatcher never see the tracker's last update, as if it were lost.

  Re-implemented rather than wrapped: the original coalesces every update in
  flight into the latest one, and on a short run the ordinary updates reach
  the dispatcher in the same drain as the last, so dropping the drain's result
  would lose them too and the callback would never run.
  """

  def get_last_stats(self: ProgressDispatcher) -> AcousticProgressStats | None:
    latest = None
    while True:
      try:
        update = self._callback_queue.get(  # noqa: SLF001
          block=True,
          timeout=self._check_interval,  # noqa: SLF001
        )
      except Empty:
        return latest
      if not update.finished:
        latest = update

  monkeypatch.setattr(ProgressDispatcher, "get_last_stats", get_last_stats)


def _deliver_each_update(monkeypatch: pytest.MonkeyPatch) -> None:
  """Hand the callback every update on its own, not the latest of a drain.

  The original returns only once the queue has been quiet for a whole check
  interval, about as long as the tracker waits between updates, so on a short
  run the callback often sees nothing before the closing update.
  """

  def get_last_stats(self: ProgressDispatcher) -> AcousticProgressStats | None:
    try:
      return self._callback_queue.get(  # noqa: SLF001
        block=True,
        timeout=self._check_interval,  # noqa: SLF001
      )
    except Empty:
      return None

  monkeypatch.setattr(ProgressDispatcher, "get_last_stats", get_last_stats)


def _fails_at_the_deadline(
  entered: _Entered,
  updates: list[AcousticProgressStats],
  *,
  kill_tracker_after_it_finished: bool,
) -> None:
  """Run once more, its last update lost, and check it fails at the deadline."""
  tracker = entered.tracker
  run = _Step(lambda: entered.session.run([str(entered.audio)]), "run()")
  # A later run starts with the previous run's finish signal still set, until
  # its reset clears it.
  cleared_by = time.monotonic() + _STEP_BOUND_S
  while entered.tracker_finished.is_set():
    assert time.monotonic() < cleared_by, "the tracker's finish signal stayed set"
    time.sleep(0.01)
  assert entered.tracker_finished.wait(_STEP_BOUND_S), "the tracker never finished"
  tracker_finished_at = time.monotonic()
  if kill_tracker_after_it_finished:
    time.sleep(_SETTLE_S)
    tracker.kill()
    tracker.join()

  # Not before the deadline, which counts from the tracker's finish signal: a
  # wait that fails a signalled child at once is the liveness exception the
  # design rules out, and would fail healthy runs whose last update is merely
  # late.
  before_deadline_s = _DEADLINE_S - 1.0 - (time.monotonic() - tracker_finished_at)
  assert not run.done.wait(max(0.0, before_deadline_s)), (
    "run() gave up before the deadline"
  )
  assert run.done.wait(_STEP_BOUND_S), (
    f"run() did not return within {_STEP_BOUND_S:.0f} s of the tracker "
    f"finishing; the wait for the progress dispatcher is unbounded"
  )
  elapsed = time.monotonic() - tracker_finished_at
  _LOGGER.info(
    "run() returned %.1f s after the tracker finished (deadline %.0f s)",
    elapsed,
    _DEADLINE_S,
  )
  assert elapsed <= _DEADLINE_S + _MARGIN_S, (
    f"run() returned {elapsed:.1f} s after the tracker finished, later than "
    f"the {_DEADLINE_S:.0f} s deadline plus {_MARGIN_S:.0f} s margin"
  )

  error = run.error
  assert isinstance(error, RuntimeError), error
  assert f"Pipeline process '{tracker.name}'" in str(error)
  if kill_tracker_after_it_finished:
    assert "exited unexpectedly" in str(error)
    assert f"exit code {tracker.exitcode}" in str(error)
    assert "still alive" not in str(error)
  else:
    assert "still alive" in str(error)
    assert "exited unexpectedly" not in str(error)

  assert updates, "the progress callback never ran before the failure"
  assert not any(update.finished for update in updates)


def _run_with_lost_last_update(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  *,
  kill_tracker_after_it_finished: bool,
) -> None:
  monkeypatch.setattr(process_manager, "_PROMISED_MESSAGE_DEADLINE_S", _DEADLINE_S)
  _drop_finished_updates(monkeypatch)
  updates: list[AcousticProgressStats] = []

  with _entered_session(tmp_path, updates.append) as entered:
    _fails_at_the_deadline(
      entered,
      updates,
      kill_tracker_after_it_finished=kill_tracker_after_it_finished,
    )


def test_a_tracker_killed_after_finishing_is_reported_dead(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """The regression: the tracker exited after signalling, its last update unsent."""
  _run_with_lost_last_update(tmp_path, monkeypatch, kill_tracker_after_it_finished=True)


def test_a_live_tracker_whose_last_update_never_arrives_is_reported_alive(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _run_with_lost_last_update(
    tmp_path, monkeypatch, kill_tracker_after_it_finished=False
  )


def test_a_later_run_whose_last_update_never_arrives_fails_at_the_deadline(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """The first run's closing update does not stand in for the second's."""
  monkeypatch.setattr(process_manager, "_PROMISED_MESSAGE_DEADLINE_S", _DEADLINE_S)
  updates: list[AcousticProgressStats] = []

  with _entered_session(tmp_path, updates.append) as entered:
    first = _Step(lambda: entered.session.run([str(entered.audio)]), "first run()")
    assert first.done.wait(_STEP_BOUND_S), (
      f"the first run() did not return within {_STEP_BOUND_S:.0f} s"
    )
    assert first.error is None, first.error
    assert any(update.finished for update in updates), (
      "the first run's closing update never reached the callback"
    )

    _drop_finished_updates(monkeypatch)
    updates.clear()
    _fails_at_the_deadline(entered, updates, kill_tracker_after_it_finished=False)


def test_a_closing_callback_slower_than_the_deadline_does_not_fail_the_run(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(process_manager, "_PROMISED_MESSAGE_DEADLINE_S", _DEADLINE_S)
  updates: list[AcousticProgressStats] = []
  closing_callback_returned_at: list[float] = []

  def callback(update: AcousticProgressStats) -> None:
    updates.append(update)
    if update.finished:
      time.sleep(_SLOW_CALLBACK_S)
      closing_callback_returned_at.append(time.monotonic())

  with _entered_session(tmp_path, callback) as entered:
    run = _Step(lambda: entered.session.run([str(entered.audio)]), "run()")
    assert entered.tracker_finished.wait(_STEP_BOUND_S), "the tracker never finished"
    tracker_finished_at = time.monotonic()
    assert run.done.wait(_STEP_BOUND_S), (
      f"run() did not return within {_STEP_BOUND_S:.0f} s of the tracker finishing"
    )
    assert run.finished_at is not None
    elapsed = run.finished_at - tracker_finished_at
    _LOGGER.info(
      "run() returned %.1f s after the tracker finished (deadline %.0f s, "
      "closing callback %.0f s)",
      elapsed,
      _DEADLINE_S,
      _SLOW_CALLBACK_S,
    )

    assert run.error is None, f"the slow closing callback failed the run: {run.error}"
    result = run.result
    assert isinstance(result, AcousticFilePredictionResult), result
    assert result.n_inputs == 1
    assert result.max_n_segments == _N_SEGMENTS

    assert [update.finished for update in updates].count(True) == 1, updates
    assert updates[-1].finished, "the closing update was not the last one"
    assert updates[-1].progress_pct == 100.0
    # Still empty if run() returned while the closing callback was asleep.
    callback_returned_at = (
      closing_callback_returned_at[0] if closing_callback_returned_at else float("inf")
    )
    assert run.finished_at >= callback_returned_at, (
      "run() returned before the closing callback did"
    )


def test_an_ordinary_callback_slower_than_the_deadline_does_not_fail_the_run(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """The tracker finishes while the callback still holds an earlier update.

  The closing update waits in the queue until the callback returns: the
  dispatcher cannot read it before, and that time is the caller's, not the
  message's.
  """
  monkeypatch.setattr(process_manager, "_PROMISED_MESSAGE_DEADLINE_S", _DEADLINE_S)
  _deliver_each_update(monkeypatch)
  updates: list[AcousticProgressStats] = []
  tracker_seen = threading.Event()

  def callback(update: AcousticProgressStats) -> None:
    updates.append(update)
    if len(updates) == 1 and not update.finished:
      # Busy from before the tracker finishes until past the deadline counted
      # from its finish, however long the run takes.
      tracker_seen.wait(_STEP_BOUND_S)
      time.sleep(_SLOW_CALLBACK_S)

  with _entered_session(tmp_path, callback) as entered:
    run = _Step(lambda: entered.session.run([str(entered.audio)]), "run()")
    assert entered.tracker_finished.wait(_STEP_BOUND_S), "the tracker never finished"
    tracker_finished_at = time.monotonic()
    tracker_seen.set()
    assert run.done.wait(_STEP_BOUND_S), (
      f"run() did not return within {_STEP_BOUND_S:.0f} s of the tracker finishing"
    )
    assert run.finished_at is not None
    elapsed = run.finished_at - tracker_finished_at
    _LOGGER.info(
      "run() returned %.1f s after the tracker finished (deadline %.0f s, "
      "ordinary callback %.0f s)",
      elapsed,
      _DEADLINE_S,
      _SLOW_CALLBACK_S,
    )

    assert run.error is None, f"the slow ordinary callback failed the run: {run.error}"
    result = run.result
    assert isinstance(result, AcousticFilePredictionResult), result
    assert result.n_inputs == 1
    assert result.max_n_segments == _N_SEGMENTS

    assert not updates[0].finished, "the first update was the closing one"
    assert [update.finished for update in updates].count(True) == 1, updates
    assert updates[-1].finished, "the closing update was not the last one"
    assert updates[-1].progress_pct == 100.0
    assert elapsed > _DEADLINE_S, (
      "the callback did not hold the closing update past the deadline"
    )


class _CallbackAbort(BaseException):
  """Not an `Exception`, so the dispatcher's handler lets it end the thread."""


_ABORT_TEXT = "stop"


@pytest.mark.parametrize(
  "on_closing_update",
  [False, True],
  ids=["raised-on-the-first-update", "raised-on-the-closing-update"],
)
def test_a_callback_raising_a_base_exception_fails_the_run_at_once(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, on_closing_update: bool
) -> None:
  """The run fails as soon as the callback raises, with the exception named.

  Raised on the first update, the run is cut short long before its work is
  done; raised on the closing update, the wait for the dispatcher to finish,
  which has no deadline, ends at once.
  """
  monkeypatch.setattr(process_manager, "_PROMISED_MESSAGE_DEADLINE_S", _LONG_DEADLINE_S)
  if not on_closing_update:
    _deliver_each_update(monkeypatch)
  n_segments = _N_SEGMENTS if on_closing_update else _LONG_RUN_SEGMENTS
  raised: list[tuple[AcousticProgressStats, float]] = []

  def callback(update: AcousticProgressStats) -> None:
    if update.finished or not on_closing_update:
      raised.append((update, time.monotonic()))
      raise _CallbackAbort(_ABORT_TEXT)

  # pytest turns an exception that ends a thread into a warning after the test
  # phase. This one is the point of the test, so it is taken and asserted on
  # here; any other still reaches pytest.
  thread_deaths: list[str | None] = []
  forward = threading.excepthook

  def excepthook(args: threading.ExceptHookArgs) -> None:
    if args.exc_type is _CallbackAbort:
      thread_deaths.append(None if args.thread is None else args.thread.name)
      return
    forward(args)

  monkeypatch.setattr(threading, "excepthook", excepthook)

  with _entered_session(tmp_path, callback, n_segments) as entered:
    dispatcher = entered.manager._progress_dispatcher_thread  # noqa: SLF001
    assert dispatcher is not None
    session_log = entered.session._resources.logging_resources.session_log_file  # noqa: SLF001
    run_started_at = time.monotonic()
    run = _Step(lambda: entered.session.run([str(entered.audio)]), "run()")
    assert run.done.wait(_STEP_BOUND_S), (
      f"run() did not return within {_STEP_BOUND_S:.0f} s"
    )
    assert run.finished_at is not None
    assert raised, "the progress callback never raised"
    update, raised_at = raised[0]
    assert update.finished is on_closing_update
    _LOGGER.info(
      "run() returned %.1f s after the raise and %.1f s after it started",
      run.finished_at - raised_at,
      run.finished_at - run_started_at,
    )
    if not on_closing_update:
      run_s = run.finished_at - run_started_at
      assert run_s < _LONG_RUN_MIN_S, (
        f"run() returned {run_s:.1f} s after it started, no sooner than the "
        f"whole run takes; the raise did not cut it short"
      )
    after_raise_s = run.finished_at - raised_at
    assert after_raise_s <= _NOTICE_S + _MARGIN_S, (
      f"run() returned {after_raise_s:.1f} s after the callback raised, later "
      f"than {_NOTICE_S:.0f} s plus {_MARGIN_S:.0f} s margin"
    )

    error = run.error
    assert isinstance(error, RuntimeError), error
    assert isinstance(error.__cause__, _CallbackAbort), error.__cause__
    text = str(error)
    assert text.startswith(f"{_CallbackAbort.__name__}: {_ABORT_TEXT}\n\n"), text
    assert "Analysis was cancelled" in text
    assert "Progress dispatcher thread" not in text
    assert "exited unexpectedly" not in text
    assert "still alive" not in text

    # Re-raised after the cancel, so the thread may still be on its way out.
    dispatcher.join(_NOTICE_S)
    assert not dispatcher.is_alive(), "the exception did not end the dispatcher"
    assert thread_deaths == [dispatcher.name]

  # The traceback has to be in the session log the error points at, not only
  # on stderr.
  assert _CallbackAbort.__name__ in session_log.read_text(encoding="utf-8"), (
    "the session log does not show the exception that ended the dispatcher"
  )
