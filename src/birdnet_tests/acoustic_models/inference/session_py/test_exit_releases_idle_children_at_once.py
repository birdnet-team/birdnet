"""Leaving a session does not wait out the idle children's one-second poll.

Idle children and threads poll their start signal with a 1 s timeout and look
at the end event only when it expires; they re-enter that wait right before
`run()` returns, so `end()` releases every idle wait and the exit saves nearly
the whole poll. Measured against a control: the same sessions with the wake
patched out, interleaved with the real ones so a load burst hits both alike.
A loaded machine slows every exit, woken or not, but cannot shrink the poll
the control waits out, so the gap between the fastest woken and the fastest
unwoken exit is an invariant where an absolute bound was not. A clean exit
must also be clean: a child that mistakes the wake for a run start fails and
cancels the session, and that exit is prompt too. The suite runs the package
logger at DEBUG, so the children's log traffic is part of what is measured:
the parent's writer reads it continuously, and a writer that let the pipe fill
would show up here as a slow exit.

After a cancelled run the children sit in other one-second waits as well (the
producer on a free ring slot, the consumer on its own poll), which the wake
does not reach, so that exit's duration is not the wake's to decide and is not
timed; the case checks that the wake runs there at all.

With `wake_idle_children` made a no-op, both timed cases fail on the saving;
with `end()` skipping the wake after a cancel, the cancelled case fails.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from birdnet.acoustic.inference.process_manager import ProcessManager
from birdnet_tests.fake_acoustic_backend import (
  RaisingBackend,
  fake_predict_session,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf

N_SESSIONS = 3
# The children's start-signal poll, which an unwoken exit waits out.
IDLE_POLL_S = 1.0
# The wake must save most of that poll on the fastest exit of each kind.
MIN_SAVING_S = 0.5 * IDLE_POLL_S
# An unwoken exit cannot be shorter than the poll left when `end()` is called,
# which is nearly the whole poll; a control below this never parked.
MIN_UNWOKEN_S = 0.8 * IDLE_POLL_S
# A cancelled exit is bounded by the teardown budget; this only catches a hang.
MAX_CANCELLED_EXIT_S = 10.0


# The second case also starts the performance tracker, the progress dispatcher
# and the file-completion dispatcher, each with its own idle poll.
TIMED_CASES: dict[str, dict[str, object]] = {
  "plain": {},
  "every_idle_waiter": {
    "show_stats": "progress",
    "progress_callback": lambda stats: None,
    "on_file_complete": lambda result: None,
  },
}


def _timed_exit(
  session_dir: Path, audio: Path, options: dict[str, object], cancelled: bool
) -> float:
  """One session: enter, run, leave; return how long leaving took."""
  session_dir.mkdir()
  session = fake_predict_session(session_dir, n_workers=2, **options)  # type: ignore[arg-type]
  session.__enter__()
  manager = session._process_manager  # noqa: SLF001
  resources = session._resources  # noqa: SLF001
  assert manager is not None and resources is not None
  if cancelled:
    with pytest.raises(RuntimeError, match="synthetic model failure"):
      session.run([str(audio)])
  else:
    session.run([str(audio)])
  start = time.perf_counter()
  session.__exit__(None, None, None)
  duration = time.perf_counter() - start
  if not cancelled:
    # A child that mistakes the wake for a run start fails and cancels the
    # session; that exit is prompt as well, but not clean.
    assert manager.run_error is None
    assert not resources.processing_resources.cancel_event.is_set()
  return duration


@pytest.mark.parametrize("options", TIMED_CASES.values(), ids=TIMED_CASES.keys())
def test_the_wake_saves_most_of_the_idle_poll(
  tmp_path: Path, options: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
  audio = write_marked_audio(tmp_path / "in.wav", 8)
  woken: list[float] = []
  unwoken: list[float] = []

  for i in range(N_SESSIONS):
    woken.append(_timed_exit(tmp_path / f"woken{i}", audio, options, cancelled=False))
    with monkeypatch.context() as patched:
      patched.setattr(ProcessManager, "wake_idle_children", lambda self: None)
      unwoken.append(
        _timed_exit(tmp_path / f"unwoken{i}", audio, options, cancelled=False)
      )

  summary = (
    f"woken {[round(d, 3) for d in woken]}, unwoken {[round(d, 3) for d in unwoken]}"
  )
  assert min(unwoken) >= MIN_UNWOKEN_S, (
    f"the control left in {min(unwoken):.3f} s, less than the poll it should "
    f"have waited out; the children were not idle when the session ended ({summary})"
  )
  assert min(woken) <= min(unwoken) - MIN_SAVING_S, (
    f"the wake saved only {min(unwoken) - min(woken):.3f} s of the "
    f"{IDLE_POLL_S:.0f} s poll on the fastest exit; the idle children were not "
    f"released ({summary})"
  )


def test_the_wake_runs_after_a_cancelled_run(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  # Eight segments: with fewer, a run the model fails on its second batch is
  # over before any waiter has parked again.
  audio = write_marked_audio(tmp_path / "in.wav", 8)
  wakes: list[float] = []
  real_wake = ProcessManager.wake_idle_children

  def recording_wake(self: ProcessManager) -> None:
    wakes.append(time.perf_counter())
    real_wake(self)

  monkeypatch.setattr(ProcessManager, "wake_idle_children", recording_wake)

  duration = _timed_exit(
    tmp_path / "cancelled", audio, {"backend": RaisingBackend}, cancelled=True
  )

  assert len(wakes) == 1, f"the wake ran {len(wakes)} times during the exit"
  assert duration < MAX_CANCELLED_EXIT_S, (
    f"__exit__ took {duration:.1f} s after a cancel"
  )
