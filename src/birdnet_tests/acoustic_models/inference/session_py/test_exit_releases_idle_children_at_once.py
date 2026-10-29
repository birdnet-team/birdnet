"""Leaving a session does not wait out the idle children's one-second poll.

Idle children and threads poll their start signal with a 1 s timeout and look
at the end event only when it expires; they re-enter that wait right before
`run()` returns, so `end()` releases every idle wait. Asserts on the fastest
of a few exits, like `test_session_call_overhead`: a loaded machine can only
make an exit slower, and the poll is a full second on any machine. A clean
exit must also be clean: a child that mistakes the wake for a run start fails
and cancels the session, and that exit is prompt too. The release also covers
a cancelled run, where the children's last log records leave through the
bounded drain in `_abandon_logging_feeder` rather than a wait on the poll.
The package logger runs at INFO, the library default: the suite raises it to
DEBUG, where the children's log traffic adds its own half second to every
exit.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from birdnet_tests.fake_acoustic_backend import (
  RaisingBackend,
  fake_predict_session,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf

N_SESSIONS = 3
MAX_FASTEST_EXIT_S = 0.5


@pytest.fixture
def info_level() -> Iterator[None]:
  # The session captures the package logger's level when it is created.
  logger = logging.getLogger("birdnet")
  previous = logger.level
  logger.setLevel(logging.INFO)
  yield
  logger.setLevel(previous)


# The second case also starts the performance tracker, the progress dispatcher
# and the file-completion dispatcher, each with its own idle poll. The third
# ends its run with the model raising, so the exit follows a cancel.
CASES: dict[str, tuple[dict[str, object], bool]] = {
  "plain": ({}, False),
  "every_idle_waiter": (
    {
      "show_stats": "progress",
      "progress_callback": lambda stats: None,
      "on_file_complete": lambda result: None,
    },
    False,
  ),
  "after_a_cancelled_run": ({"backend": RaisingBackend}, True),
}


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
@pytest.mark.usefixtures("info_level")
def test_fastest_exit_is_well_below_the_idle_poll(
  tmp_path: Path, case: tuple[dict[str, object], bool]
) -> None:
  options, cancelled = case
  # Eight segments: with fewer, a run the model fails on its second batch is
  # over before any waiter has parked again, and the cancelled exit is fast
  # even without the wake.
  audio = write_marked_audio(tmp_path / "in.wav", 8)
  durations: list[float] = []

  for i in range(N_SESSIONS):
    session_dir = tmp_path / str(i)
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
    durations.append(time.perf_counter() - start)
    if not cancelled:
      # A child that mistakes the wake for a run start fails and cancels the
      # session; that exit is prompt as well, but not clean.
      assert manager.run_error is None
      assert not resources.processing_resources.cancel_event.is_set()

  fastest = min(durations)
  assert fastest < MAX_FASTEST_EXIT_S, (
    f"fastest __exit__ took {fastest:.3f} s over {N_SESSIONS} sessions "
    f"(all: {[round(d, 3) for d in durations]}); the idle children were not "
    f"released and waited out their poll"
  )
