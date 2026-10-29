"""`end()` sets the start signal of every thread the session parks between runs.

The input analyzer, the progress dispatcher and the file-completion dispatcher
are threads of the parent that poll a `threading.Event` with a 1 s timeout
between runs. A thread woken by `end()` returns before it clears that event,
so the event is still set right after `end()`; a thread the wake skipped sees
the end event only when its poll expires, which the exit-time test can miss
when the run itself used up most of that poll. The child processes' start
signals are semaphores the children consume on the wake, so they are covered
by the exit-time and kill tests instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from birdnet_tests.fake_acoustic_backend import (
  fake_predict_session,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf


def test_every_thread_side_start_signal_is_set_after_end(tmp_path: Path) -> None:
  audio = write_marked_audio(tmp_path / "in.wav", 4)
  session = fake_predict_session(
    tmp_path,
    n_workers=2,
    show_stats="progress",
    progress_callback=lambda stats: None,
    on_file_complete=lambda result: None,
  )
  with session:
    session.run([str(audio)])
    resources = session._resources  # noqa: SLF001
    assert resources is not None
    assert not resources.analyzer_resources.start_signal.is_set()
    session.end()
    assert resources.analyzer_resources.start_signal.is_set()
    assert resources.stats_resources.callback_start_signal is not None
    assert resources.stats_resources.callback_start_signal.is_set()
    assert resources.file_completion_resources.start_signal is not None
    assert resources.file_completion_resources.start_signal.is_set()
