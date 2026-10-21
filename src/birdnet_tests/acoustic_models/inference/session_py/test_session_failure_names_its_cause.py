"""A run failed by this process's own pipeline threads says why.

The analyzer, the consumer and the two callback dispatchers run as threads of
the calling process and turn every failure into a cancelled run. The caller
must get the exception itself, not only a pointer to the session log.
"""

from pathlib import Path

import pytest

import birdnet.acoustic.inference.core.input_analyzer as input_analyzer
from birdnet.acoustic.inference.core.consumer import Consumer
from birdnet.acoustic.inference.core.input_analyzer import InputAnalyzer
from birdnet.acoustic.inference.session import AcousticPredictionSession
from birdnet_tests.fake_acoustic_backend import (
  fake_predict_session,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf


def _run_and_get_failure(
  session: AcousticPredictionSession, audio: Path
) -> BaseException:
  with (
    pytest.raises(RuntimeError, match=r"Analysis was cancelled") as info,
    session,
  ):
    session.run(audio)
  return info.value


def test_raising_on_file_complete_is_the_cause(tmp_path: Path) -> None:
  def boom(result: object) -> None:
    raise ValueError("callback failure")

  audio = write_marked_audio(tmp_path / "a.wav", 2)
  failure = _run_and_get_failure(
    fake_predict_session(tmp_path, n_workers=1, on_file_complete=boom), audio
  )
  assert isinstance(failure.__cause__, ValueError)
  assert "ValueError: callback failure" in str(failure)


def test_raising_progress_callback_is_the_cause(tmp_path: Path) -> None:
  def boom(stats: object) -> None:
    raise ValueError("progress failure")

  audio = write_marked_audio(tmp_path / "a.wav", 2)
  failure = _run_and_get_failure(
    fake_predict_session(
      tmp_path, n_workers=1, show_stats="progress", progress_callback=boom
    ),
    audio,
  )
  assert isinstance(failure.__cause__, ValueError)
  assert "ValueError: progress failure" in str(failure)


def test_input_too_long_is_the_cause(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  # Two supported segments, so a three-segment file is too long.
  monkeypatch.setattr(input_analyzer, "max_value_for_uint_dtype", lambda dtype: 2)
  audio = write_marked_audio(tmp_path / "a.wav", 3)
  failure = _run_and_get_failure(fake_predict_session(tmp_path, n_workers=1), audio)
  assert isinstance(failure.__cause__, ValueError)
  assert "exceeds the maximum supported amount of segments 2" in str(failure)


def test_analyzer_exception_is_the_cause(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  def boom(self: InputAnalyzer) -> None:
    raise OSError("analyzer failure")

  monkeypatch.setattr(InputAnalyzer, "run_main_loop", boom)
  audio = write_marked_audio(tmp_path / "a.wav", 2)
  failure = _run_and_get_failure(fake_predict_session(tmp_path, n_workers=1), audio)
  assert isinstance(failure.__cause__, OSError)
  assert "OSError: analyzer failure" in str(failure)


def test_consumer_exception_is_the_cause(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  def boom(self: Consumer) -> None:
    raise OSError("consumer failure")

  monkeypatch.setattr(Consumer, "_run_main_loop", boom)
  audio = write_marked_audio(tmp_path / "a.wav", 2)
  failure = _run_and_get_failure(fake_predict_session(tmp_path, n_workers=1), audio)
  assert isinstance(failure.__cause__, OSError)
  assert "OSError: consumer failure" in str(failure)


def test_failed_completion_finalizing_is_the_cause(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  def boom(self: Consumer) -> None:
    raise OSError("finalizing failure")

  monkeypatch.setattr(Consumer, "_finalize_markers", boom)
  audio = write_marked_audio(tmp_path / "a.wav", 2)
  session = fake_predict_session(
    tmp_path, n_workers=1, on_file_complete=lambda result: None
  )
  failure = _run_and_get_failure(session, audio)
  assert isinstance(failure.__cause__, OSError)
  assert "OSError: finalizing failure" in str(failure)


def test_an_earlier_runs_error_is_not_the_cause(tmp_path: Path) -> None:
  fail = False

  def boom_on_second_run(result: object) -> None:
    if fail:
      raise ValueError("second run")

  audio = write_marked_audio(tmp_path / "a.wav", 2)
  session = fake_predict_session(
    tmp_path, n_workers=1, on_file_complete=boom_on_second_run
  )
  with session:
    session.run(audio)
    # Left over from the finished run, e.g. a callback that failed after it.
    assert session._process_manager is not None
    session._process_manager.record_run_error(ValueError("first run"))
    fail = True
    with pytest.raises(RuntimeError) as info:
      session.run(audio)
  assert str(info.value.__cause__) == "second run"
