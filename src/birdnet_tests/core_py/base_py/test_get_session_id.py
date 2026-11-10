from pathlib import Path
from types import SimpleNamespace

import pytest

import birdnet.core.base as base_module
from birdnet.core.base import get_session_id, get_session_tag
from birdnet_tests.fake_acoustic_backend import (
  fake_predict_session,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf


def _stop_the_clock(monkeypatch: pytest.MonkeyPatch) -> None:
  # As between two ticks of a coarse wall clock (Windows before Python 3.13).
  clock = SimpleNamespace(time_ns=lambda: 1_762_165_676_846_175_803)
  monkeypatch.setattr(base_module, "time", clock)


def test_ids_differ_while_the_clock_stands_still(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _stop_the_clock(monkeypatch)

  ids = [get_session_id() for _ in range(100)]

  assert len(set(ids)) == 100
  assert len({get_session_tag(i) for i in ids}) == 100


def test_two_sessions_built_in_the_same_clock_tick_run_side_by_side(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  # Their shared memory is named after the id: equal ids are equal names.
  _stop_the_clock(monkeypatch)
  audio = write_marked_audio(tmp_path / "audio.wav", 2)
  first = fake_predict_session(tmp_path, n_workers=1)
  second = fake_predict_session(tmp_path, n_workers=1)

  with first, second:
    results = [first.run(audio), second.run(audio)]

  assert [r.n_inputs for r in results] == [1, 1]
