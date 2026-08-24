"""The session must collect its promised messages through the bounded reads.

Every deadline and every give-up in the collection machinery is unit-tested
against stand-ins — which means the library could quietly stop *using* any of
it and every one of those tests would stay green. These tests pin the wiring:
a real run, through the real session, must route its collection reads through
``ProcessManager.read_promised``, and a warm call must not carry a fixed
per-call barrier from the machinery's own bookkeeping.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from birdnet.acoustic.inference.process_manager import ProcessManager
from birdnet_tests.fake_acoustic_backend import fake_predict_session

pytestmark = pytest.mark.no_tf

_SAMPLE_RATE = 48_000


def _clip(seconds: float = 3.0) -> tuple[np.ndarray, int]:
  rng = np.random.default_rng(7)
  samples = rng.standard_normal(int(_SAMPLE_RATE * seconds)).astype(np.float32)
  return samples * 0.1, _SAMPLE_RATE


def test_collection_goes_through_the_bounded_read(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A healthy run must call read_promised for the producer reports.

  Reverting the session to a plain blocking ``get`` would leave every unit
  test green and reintroduce the unbounded wait; this is the test that fails
  instead.
  """
  calls: list[tuple[int, str]] = []
  original = ProcessManager.read_promised

  def recording(self: ProcessManager, q, n: int, what: str):  # noqa: ANN001, ANN202
    calls.append((n, what))
    return original(self, q, n, what)

  monkeypatch.setattr(ProcessManager, "read_promised", recording)

  with fake_predict_session(tmp_path, n_workers=1, n_producers=1) as session:
    result = session.run_arrays(_clip())

  assert result is not None
  assert any("unprocessed-input" in what for _, what in calls), (
    f"the session collected its producer reports without the bounded read; "
    f"calls seen: {calls}"
  )


def test_a_warm_call_pays_no_fixed_reader_toll(tmp_path: Path) -> None:
  """The collection machinery must not put a fixed barrier into warm calls.

  This pipeline has reintroduced a fixed per-call cost through teardown
  bookkeeping twice (a poll interval in #75, reader closes in the change this
  file belongs to), each time invisible to the real-model overhead guard
  because its threshold is sized for a 1 s barrier. The stub backend does
  ~2 ms of work, so the *fastest* of several warm calls is almost pure
  pipeline overhead — and a fixed toll in the tens of milliseconds moves it,
  while scheduler noise cannot (noise only ever makes a call slower).
  """
  audio = _clip()
  with fake_predict_session(tmp_path, n_workers=1) as session:
    session.run_arrays(audio)  # warm-up

    durations = []
    for _ in range(8):
      start = time.perf_counter()
      session.run_arrays(audio)
      durations.append(time.perf_counter() - start)

  fastest = min(durations)
  assert fastest < 0.040, (
    f"fastest warm run_arrays took {fastest * 1000:.1f} ms on a stub backend "
    f"whose inference is ~2 ms; a fixed per-call toll is back. "
    f"Samples: {[round(d * 1000, 1) for d in durations]}"
  )
