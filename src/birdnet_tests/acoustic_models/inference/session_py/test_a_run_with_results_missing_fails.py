"""A run whose results come back short fails instead of returning them.

The rows the producers wrote into the ring and the rows that came back as
results must agree. A block lost between the workers and the consumer (here
dropped by a reader that discards one, test-only) fails the run with both
numbers, without a hang; a healthy run with several producers and batches of
several rows gets every row back, compared once the producers' reports are in.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from birdnet.acoustic.inference import process_manager
from birdnet.acoustic.inference.core.sync import ThreadedQueueReader
from birdnet.acoustic.inference.process_manager import (
  PipelineThreadError,
  ProcessManager,
)
from birdnet_tests.fake_acoustic_backend import (
  fake_encode_session,
  fake_predict_session,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf

_DEADLINE_S = 60.0


def _within_deadline(step: Callable[[], object]) -> BaseException | None:
  done = threading.Event()
  outcome: dict[str, BaseException] = {}

  def call() -> None:
    try:
      step()
    except BaseException as e:  # noqa: BLE001
      outcome["error"] = e
    finally:
      done.set()

  threading.Thread(target=call, daemon=True).start()
  assert done.wait(_DEADLINE_S), f"the run did not end within {_DEADLINE_S:.0f} s"
  return outcome.get("error")


class _DropsOneResultBlock(ThreadedQueueReader):
  """The results reader, losing the first block it would hand on."""

  def __init__(self, q: object, name: str) -> None:
    super().__init__(q, name)
    self._drop = name.endswith("-ResultsReader")

  def get(self, timeout: float) -> object:
    item = super().get(timeout)
    if self._drop and item is not None:
      self._drop = False
      raise queue.Empty
    return item


def test_a_run_with_a_result_block_lost_fails_with_both_numbers(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(process_manager, "ThreadedQueueReader", _DropsOneResultBlock)
  files = [str(write_marked_audio(tmp_path / f"in_{i}.wav", 3)) for i in range(2)]

  def run() -> None:
    with fake_predict_session(tmp_path, n_workers=2, batch_size=1) as session:
      session.run(files)

  error = _within_deadline(run)

  assert isinstance(error, RuntimeError), f"the run did not fail: {error!r}"
  assert isinstance(error.__cause__, PipelineThreadError), error.__cause__
  text = str(error)
  assert text.startswith(
    "Results are missing: 6 segments went to the workers, but results for only "
    "5 came back."
  ), text
  assert text.count("Please check the logs") == 1, text


_SEGMENTS = (5, 7, 3)


@pytest.mark.parametrize(
  ("mode", "inputs"), [("predict", "files"), ("encode", "arrays")]
)
def test_a_healthy_run_gets_back_every_row_the_producers_wrote(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, inputs: str
) -> None:
  seen: list[tuple[int, int, bool]] = []
  check = ProcessManager.check_all_rows_arrived

  def recording_check(self: ProcessManager, rows_received: int) -> None:
    producers = self._res.producer_resources  # noqa: SLF001
    # Reading a producer's report is what orders its count; a finish flag
    # seen set does not.
    reports_read = producers._unprocessed_inputs is not None  # noqa: SLF001
    rows_written = sum(rows.value for rows in producers.rows_written)
    seen.append((rows_written, rows_received, reports_read))
    check(self, rows_received)

  monkeypatch.setattr(ProcessManager, "check_all_rows_arrived", recording_check)
  paths = [
    write_marked_audio(tmp_path / f"in_{i}_{n}.wav", n) for i, n in enumerate(_SEGMENTS)
  ]
  shape = {"n_workers": 2, "n_producers": 2, "batch_size": 3}
  if mode == "predict":
    # Nothing below a threshold, so a masked row is a missing one.
    session = fake_predict_session(
      tmp_path, top_k=None, default_confidence_threshold=-float("inf"), **shape
    )
  else:
    session = fake_encode_session(tmp_path, **shape)
  with session:
    if inputs == "files":
      result = session.run([str(p) for p in paths])
    else:
      result = session.run_arrays([sf.read(p, dtype="float32") for p in paths])

  assert seen == [(sum(_SEGMENTS), sum(_SEGMENTS), True)]
  masked = np.asarray(
    result.species_masked if mode == "predict" else result.embeddings_masked
  )
  for i, n in enumerate(_SEGMENTS):
    assert not masked[i, :n].any(), f"input {i} has rows missing"
