"""An acoustic TFLite backend runs the rows of a batch one by one.

The interpreter has one thread, on which a batch gains nothing, and with a
worker on every core it costs the caches they share. One row at a time also
keeps the input shape, so the short last batch of a run needs no reallocation.
A backend that does not ask for it, as the geo models' do not, runs a batch as
it is. Pinned on an interpreter stand-in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from birdnet.core import backends
from birdnet.core.backends import PinnedTFBackend
from birdnet.globals import MODEL_PRECISIONS

pytestmark = pytest.mark.no_tf

_INPUT_IDX = 0
_PREDICTION_IDX = 7


class _Interpreter:
  def __init__(self) -> None:
    self.calls: list[str] = []
    self.shapes: list[tuple[int, ...]] = []
    self._input = np.zeros((1, 3), dtype=np.float32)

  def get_input_details(self) -> list[dict]:
    return [
      {
        "index": _INPUT_IDX,
        "shape": np.array([1, 3]),
        "shape_signature": np.array([-1, 3]),
      }
    ]

  def get_output_details(self) -> list[dict]:
    return [{"index": _PREDICTION_IDX, "shape": np.array([1, 2])}]

  def resize_tensor_input(
    self, index: int, shape: tuple[int, ...], strict: bool = False
  ) -> None:
    self.calls.append("resize")

  def allocate_tensors(self) -> None:
    self.calls.append("allocate")

  def set_tensor(self, index: int, value: np.ndarray) -> None:
    self.shapes.append(value.shape)
    self._input = value.copy()

  def invoke(self) -> None:
    self.calls.append("invoke")

  def get_tensor(self, index: int) -> np.ndarray:
    # One row per input row, so the order of the rows shows in the result.
    return self._input[:, :2] * 2


class _Backend(PinnedTFBackend):
  @classmethod
  def in_idx(cls) -> int:
    return _INPUT_IDX

  @classmethod
  def prediction_out_idx(cls) -> int:
    return _PREDICTION_IDX

  @classmethod
  def encoding_out_idx(cls) -> int | None:
    return None

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return "fp32"

  @classmethod
  def supports_encoding(cls) -> bool:
    return False

  @classmethod
  def runs_rows_one_by_one(cls) -> bool:
    return True


class _BatchBackend(_Backend):
  @classmethod
  def runs_rows_one_by_one(cls) -> bool:
    return False


def _load(
  monkeypatch: pytest.MonkeyPatch, backend_type: type[_Backend]
) -> tuple[_Backend, _Interpreter]:
  interp = _Interpreter()
  monkeypatch.setattr(backends, "load_tf_model", lambda *_args, **_kwargs: interp)
  backend = backend_type(Path("model.file"), "CPU", False, inference_library="litert")
  backend.load()
  interp.calls.clear()
  return backend, interp


@pytest.fixture
def loaded(monkeypatch: pytest.MonkeyPatch) -> tuple[_Backend, _Interpreter]:
  return _load(monkeypatch, _Backend)


_BATCH = np.arange(1, 10, dtype=np.float32).reshape(3, 3)


def test_every_row_is_its_own_invocation(loaded: tuple[_Backend, _Interpreter]) -> None:
  backend, interp = loaded

  scores = backend.predict(_BATCH)

  assert interp.shapes == [(1, 3)] * 3
  np.testing.assert_array_equal(scores, _BATCH[:, :2] * 2)


def test_a_shorter_batch_needs_no_reallocation(
  loaded: tuple[_Backend, _Interpreter],
) -> None:
  backend, interp = loaded
  backend.predict(_BATCH)
  interp.calls.clear()

  scores = backend.predict(_BATCH[:2])

  assert interp.calls == ["invoke", "invoke"]
  assert scores.shape == (2, 2)


def test_a_backend_that_does_not_ask_for_it_runs_the_batch_as_it_is(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  backend, interp = _load(monkeypatch, _BatchBackend)

  scores = backend.predict(_BATCH)

  assert interp.shapes == [(3, 3)]
  np.testing.assert_array_equal(scores, _BATCH[:, :2] * 2)
