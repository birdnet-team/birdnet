"""When a TFLite backend runs on the interpreter's default delegate, XNNPACK.

The delegate keeps no tensor inside the part of the graph it takes over, so a
backend may only use it while every tensor it reads is a graph output. The 2.4
models read their embeddings from inside the graph, the 3.0 models from an
output. On Apple silicon no backend uses it. Pinned on interpreter stand-ins;
the real interpreters are covered by the model tests.
"""

from __future__ import annotations

import platform
from pathlib import Path

import numpy as np
import pytest

from birdnet.core import backends
from birdnet.core.backends import PinnedTFBackend
from birdnet.globals import MODEL_PRECISIONS

pytestmark = pytest.mark.no_tf

_INPUT_IDX = 0
_INNER_IDX = 5
_OUTPUT_IDX = 7
_SECOND_OUTPUT_IDX = 8
_WIDTH = 4


class _Interpreter:
  """Refuses to hand out a tensor inside the graph while it is delegated."""

  def __init__(self, default_delegates: bool, outputs: tuple[int, ...]) -> None:
    self.default_delegates = default_delegates
    self._outputs = outputs
    self._rows = 1

  def get_input_details(self) -> list[dict]:
    return [{"index": _INPUT_IDX, "shape": np.array([1, 3])}]

  def get_output_details(self) -> list[dict]:
    return [{"index": i, "shape": np.array([1, _WIDTH])} for i in self._outputs]

  def resize_tensor_input(
    self, index: int, shape: tuple[int, ...], strict: bool = False
  ) -> None:
    self._rows = shape[0]

  def allocate_tensors(self) -> None:
    pass

  def set_tensor(self, index: int, value: np.ndarray) -> None:
    pass

  def invoke(self) -> None:
    pass

  def get_tensor(self, index: int) -> np.ndarray:
    if self.default_delegates and index not in self._outputs:
      raise ValueError("Tensor data is null. Run allocate_tensors() first")
    return np.full((self._rows, _WIDTH), index, dtype=np.float32)


def _backend_type(
  prediction_idx: int, embeddings_idx: int, uses_default_delegates: bool = True
) -> type[PinnedTFBackend]:
  class _Backend(PinnedTFBackend):
    @classmethod
    def in_idx(cls) -> int:
      return _INPUT_IDX

    @classmethod
    def prediction_out_idx(cls) -> int:
      return prediction_idx

    @classmethod
    def encoding_out_idx(cls) -> int | None:
      return embeddings_idx

    @classmethod
    def uses_default_delegates(cls) -> bool:
      return uses_default_delegates

    @classmethod
    def precision(cls) -> MODEL_PRECISIONS:
      return "fp32"

    @classmethod
    def supports_encoding(cls) -> bool:
      return True

  return _Backend


@pytest.fixture(autouse=True)
def _on_a_machine_where_the_delegate_pays_off(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(platform, "system", lambda: "Linux")
  monkeypatch.setattr(platform, "machine", lambda: "x86_64")


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[_Interpreter]:
  """Every interpreter the backend opens, in order."""
  interpreters: list[_Interpreter] = []

  def fake_load_tf_model(
    model_path: Path,
    library: str,
    allocate_tensors: bool = False,
    preserve_all_tensors: bool = False,
    default_delegates: bool = False,
  ) -> _Interpreter:
    interp = _Interpreter(default_delegates, (_OUTPUT_IDX, _SECOND_OUTPUT_IDX))
    interpreters.append(interp)
    return interp

  monkeypatch.setattr(backends, "load_tf_model", fake_load_tf_model)
  return interpreters


def _load(backend_type: type[PinnedTFBackend]) -> PinnedTFBackend:
  backend = backend_type(Path("model.file"), "CPU", False, inference_library="litert")
  backend.load()
  return backend


_BATCH = np.zeros((2, 3), dtype=np.float32)


def test_a_prediction_read_from_a_graph_output_runs_delegated(
  opened: list[_Interpreter],
) -> None:
  backend = _load(_backend_type(_OUTPUT_IDX, _INNER_IDX))

  scores = backend.predict(_BATCH)

  assert [interp.default_delegates for interp in opened] == [True]
  assert scores.shape == (2, _WIDTH)


def test_a_prediction_read_from_inside_the_graph_runs_without_the_delegate(
  opened: list[_Interpreter],
) -> None:
  backend = _load(_backend_type(_INNER_IDX, _INNER_IDX))

  scores = backend.predict(_BATCH)

  assert [interp.default_delegates for interp in opened] == [True, False]
  assert float(scores[0, 0]) == _INNER_IDX


def test_embeddings_read_from_inside_the_graph_drop_the_delegate_once(
  opened: list[_Interpreter],
) -> None:
  backend = _load(_backend_type(_OUTPUT_IDX, _INNER_IDX))

  first = backend.encode(_BATCH)
  backend.encode(_BATCH)

  assert [interp.default_delegates for interp in opened] == [True, False]
  assert float(first[0, 0]) == _INNER_IDX


def test_embeddings_read_from_a_graph_output_keep_the_delegate(
  opened: list[_Interpreter],
) -> None:
  backend = _load(_backend_type(_OUTPUT_IDX, _SECOND_OUTPUT_IDX))

  embeddings = backend.encode(_BATCH)

  assert [interp.default_delegates for interp in opened] == [True]
  assert float(embeddings[0, 0]) == _SECOND_OUTPUT_IDX


def test_a_backend_that_does_not_ask_for_the_delegate_never_gets_it(
  opened: list[_Interpreter],
) -> None:
  backend = _load(_backend_type(_OUTPUT_IDX, _INNER_IDX, uses_default_delegates=False))

  backend.predict(_BATCH)
  backend.encode(_BATCH)

  assert [interp.default_delegates for interp in opened] == [False]


@pytest.mark.parametrize(
  ("system", "machine", "pays_off"),
  [
    ("Linux", "x86_64", True),
    ("Linux", "aarch64", True),
    ("Windows", "AMD64", True),
    ("Darwin", "x86_64", True),
    ("Darwin", "arm64", False),
  ],
)
def test_on_apple_silicon_a_backend_that_asks_for_the_delegate_runs_without_it(
  monkeypatch: pytest.MonkeyPatch,
  opened: list[_Interpreter],
  system: str,
  machine: str,
  pays_off: bool,
) -> None:
  monkeypatch.setattr(platform, "system", lambda: system)
  monkeypatch.setattr(platform, "machine", lambda: machine)
  backend = _load(_backend_type(_OUTPUT_IDX, _INNER_IDX))

  backend.predict(_BATCH)

  assert backends.default_delegates_pay_off() is pays_off
  assert [interp.default_delegates for interp in opened] == [pays_off]


def test_the_batch_is_resized_again_after_the_delegate_is_dropped(
  opened: list[_Interpreter],
) -> None:
  backend = _load(_backend_type(_OUTPUT_IDX, _INNER_IDX))
  backend.predict(_BATCH)

  embeddings = backend.encode(_BATCH)

  assert embeddings.shape == (2, _WIDTH)
