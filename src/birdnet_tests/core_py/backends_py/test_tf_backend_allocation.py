"""When a TFLite backend allocates its tensors.

At load, so that a file the interpreter cannot run fails there, which is what
the validity check of ``load_custom`` relies on. The exception is a file that
leaves its input length open, as the acoustic 3.0 export does: it stores one
sample in place of the length, LiteRT 2.2.0 and later refuse to allocate that,
and so it is allocated with its first batch, after the resize (issue #110).
Pinned on interpreter stand-ins, whatever LiteRT is installed.
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
_N_SPECIES = 5


class _Interpreter:
  """Refuses to allocate an input length of one sample, as LiteRT 2.2.0 does."""

  def __init__(self, shape_signature: list[int]) -> None:
    self.calls: list[str] = []
    self._signature = shape_signature
    self._shape = tuple(1 if dim < 0 else dim for dim in shape_signature)

  def get_input_details(self) -> list[dict]:
    return [
      {
        "index": _INPUT_IDX,
        "shape": np.array(self._shape),
        "shape_signature": np.array(self._signature),
      }
    ]

  def get_output_details(self) -> list[dict]:
    return [{"index": _PREDICTION_IDX, "shape": np.array([1, _N_SPECIES])}]

  def resize_tensor_input(
    self, index: int, shape: tuple[int, ...], strict: bool = False
  ) -> None:
    assert index == _INPUT_IDX
    self._shape = tuple(shape)
    self.calls.append("resize")

  def allocate_tensors(self) -> None:
    if self._shape[1] == 1:
      raise RuntimeError("MirrorPad paddings must be no greater than the input")
    self.calls.append("allocate")

  def set_tensor(self, index: int, value: np.ndarray) -> None:
    assert self.calls[-1] == "allocate"

  def invoke(self) -> None:
    pass

  def get_tensor(self, index: int) -> np.ndarray:
    return np.zeros((self._shape[0], _N_SPECIES), dtype=np.float32)


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


def _backend_on(monkeypatch: pytest.MonkeyPatch, interp: _Interpreter) -> _Backend:
  def fake_load_tf_model(
    model_path: Path, library: str, allocate_tensors: bool = False, **_: object
  ) -> _Interpreter:
    if allocate_tensors:
      interp.allocate_tensors()
    return interp

  monkeypatch.setattr(backends, "load_tf_model", fake_load_tf_model)
  return _Backend(Path("model.file"), "CPU", False, inference_library="litert")


def test_an_open_input_length_is_allocated_with_the_first_batch(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  interp = _Interpreter([1, -1])
  backend = _backend_on(monkeypatch, interp)

  backend.load()
  assert interp.calls == []
  # Read from the file's description, which needs no allocation.
  assert backend.n_species == _N_SPECIES

  scores = backend.predict(np.zeros((1, 96_000), dtype=np.float32))

  assert interp.calls == ["resize", "allocate"]
  assert scores.shape == (1, _N_SPECIES)


def test_a_fixed_input_length_is_allocated_at_load(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """An open batch size is no open length: the 2.4 exports have one."""
  interp = _Interpreter([-1, 144_000])
  backend = _backend_on(monkeypatch, interp)

  backend.load()

  assert interp.calls == ["allocate"]


def test_a_file_that_cannot_be_allocated_fails_at_load(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Not at its first prediction: the validity check only loads."""
  backend = _backend_on(monkeypatch, _Interpreter([-1, 1]))

  with pytest.raises(RuntimeError, match="MirrorPad"):
    backend.load()
