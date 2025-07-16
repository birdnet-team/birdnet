from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, final

import numpy as np

from birdnet.acoustic_models.base import AcousticInferenceBackend
from birdnet.helper import load_tflite_model
from birdnet.io_lock import IOLockHandler

if TYPE_CHECKING:
  from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter


class AcousticTFBackend(AcousticInferenceBackend):
  def __init__(self, model_path: Path) -> None:
    super().__init__()
    self._model_path = model_path
    self._interp: TFLiteInterpreter | None = None
    self._in_idx: int | None = None
    self._out_idx: int | None = None
    self._cached_shape: tuple[int, ...] | None = None

  @final
  def load(self, io_lock_handler: IOLockHandler) -> None:
    assert self._interp is None

    self._interp = load_tflite_model(
      self._model_path, io_lock_handler, allocate_tensors=True
    )
    self._in_idx = self._interp.get_input_details()[0]["index"]
    self._out_idx = self._interp.get_output_details()[0]["index"]

    # self._in_view = self._interp.tensor(self._in_idx)()[0]

  def _set_tensor(self, batch: np.ndarray):
    assert self._interp is not None
    assert batch.flags["C_CONTIGUOUS"]
    assert batch.ndim == 2
    assert self._interp is not None

    shape = batch.shape
    if self._cached_shape != shape:
      self._interp.resize_tensor_input(self._in_idx, shape, strict=True)
      self._interp.allocate_tensors()
      self._cached_shape = shape
    # self._in_view[:n, :] = batch
    self._interp.set_tensor(self._in_idx, batch)

  @final
  def infer(self, batch: np.ndarray, device_name: str) -> np.ndarray:
    # TODO: implement load on different CPUs
    if "CPU" not in device_name:
      raise ValueError("TensorFlow models can only be loaded on CPU!")

    assert self._interp is not None
    self._set_tensor(batch)
    self._interp.invoke()
    res: np.ndarray = self._interp.get_tensor(self._out_idx)
    assert res.dtype == np.float32
    return res
