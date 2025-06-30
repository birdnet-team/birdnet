from pathlib import Path
from typing import final

import numpy as np

#
from birdnet_v2.acoustic_models.base import AcousticInferenceBackend


class AcousticTFBackend(AcousticInferenceBackend):
  def __init__(self, model_path: Path) -> None:
    super().__init__()
    self._model_path = str(model_path.absolute())
    self._interp: tflite.Interpreter | None = None
    self._in_idx: int | None = None
    self._out_idx: int | None = None
    self._cached_shape: tuple[int, ...] | None = None

  @final
  def lazy_load(self, logical_device_name: str) -> None:
    assert self._interp is None

    from tensorflow.lite.python import interpreter as tflite

    # memory_map not working for TF 2.15.1:
    # f = open(self._model_path, "rb")
    # self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    interp = tflite.Interpreter(self._model_path, num_threads=1)
    interp.allocate_tensors()
    self._interp = interp
    self._in_idx = interp.get_input_details()[0]["index"]
    self._out_idx = interp.get_output_details()[0]["index"]

  def _set_tensor(self, batch: np.ndarray):
    assert self._interp is not None
    assert batch.flags["C_CONTIGUOUS"]
    assert batch.ndim == 2

    shape = batch.shape
    if self._cached_shape != shape:
      self._interp.resize_tensor_input(self._in_idx, shape, strict=True)
      self._interp.allocate_tensors()
      self._cached_shape = shape
    self._interp.set_tensor(self._in_idx, batch)

  @final
  def infer(self, batch: np.ndarray) -> np.ndarray:
    assert self._interp is not None
    self._set_tensor(batch)
    self._interp.invoke()
    res: np.ndarray = self._interp.get_tensor(self._out_idx)
    assert res.dtype == np.float32
    return res
