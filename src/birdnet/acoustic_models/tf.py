import logging
import os
import time
from pathlib import Path
from typing import Any, final

import numpy as np

from birdnet.acoustic_models.base import AcousticInferenceBackend
from birdnet.io_lock import IOLockHandler
from birdnet.logging_utils import get_logger


class AcousticTFBackend(AcousticInferenceBackend):
  def __init__(self, model_path: Path) -> None:
    super().__init__()
    self._model_path = str(model_path.absolute())
    self._interp: Any | None = None
    self._in_idx: int | None = None
    self._out_idx: int | None = None
    self._cached_shape: tuple[int, ...] | None = None

  @final
  def load(self, device_name: str, io_lock_handler: IOLockHandler) -> None:
    assert self._interp is None

    if "CPU" not in device_name:
      raise ValueError("TensorFlow models can only be loaded on CPU!")

    import absl.logging as absl_logging

    absl_verbosity_before = absl_logging.get_verbosity()
    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold("error")
    tf_verbosity_before = logging.getLogger("tensorflow").level
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # import tflite_runtime.interpreter as tflite
    from tensorflow.lite.python import interpreter as tflite
    from tensorflow.lite.python.interpreter import OpResolverType

    # memory_map not working for TF 2.15.1:
    # f = open(self._model_path, "rb")
    # self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    with io_lock_handler:
      start = time.perf_counter()
      interp = tflite.Interpreter(
        self._model_path,
        num_threads=1,
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,  # tensor#187 is a dynamic-sized tensor
      )
      end = time.perf_counter()
    logger = get_logger(__name__)
    logger.debug(
      f"Model loaded from {self._model_path} on device CPU in {end - start:.2f} seconds."
    )

    interp.allocate_tensors()

    self._interp = interp
    self._in_idx = interp.get_input_details()[0]["index"]
    self._out_idx = interp.get_output_details()[0]["index"]

    # self._in_view = self._interp.tensor(self._in_idx)()[0]

    absl_logging.set_verbosity(absl_verbosity_before)
    logging.getLogger("tensorflow").setLevel(tf_verbosity_before)

    # tf.random.set_seed(0)

  def _set_tensor(self, batch: np.ndarray):
    from tensorflow.lite.python.interpreter import Interpreter

    assert self._interp is not None
    assert batch.flags["C_CONTIGUOUS"]
    assert batch.ndim == 2
    interpr: Interpreter = self._interp

    shape = batch.shape
    if self._cached_shape != shape:
      interpr.resize_tensor_input(self._in_idx, shape, strict=True)
      interpr.allocate_tensors()
      self._cached_shape = shape
    # self._in_view[:n, :] = batch
    interpr.set_tensor(self._in_idx, batch)

  @final
  def infer(self, batch: np.ndarray) -> np.ndarray:
    from tensorflow.lite.python.interpreter import Interpreter

    assert self._interp is not None
    interpr: Interpreter = self._interp
    self._set_tensor(batch)
    interpr.invoke()
    res: np.ndarray = interpr.get_tensor(self._out_idx)
    assert res.dtype == np.float32
    return res
