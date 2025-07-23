from __future__ import annotations

import logging
import multiprocessing
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, final

import numpy as np

from birdnet.base import InferenceBackend
from birdnet.globals import (
  LIBRARY_LITERT,
  LIBRARY_TF,
  LIBRARY_TYPES,
)
from birdnet.helper import (
  load_litert_model,
  load_tf_model,
)
from birdnet.logging_utils import get_logger

if TYPE_CHECKING:
  from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter
  from tensorflow.lite.python.interpreter import Interpreter as TFInterpreter


class InferenceBackendLoader:
  def __init__(
    self,
    backend_type: type[InferenceBackend],
    backend_kwargs: dict,
  ) -> None:
    self._backend_type = backend_type
    self._backend_kwargs = backend_kwargs
    self._backend: InferenceBackend | None = None

  def _load_backend(self) -> InferenceBackend:
    assert self._backend is None
    backend = self._backend_type(**self._backend_kwargs)
    backend.load()
    self._backend = backend
    return backend

  def on_before_worker_initialized(self) -> None:
    if (
      multiprocessing.get_start_method() == "fork" and self._backend_type.supports_cow()
    ):
      self._load_backend()

  def load_backend(self) -> InferenceBackend:
    if self._backend is None:
      return self._load_backend()
    assert self._backend is not None
    return self._backend

  @property
  def backend(self) -> InferenceBackend:
    assert self._backend is not None
    return self._backend


class TFInferenceBackend(InferenceBackend):
  def __init__(self, model_path: Path, inference_library: LIBRARY_TYPES) -> None:
    super().__init__()
    self._model_path = model_path
    self._interp: TFLiteInterpreter | TFInterpreter | None = None
    self._inference_library = inference_library
    self._in_idx: int | None = None
    self._out_idx: int | None = None
    self._cached_shape: tuple[int, ...] | None = None

  @final
  @classmethod
  def supports_cow(cls) -> bool:
    return True

  def load(self) -> None:
    assert self._interp is None
    if self._inference_library == LIBRARY_TF:
      self._interp = load_tf_model(self._model_path, allocate_tensors=True)
    elif self._inference_library == LIBRARY_LITERT:
      self._interp = load_litert_model(self._model_path, allocate_tensors=True)
    else:
      raise AssertionError()

    self._in_idx = self._interp.get_input_details()[0]["index"]  # type: ignore
    self._out_idx = self._interp.get_output_details()[0]["index"]  # type: ignore

  def _set_tensor(self, batch: np.ndarray) -> None:
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


class PBInferenceBackend(InferenceBackend):
  def __init__(
    self, model_path: Path, signature_name: str, prediction_key: str, input_key: str
  ) -> None:
    super().__init__()
    self._model_path = str(model_path.absolute())
    self._cached_logical_device: Any | None = None
    self._infer_fn: Callable | None = None
    self._cached_device_name: str | None = None
    self._signature_name = signature_name
    self._prediction_key = prediction_key
    self._input_key = input_key

  @final
  @classmethod
  def supports_cow(cls) -> bool:
    return False

  @final
  def load(self) -> None:
    import absl.logging

    absl_verbosity_before = absl.logging.get_verbosity()
    absl.logging.set_verbosity(absl.logging.ERROR)
    tf_verbosity_before = logging.getLogger("tensorflow").level
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf

    tf.random.set_seed(0)

    # Note: memory growth needs to be set before loading the model and maybe only once in the main process
    # physical_gpu_device = gpus_with_name[0]
    # if tf.config.experimental.get_memory_growth(physical_gpu_device) is False:
    #   tf.config.experimental.set_memory_growth(physical_gpu_device, True)

    start = time.perf_counter()
    audio_model = tf.saved_model.load(self._model_path)
    end = time.perf_counter()
    logger = get_logger(__name__)
    logger.debug(f"Model loaded from {self._model_path} in {end - start:.2f} seconds.")

    absl.logging.set_verbosity(absl_verbosity_before)
    logging.getLogger("tensorflow").setLevel(tf_verbosity_before)

    self._infer_fn = audio_model.signatures[self._signature_name]  # type: ignore

  def _set_logical_device(self, device_name: str) -> None:
    assert "GPU" in device_name or "CPU" in device_name
    import tensorflow as tf

    if "GPU" in device_name:
      physical_devices = tf.config.list_physical_devices("GPU")
      if len(physical_devices) == 0:
        raise ValueError(
          "No GPU found! Please check your TensorFlow installation and ensure that a GPU is available."
        )

      gpus_with_name = [gpu for gpu in physical_devices if device_name in gpu.name]

      if len(gpus_with_name) == 0:
        raise ValueError(f"No GPU with name '{device_name}' found!")

      self._cached_logical_device = [
        log_dev
        for log_dev in tf.config.list_logical_devices()
        if device_name in log_dev.name
      ][0]

    elif "CPU" in device_name:
      all_devices_with_name: list = [
        log_dev
        for log_dev in tf.config.list_logical_devices()
        if device_name in log_dev.name
      ]
      if len(all_devices_with_name) == 0:
        raise ValueError(f"No CPU with name '{device_name}' found!")
      self._cached_logical_device = all_devices_with_name[0]
    else:
      raise ValueError(f"Unsupported device name: {device_name}")

  @final
  def infer(self, batch: np.ndarray, device_name: str) -> np.ndarray:
    if self._cached_device_name is None or self._cached_device_name != device_name:
      self._set_logical_device(device_name)
      self._cached_device_name = device_name

    assert self._cached_logical_device is not None
    assert self._infer_fn is not None
    from tensorflow import Tensor, device, float32

    with device(self._cached_logical_device.name):  # type: ignore
      # prediction = self._audio_model.basic(batch)["scores"]
      predictions = self._infer_fn(**{self._input_key: batch})  # MNET_INPUT oder inputs
    scores: Tensor = predictions[self._prediction_key]
    assert scores.dtype == float32
    scores_np = scores.numpy()  # type: ignore
    assert scores_np.dtype == np.float32
    return scores_np
