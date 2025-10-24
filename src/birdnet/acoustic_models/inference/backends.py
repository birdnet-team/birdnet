from __future__ import annotations

import logging
import multiprocessing
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import (
  TYPE_CHECKING,
  Any,
  Literal,
  Protocol,
  final,
  overload,
  runtime_checkable,
)

import numpy as np

from birdnet.globals import (
  LIBRARY_LITERT,
  LIBRARY_TF,
  LIBRARY_TYPES,
)
from birdnet.logging_utils import get_logger

if TYPE_CHECKING:
  from ai_edge_litert.interpreter import Interpreter as LiteRTInterpreter
  from tensorflow.lite.python.interpreter import Interpreter as TFInterpreter


class InferenceBackend(ABC):
  def __init__(self, model_path: Path, device_name: str) -> None:
    self._model_path = model_path
    self._device_name = device_name

  @abstractmethod
  def load(self) -> None: ...

  @abstractmethod
  def predict(self, batch: np.ndarray) -> np.ndarray: ...

  @abstractmethod
  def embed(self, batch: np.ndarray) -> np.ndarray: ...

  @classmethod
  @abstractmethod
  def supports_cow(cls) -> bool: ...


@runtime_checkable
class VersionedInferenceBackendProtocol(Protocol):
  def __init__(
    self,
    model_path: Path,
    device_name: str,
    **kwargs: Any,
  ) -> None: ...

  def load(self) -> None: ...

  def predict(self, batch: np.ndarray) -> np.ndarray: ...

  def embed(self, batch: np.ndarray) -> np.ndarray: ...

  @classmethod
  def supports_cow(cls) -> bool: ...

  @classmethod
  def check_model_can_be_loaded(
    cls,
    model_path: Path,
    **kwargs: Any,
  ) -> int | None: ...


@runtime_checkable
class VersionedAcousticInferenceBackendProtocol(
  VersionedInferenceBackendProtocol, Protocol
):
  pass


@runtime_checkable
class VersionedGeoInferenceBackendProtocol(VersionedInferenceBackendProtocol, Protocol):
  pass


class TFInferenceBackend(InferenceBackend, ABC):
  def __init__(
    self,
    model_path: Path,
    device_name: str,
    in_idx: int,
    scores_out_idx: int,
    emb_out_idx: int,
    inference_library: LIBRARY_TYPES,
  ) -> None:
    assert device_name == "CPU"
    super().__init__(model_path, device_name)
    self._interp: LiteRTInterpreter | TFInterpreter | None = None
    self._inference_library: LIBRARY_TYPES = inference_library
    self._in_idx: int = in_idx
    self._scores_out_idx: int = scores_out_idx
    self._emb_out_idx: int = emb_out_idx
    self._cached_shape: tuple[int, ...] | None = None

  @final
  @classmethod
  def supports_cow(cls) -> bool:
    return True

  def load(self) -> None:
    assert self._interp is None
    self._interp = load_tf_model(
      self._model_path, self._inference_library, allocate_tensors=True
    )

    # self._in_idx = self._interp.get_input_details()[0]["index"]  # type: ignore
    # self._out_idx = self._interp.get_output_details()[0]["index"]  # type: ignore

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

  def _infer(self, batch: np.ndarray, out_idx: int) -> np.ndarray:
    # TODO: implement load on different CPUs
    assert self._interp is not None

    self._set_tensor(batch)
    self._interp.invoke()
    res: np.ndarray = self._interp.get_tensor(out_idx)
    assert res.dtype == np.float32
    return res

  @final
  def predict(self, batch: np.ndarray) -> np.ndarray:
    return self._infer(batch, self._scores_out_idx)

  @final
  def embed(self, batch: np.ndarray) -> np.ndarray:
    return self._infer(batch, self._emb_out_idx)


class PBInferenceBackend(InferenceBackend, ABC):
  def __init__(
    self,
    model_path: Path,
    device_name: str,
    input_key: str,
    scores_signature_name: str,
    scores_prediction_key: str,
    emb_signature_name: str,
    emb_prediction_key: str,
  ) -> None:
    super().__init__(model_path, device_name)
    self._logical_device: Any | None = None
    self._predict_fn: Callable | None = None
    self._emb_fn: Callable | None = None
    self._cached_device_name: str | None = None
    self._scores_signature_name = scores_signature_name
    self._scores_prediction_key = scores_prediction_key
    self._emb_signature_name = emb_signature_name
    self._emb_prediction_key = emb_prediction_key
    self._input_key = input_key

  @final
  @classmethod
  def supports_cow(cls) -> bool:
    return False

  @final
  def load(self) -> None:
    model = load_pb_model(self._model_path)
    self._predict_fn = model.signatures[self._scores_signature_name]
    self._emb_fn = model.signatures[self._emb_signature_name]

    self._set_logical_device(self._device_name)

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

      self._logical_device = [
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
      self._logical_device = all_devices_with_name[0]
    else:
      raise AssertionError()

  @final
  def predict(self, batch: np.ndarray) -> np.ndarray:
    assert self._logical_device is not None
    assert self._predict_fn is not None
    from tensorflow import Tensor, device, float32

    with device(self._logical_device.name):  # type: ignore
      # prediction = self._audio_model.basic(batch)["scores"]
      predictions = self._predict_fn(**{self._input_key: batch})
    scores: Tensor = predictions[self._scores_prediction_key]
    assert scores.dtype == float32
    scores_np = scores.numpy()  # type: ignore
    assert scores_np.dtype == np.float32
    return scores_np

  @final
  def embed(self, batch: np.ndarray) -> np.ndarray:
    assert self._logical_device is not None
    assert self._emb_fn is not None
    from tensorflow import Tensor, device, float32

    with device(self._logical_device.name):  # type: ignore
      predictions = self._emb_fn(**{self._input_key: batch})
    emb: Tensor = predictions[self._emb_prediction_key]
    assert emb.dtype == float32
    emb_np = emb.numpy()  # type: ignore
    assert emb_np.dtype == np.float32
    return emb_np


class InferenceBackendLoader:
  def __init__(
    self,
    model_path: Path,
    backend_type: type[VersionedInferenceBackendProtocol],
    backend_custom_kwargs: dict[str, object],
  ) -> None:
    self._model_path = model_path
    self._backend_type = backend_type
    self._backend_kwargs = backend_custom_kwargs

    self._backend: VersionedInferenceBackendProtocol | None = None

  def unload_backend(self) -> None:
    self._backend = None

  def _load_backend(self, device_name: str) -> VersionedInferenceBackendProtocol:
    assert self._backend is None
    backend = self._backend_type(
      model_path=self._model_path,
      device_name=device_name,
      **self._backend_kwargs,
    )
    backend.load()
    self._backend = backend
    return backend

  def on_before_worker_initialized(self, devices: list[str]) -> None:
    unique_devices = set(devices)
    same_device_for_all_workers = len(unique_devices) == 1
    if (
      same_device_for_all_workers
      and multiprocessing.get_start_method() == "fork"
      and self._backend_type.supports_cow()
    ):
      device_name = unique_devices.pop()
      self._load_backend(device_name)

  def load_backend(self, device_name: str) -> VersionedInferenceBackendProtocol:
    if self._backend is None:
      return self._load_backend(device_name)
    assert self._backend is not None
    return self._backend

  @property
  def backend(self) -> VersionedInferenceBackendProtocol:
    assert self._backend is not None
    return self._backend


def load_pb_model(model_path: Path):
  import absl.logging

  absl_verbosity_before = absl.logging.get_verbosity()
  absl.logging.set_verbosity(absl.logging.ERROR)
  tf_verbosity_before = logging.getLogger("tensorflow").level
  logging.getLogger("tensorflow").setLevel(logging.ERROR)
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  import tensorflow as tf

  # Note: memory growth needs to be set before loading the model and maybe only once in the main process
  # physical_gpu_device = gpus_with_name[0]
  # if tf.config.experimental.get_memory_growth(physical_gpu_device) is False:
  #   tf.config.experimental.set_memory_growth(physical_gpu_device, True)

  start = time.perf_counter()
  model = tf.saved_model.load(str(model_path.absolute()))
  end = time.perf_counter()
  logger = get_logger(__name__)
  logger.debug(
    f"Model loaded from {model_path.absolute()} in {end - start:.2f} seconds."
  )

  absl.logging.set_verbosity(absl_verbosity_before)
  logging.getLogger("tensorflow").setLevel(tf_verbosity_before)
  return model


@overload
def load_tf_model(
  model_path: Path,
  library: Literal["tf"],
  allocate_tensors: bool = False,
) -> TFInterpreter: ...
@overload
def load_tf_model(
  model_path: Path,
  library: Literal["litert"],
  allocate_tensors: bool = False,
) -> LiteRTInterpreter: ...


def load_tf_model(
  model_path: Path,
  library: LIBRARY_TYPES,
  allocate_tensors: bool = False,
):
  if library == LIBRARY_TF:
    return load_lib_tf_model(model_path, allocate_tensors=allocate_tensors)
  elif library == LIBRARY_LITERT:
    return load_lib_litert_model(model_path, allocate_tensors=allocate_tensors)
  else:
    raise AssertionError()


def load_lib_tf_model(
  model_path: Path,
  allocate_tensors: bool = False,
) -> TFInterpreter:
  assert model_path.is_file()
  assert tf_installed()

  import absl.logging as absl_logging

  absl_verbosity_before = absl_logging.get_verbosity()
  absl_logging.set_verbosity(absl_logging.ERROR)
  absl_logging.set_stderrthreshold("error")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  tf_verbosity_before: int | None = None
  tf_verbosity_before = logging.getLogger("tensorflow").level
  logging.getLogger("tensorflow").setLevel(logging.ERROR)
  # NOTE: import in this way is not possible:
  # `import tensorflow.lite.python.interpreter as tflite`
  from tensorflow.lite.python import interpreter as tflite

  # memory_map not working for TF 2.15.1:
  # f = open(self._model_path, "rb")
  # self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
  start = time.perf_counter()
  try:
    interp = tflite.Interpreter(
      str(model_path.absolute()),
      num_threads=1,
      experimental_op_resolver_type=tflite.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,  # tensor#187 is a dynamic-sized tensor # type: ignore
    )
  except ValueError as e:
    raise ValueError(
      f"Failed to load model '{model_path.absolute()}' using 'tensorflow'. Ensure it is a valid TFLite model."
    ) from e

  end = time.perf_counter()
  logger = get_logger(__name__)
  logger.debug(
    f"Model loaded from {model_path.absolute()} using 'tensorflow' in {end - start:.2f} seconds."
  )

  if allocate_tensors:
    interp.allocate_tensors()

  import absl.logging as absl_logging

  assert absl_verbosity_before is not None
  assert tf_verbosity_before is not None
  absl_logging.set_verbosity(absl_verbosity_before)
  logging.getLogger("tensorflow").setLevel(tf_verbosity_before)

  return interp


def load_lib_litert_model(
  model_path: Path,
  allocate_tensors: bool = False,
) -> LiteRTInterpreter:
  assert model_path.is_file()
  assert litert_installed()

  from ai_edge_litert import interpreter as tflite

  start = time.perf_counter()
  try:
    interp = tflite.Interpreter(
      str(model_path.absolute()),
      num_threads=1,
      experimental_op_resolver_type=tflite.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,  # tensor#187 is a dynamic-sized tensor # type: ignore
    )
  except ValueError as e:
    raise ValueError(
      f"Failed to load model '{model_path.absolute()}' using 'ai_edge_litert'. Ensure it is a valid TFLite model."
    ) from e

  end = time.perf_counter()
  logger = get_logger(__name__)
  logger.debug(
    f"Model loaded from {model_path.absolute()} using 'ai_edge_litert' in {end - start:.2f} seconds."
  )

  if allocate_tensors:
    interp.allocate_tensors()

  return interp


def tf_installed() -> bool:
  import importlib.util

  return importlib.util.find_spec("tensorflow") is not None


def litert_installed() -> bool:
  import importlib.util

  return importlib.util.find_spec("ai_edge_litert") is not None


def _get_pb_n_species(
  model_path: Path, signature_name: str, prediction_key: str
) -> int | None:
  try:
    loaded_model = load_pb_model(model_path)
    n_species_in_model: int = (
      loaded_model.signatures[signature_name]  # type: ignore
      .output_shapes[prediction_key]
      .dims[1]
      .value  # type: ignore
    )
    return n_species_in_model
  except Exception:
    return None


def check_pb_model_can_be_loaded(
  model_path: Path, signature_name: str, prediction_key: str
) -> int | None:
  try:
    with ProcessPoolExecutor(max_workers=1) as executor:
      future = executor.submit(
        _get_pb_n_species, model_path, signature_name, prediction_key
      )
      result = future.result(timeout=None)
      return result
  except Exception as e:
    get_logger(__name__).error(f"Failed to load Protobuf model in subprocess: {e}")
    return None


def _get_tf_n_species(
  model_path: Path, library: LIBRARY_TYPES, out_idx: int
) -> int | None:
  try:
    loaded_model = load_tf_model(model_path, library, allocate_tensors=False)
    n_species_in_model = loaded_model.get_output_details()[0]["shape"][1]
    return n_species_in_model
  except Exception:
    return None


def check_tf_model_can_be_loaded(
  model_path: Path, inference_library: LIBRARY_TYPES, out_idx: int
) -> int | None:
  try:
    with ProcessPoolExecutor(max_workers=1) as executor:
      future = executor.submit(
        _get_tf_n_species, model_path, inference_library, out_idx
      )
      result = future.result(timeout=None)
      return result
  except Exception as e:
    get_logger(__name__).error(f"Failed to load TensorFlow model in subprocess: {e}")
    return None
