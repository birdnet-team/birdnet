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
  cast,
  final,
  overload,
  runtime_checkable,
)

import numpy as np

from birdnet.globals import (
  LIBRARY_LITERT,
  LIBRARY_TF,
  LIBRARY_TYPES,
  MODEL_BACKEND_PB,
  MODEL_BACKEND_TF,
  MODEL_PRECISIONS,
)
from birdnet.logging_utils import get_logger_for_package

if TYPE_CHECKING:
  from ai_edge_litert.interpreter import Interpreter as LiteRTInterpreter
  from tensorflow.lite.python.interpreter import Interpreter as TFInterpreter


class Backend(ABC):
  def __init__(self, model_path: Path, device_name: str) -> None:
    self._model_path = model_path
    self._device_name = device_name

  @abstractmethod
  def load(self) -> None: ...

  @abstractmethod
  def unload(self) -> None: ...

  @abstractmethod
  def predict(self, batch: np.ndarray) -> np.ndarray: ...

  @abstractmethod
  def embed(self, batch: np.ndarray) -> np.ndarray: ...

  @classmethod
  @abstractmethod
  def supports_cow(cls) -> bool: ...

  @classmethod
  @abstractmethod
  def emb_supported(cls) -> bool: ...

  @property
  @abstractmethod
  def n_species(self) -> int: ...

  @classmethod
  @abstractmethod
  def precision(cls) -> MODEL_PRECISIONS: ...

  @classmethod
  @abstractmethod
  def name(cls) -> str: ...


@runtime_checkable
class VersionedBackendProtocol(Protocol):
  def __init__(
    self,
    model_path: Path,
    device_name: str,
    **kwargs: Any,
  ) -> None: ...

  def load(self) -> None: ...

  def unload(self) -> None: ...

  def predict(self, batch: np.ndarray) -> np.ndarray: ...

  def embed(self, batch: np.ndarray) -> np.ndarray: ...

  @classmethod
  def emb_supported(cls) -> bool: ...

  @classmethod
  def supports_cow(cls) -> bool: ...

  @property
  def n_species(self) -> int: ...

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS: ...

  @classmethod
  def name(cls) -> str: ...


@runtime_checkable
class VersionedAcousticBackendProtocol(VersionedBackendProtocol, Protocol):
  pass


@runtime_checkable
class VersionedGeoBackendProtocol(VersionedBackendProtocol, Protocol):
  pass


TF_BACKEND_LIB_ARG = "inference_library"


class TFBackend(Backend, ABC):
  def __init__(
    self,
    model_path: Path,
    device_name: str,
    **kwargs: dict,
  ) -> None:
    assert device_name == "CPU"
    super().__init__(model_path, device_name)
    self._interp: LiteRTInterpreter | TFInterpreter | None = None
    self._cached_shape: tuple[int, ...] | None = None
    assert TF_BACKEND_LIB_ARG in kwargs
    self._inference_library: LIBRARY_TYPES = cast(
      LIBRARY_TYPES, kwargs[TF_BACKEND_LIB_ARG]
    )

  @classmethod
  def name(cls) -> str:
    return MODEL_BACKEND_TF

  @final
  @classmethod
  def supports_cow(cls) -> bool:
    return True

  @classmethod
  @abstractmethod
  def in_idx(cls) -> int: ...

  @classmethod
  @abstractmethod
  def scores_out_idx(cls) -> int: ...

  @classmethod
  @abstractmethod
  def emb_out_idx(cls) -> int | None: ...

  def load(self) -> None:
    assert self._interp is None
    self._interp = load_tf_model(
      self._model_path, self._inference_library, allocate_tensors=True
    )

  def unload(self) -> None:
    self._interp = None
    self._cached_shape = None

  @property
  def n_species(self) -> int:
    assert self._interp is not None
    output_details = self._interp.get_output_details()
    n_species = output_details[0]["shape"][1]
    return n_species

  def _set_tensor(self, batch: np.ndarray) -> None:
    assert self._interp is not None
    assert batch.flags["C_CONTIGUOUS"]
    assert batch.ndim == 2
    assert self._interp is not None

    shape = batch.shape
    if self._cached_shape != shape:
      self._interp.resize_tensor_input(self.in_idx(), shape, strict=True)
      self._interp.allocate_tensors()
      self._cached_shape = shape
    # self._in_view[:n, :] = batch
    self._interp.set_tensor(self.in_idx(), batch)

  def _infer(self, batch: np.ndarray, out_idx: int) -> np.ndarray:
    # TODO: implement load on different CPUs
    assert self._interp is not None

    self._set_tensor(batch)
    self._interp.invoke()
    res: np.ndarray = self._interp.get_tensor(out_idx)
    return res

  @final
  def predict(self, batch: np.ndarray) -> np.ndarray:
    res = self._infer(batch, self.scores_out_idx())
    assert res.dtype == np.float32
    return res

  @final
  def embed(self, batch: np.ndarray) -> np.ndarray:
    out_idx = self.emb_out_idx()
    assert out_idx is not None
    res = self._infer(batch, out_idx)
    assert res.dtype == np.float32
    return res


class PBBackend(Backend, ABC):
  def __init__(
    self,
    model_path: Path,
    device_name: str,
    **kwargs: dict,
  ) -> None:
    super().__init__(model_path, device_name)
    self._model: Any | None = None
    self._logical_device: Any | None = None
    self._predict_fn: Callable | None = None
    self._emb_fn: Callable | None = None
    self._cached_device_name: str | None = None

  @classmethod
  def name(cls) -> str:
    return MODEL_BACKEND_PB

  @final
  @classmethod
  def supports_cow(cls) -> bool:
    return False

  @classmethod
  @abstractmethod
  def input_key(cls) -> str: ...

  @classmethod
  @abstractmethod
  def scores_signature_name(cls) -> str: ...

  @classmethod
  @abstractmethod
  def scores_prediction_key(cls) -> str: ...

  @classmethod
  @abstractmethod
  def emb_signature_name(cls) -> str | None: ...

  @classmethod
  @abstractmethod
  def emb_prediction_key(cls) -> str | None: ...

  @final
  def load(self) -> None:
    self._model = load_pb_model(self._model_path)
    self._predict_fn = self._model.signatures[self.scores_signature_name()]  # type: ignore
    if self.emb_supported():
      emb_sig_name = self.emb_signature_name()
      assert emb_sig_name is not None
      self._emb_fn = self._model.signatures[emb_sig_name]  # type: ignore

    self._set_logical_device(self._device_name)

  def unload(self) -> None:
    self._model = None
    self._logical_device = None
    self._predict_fn = None
    self._emb_fn = None
    self._cached_device_name = None

  @property
  def n_species(self) -> int:
    assert self._predict_fn is not None
    n_species_in_model: int = (
      self._predict_fn.output_shapes[self.scores_prediction_key()].dims[1].value  # type: ignore
    )
    return n_species_in_model

  def _set_logical_device(self, device_name: str) -> None:
    assert "GPU" in device_name or "CPU" in device_name
    import tensorflow as tf

    if "GPU" in device_name:
      physical_devices = tf.config.list_physical_devices("GPU")
      if len(physical_devices) == 0:
        raise ValueError(
          "No GPU found! "
          "Please check your TensorFlow installation and ensure that a GPU is "
          "available. Also ensure that birdnet is installed with GPU support "
          "(pip install birdnet[and-cuda])."
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
      predictions = self._predict_fn(**{self.input_key(): batch})
    scores: Tensor = predictions[self.scores_prediction_key()]
    assert scores.dtype == float32
    scores_np = scores.numpy()  # type: ignore
    assert scores_np.dtype == np.float32
    return scores_np

  @final
  def embed(self, batch: np.ndarray) -> np.ndarray:
    assert self.emb_supported()
    emb_pred_key = self.emb_prediction_key()
    assert emb_pred_key is not None
    assert self._emb_fn is not None
    assert self._logical_device is not None
    from tensorflow import Tensor, device, float32

    with device(self._logical_device.name):  # type: ignore
      predictions = self._emb_fn(**{self.input_key(): batch})
    emb: Tensor = predictions[emb_pred_key]
    assert emb.dtype == float32
    emb_np = emb.numpy()  # type: ignore
    assert emb_np.dtype == np.float32
    return emb_np


class BackendLoader:
  def __init__(
    self,
    model_path: Path,
    backend_type: type[VersionedBackendProtocol],
    backend_kwargs: dict[str, Any],
  ) -> None:
    self._model_path = model_path
    self._backend_type = backend_type
    self._backend_kwargs = backend_kwargs
    self._backend: VersionedBackendProtocol | None = None

  def unload_backend(self) -> None:
    assert self._backend is not None
    self._backend.unload()
    self._backend = None

  def _load_backend(self, device_name: str) -> VersionedBackendProtocol:
    assert self._backend is None
    backend = self._backend_type(
      model_path=self._model_path,
      device_name=device_name,
      **self._backend_kwargs,
    )
    backend.load()
    self._backend = backend
    return backend

  def load_backend_in_main_process_if_possible(self, devices: list[str]) -> None:
    unique_devices = set(devices)
    same_device_for_all_workers = len(unique_devices) == 1
    if (
      same_device_for_all_workers
      and multiprocessing.get_start_method() == "fork"
      and self._backend_type.supports_cow()
    ):
      device_name = unique_devices.pop()
      self._load_backend(device_name)

  def load_backend(self, device_name: str) -> VersionedBackendProtocol:
    if self._backend is None:
      return self._load_backend(device_name)
    assert self._backend is not None
    return self._backend

  @property
  def backend(self) -> VersionedBackendProtocol:
    assert self._backend is not None
    return self._backend

  @classmethod
  def _get_n_species(
    cls,
    model_path: Path,
    backend_type: type[VersionedBackendProtocol],
    kwargs: dict[str, Any],
  ) -> int | None:
    try:
      loader = cls(model_path, backend_type, kwargs)
      loader.load_backend("CPU")
      n_species_in_model = loader.backend.n_species
      return n_species_in_model
    except Exception as ex:
      get_logger_for_package(__name__).error(f"Error loading model: {ex}")
      return None

  @classmethod
  def check_model_can_be_loaded(
    cls,
    model_path: Path,
    backend_type: type[VersionedBackendProtocol],
    kwargs: dict[str, Any],
  ) -> int:
    """
    Check if the model can be loaded in a subprocess to avoid
    loading tensorflow in the main process.

    Returns the number of species in the model if successful.
    """
    try:
      n_species_in_model = None
      with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(cls._get_n_species, model_path, backend_type, kwargs)
        n_species_in_model = future.result(timeout=None)
      if n_species_in_model is None:
        raise ValueError("Failed to load model.")
      return n_species_in_model
    except Exception as e:
      get_logger_for_package(__name__).error(f"Failed to load model in subprocess: {e}")
      raise ValueError("Failed to load model.") from e


def load_pb_model(model_path: Path) -> Any:
  import absl.logging

  absl_verbosity_before = absl.logging.get_verbosity()
  absl.logging.set_verbosity(absl.logging.ERROR)
  tf_verbosity_before = logging.getLogger("tensorflow").level
  logging.getLogger("tensorflow").setLevel(logging.ERROR)
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  import tensorflow as tf

  # Note: memory growth needs to be set before loading the model and
  # maybe only once in the main process
  # physical_gpu_device = gpus_with_name[0]
  # if tf.config.experimental.get_memory_growth(physical_gpu_device) is False:
  #   tf.config.experimental.set_memory_growth(physical_gpu_device, True)

  start = time.perf_counter()
  model = tf.saved_model.load(str(model_path.absolute()))
  end = time.perf_counter()
  logger = get_logger_for_package(__name__)
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
      experimental_op_resolver_type=tflite.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
      # tensor#187 is a dynamic-sized tensor # type: ignore
    )
  except ValueError as e:
    raise ValueError(
      f"Failed to load model '{model_path.absolute()}' using 'tensorflow'. "
      "Ensure it is a valid TFLite model."
    ) from e

  end = time.perf_counter()
  logger = get_logger_for_package(__name__)
  logger.debug(
    f"Model loaded from {model_path.absolute()} using 'tensorflow' "
    f"in {end - start:.2f} seconds."
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
      experimental_op_resolver_type=tflite.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
      # tensor#187 is a dynamic-sized tensor # type: ignore
    )
  except ValueError as e:
    raise ValueError(
      f"Failed to load model '{model_path.absolute()}' using 'ai_edge_litert'. "
      "Ensure it is a valid TFLite model."
    ) from e

  end = time.perf_counter()
  logger = get_logger_for_package(__name__)
  logger.debug(
    f"Model loaded from {model_path.absolute()} using 'ai_edge_litert' "
    f"in {end - start:.2f} seconds."
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
