from __future__ import annotations

import logging
import multiprocessing
import os
import pickle
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection
from pathlib import Path
from typing import (
  TYPE_CHECKING,
  Any,
  Generic,
  Literal,
  Protocol,
  TypeVar,
  cast,
  final,
  overload,
  runtime_checkable,
)

import numpy as np

from birdnet.core.start_method import resolve_start_method
from birdnet.globals import (
  LIBRARY_LITERT,
  LIBRARY_TFLITE,
  LIBRARY_TYPES,
  MODEL_BACKEND_ONNX,
  MODEL_BACKEND_PB,
  MODEL_BACKEND_PT,
  MODEL_BACKEND_TF,
  MODEL_PRECISIONS,
)
from birdnet.utils.logging_utils import (
  get_logger_for_package,
  suppress_native_stderr,
)

if TYPE_CHECKING:
  import onnxruntime as ort
  from ai_edge_litert.interpreter import (  # type: ignore
    Interpreter as LiteRTInterpreter,
  )
  from tensorflow import Tensor
  from tensorflow.lite.python.interpreter import Interpreter as TFInterpreter
  from torch import Tensor as TorchTensor
  from torch import device as TorchDevice
  from torch.jit import RecursiveScriptModule

  from birdnet.acoustic.models.custom_tflite import CustomTFLiteInfo

BatchT = TypeVar("BatchT", np.ndarray, "Tensor", "TorchTensor")


class Backend(Generic[BatchT], ABC):
  def __init__(self, model_path: Path, device_name: str, half_precision: bool) -> None:
    self._model_path = model_path
    self._device_name = device_name
    self._half_precision = half_precision

  @abstractmethod
  def load(self) -> None: ...

  @abstractmethod
  def unload(self) -> None: ...

  @abstractmethod
  def predict(self, batch: BatchT) -> BatchT: ...

  @abstractmethod
  def encode(self, batch: BatchT) -> BatchT: ...

  @classmethod
  @abstractmethod
  def supports_cow(cls) -> bool: ...

  @classmethod
  @abstractmethod
  def supports_encoding(cls) -> bool: ...

  @property
  @abstractmethod
  def n_species(self) -> int: ...

  @classmethod
  @abstractmethod
  def precision(cls) -> MODEL_PRECISIONS: ...

  @classmethod
  @abstractmethod
  def name(cls) -> str: ...

  @classmethod
  def applies_sigmoid(cls) -> bool:
    """Whether the sigmoid is applied inside the model graph.

    A model that applies the sigmoid inside its graph must not get the
    pipeline's sigmoid on top: that squashes every score into [0.5, 0.73]. The
    prediction worker recovers the logits from such a model's scores when a
    non-default sigmoid sensitivity asks for the pipeline sigmoid after all.
    """
    return False

  @classmethod
  def max_batch_size(cls) -> int | None:
    """The largest batch the model accepts, or None when the batch is dynamic.

    Exports compiled with a static batch dimension fail inside the worker for
    anything larger, so the session rejects such a batch size up front.
    """
    return None

  @classmethod
  def fixed_input_samples(cls) -> int | None:
    """The only input length, in samples, the model accepts, or None for any.

    Exports compiled with a static input length fail inside the worker for any
    other segment length, so the session rejects such a length up front.
    """
    return None

  @abstractmethod
  def copy_to_device(self, batch: np.ndarray) -> BatchT: ...

  @abstractmethod
  def copy_from_device(self, inference_result: BatchT) -> np.ndarray: ...

  @abstractmethod
  def half_precision(self, inference_result: BatchT) -> BatchT: ...


@runtime_checkable
class VersionedBackendProtocol(Generic[BatchT], Protocol):
  def __init__(
    self,
    model_path: Path,
    device_name: str,
    **kwargs: Any,
  ) -> None: ...

  def load(self) -> None: ...

  def unload(self) -> None: ...

  def predict(self, batch: BatchT) -> BatchT: ...

  def encode(self, batch: BatchT) -> BatchT: ...

  @classmethod
  def supports_encoding(cls) -> bool: ...

  @classmethod
  def supports_cow(cls) -> bool: ...

  @property
  def n_species(self) -> int: ...

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS: ...

  @classmethod
  def name(cls) -> str: ...

  def copy_to_device(self, batch: np.ndarray) -> BatchT: ...

  def copy_from_device(self, inference_result: BatchT) -> np.ndarray: ...

  def half_precision(self, inference_result: BatchT) -> BatchT: ...


@runtime_checkable
class VersionedAcousticBackendProtocol(VersionedBackendProtocol, Protocol):
  @classmethod
  def applies_sigmoid(cls) -> bool: ...

  @classmethod
  def max_batch_size(cls) -> int | None: ...

  @classmethod
  def fixed_input_samples(cls) -> int | None: ...


@runtime_checkable
class VersionedGeoBackendProtocol(VersionedBackendProtocol, Protocol):
  @classmethod
  def year_round_week_inputs(cls) -> tuple[float, ...]: ...


TF_BACKEND_LIB_ARG = "inference_library"
CPU_THREADS_ARG = "cpu_threads"


def cpu_threads_per_worker(n_workers: int) -> int:
  """Intra-op threads for each of ``n_workers`` CPU inference processes.

  ONNX Runtime, PyTorch and TensorFlow size their thread pools to all cores in
  every process, so ``n_workers`` workers run ``n_workers`` times as many busy
  threads as there are cores. On 8 cores with 8 workers that made the 3.0 ONNX
  model 8.6x slower than one thread per worker.
  """
  import psutil

  n_cores = psutil.cpu_count(logical=False) or 1
  return max(1, n_cores // n_workers)


class TFBackend(Backend, ABC):
  def __init__(
    self,
    model_path: Path,
    device_name: str,
    half_precision: bool,
    **kwargs: object,
  ) -> None:
    assert device_name == "CPU"
    super().__init__(model_path, device_name, half_precision)
    self._interp: LiteRTInterpreter | TFInterpreter | None = None
    self._cached_shape: tuple[int, ...] | None = None
    self._input_idx: int | None = None
    self._prediction_idx: int | None = None
    self._embeddings_idx: int | None = None
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
  def prediction_out_idx(cls) -> int: ...

  @classmethod
  @abstractmethod
  def encoding_out_idx(cls) -> int | None: ...

  def _resolve_tensor_indices(
    self, interp: LiteRTInterpreter | TFInterpreter
  ) -> tuple[int, int, int | None]:
    """(input, prediction, embeddings) tensor indices of the loaded file.

    The official models pin them per file; a custom classifier resolves them
    from the file itself.
    """
    return self.in_idx(), self.prediction_out_idx(), self.encoding_out_idx()

  def load(self) -> None:
    assert self._interp is None

    self._interp = load_tf_model(
      self._model_path,
      self._inference_library,
      allocate_tensors=True,
    )
    self._input_idx, self._prediction_idx, self._embeddings_idx = (
      self._resolve_tensor_indices(self._interp)
    )

  def unload(self) -> None:
    self._interp = None
    self._cached_shape = None
    self._input_idx = None
    self._prediction_idx = None
    self._embeddings_idx = None

  @property
  def n_species(self) -> int:
    assert self._interp is not None
    assert self._prediction_idx is not None
    # From the prediction tensor, wherever the file lists it.
    details = self._interp.get_output_details()
    if not any(int(d["index"]) == self._prediction_idx for d in details):
      details = self._interp.get_tensor_details()
    for detail in details:
      if int(detail["index"]) == self._prediction_idx:
        return int(detail["shape"][1])
    raise ValueError(
      f"Model '{self._model_path.absolute()}' has no tensor "
      f"{self._prediction_idx} to read predictions from."
    )

  def _set_tensor(self, batch: np.ndarray) -> None:
    assert self._interp is not None
    assert self._input_idx is not None
    assert batch.flags["C_CONTIGUOUS"]
    assert batch.ndim == 2

    shape = batch.shape
    if self._cached_shape != shape:
      self._interp.resize_tensor_input(self._input_idx, shape, strict=True)
      self._interp.allocate_tensors()
      self._cached_shape = shape
    self._interp.set_tensor(self._input_idx, batch)

  def _infer(self, batch: np.ndarray, out_idx: int) -> np.ndarray:
    assert self._interp is not None

    self._set_tensor(batch)
    self._interp.invoke()
    res: np.ndarray = self._interp.get_tensor(out_idx)
    assert res.dtype == np.float32
    return res

  def _prepare_encoding(self) -> None:
    """Hook run before the first `encode` call of a loaded backend."""

  def copy_from_device(self, inference_result: np.ndarray) -> np.ndarray:
    assert isinstance(inference_result, np.ndarray)
    return inference_result

  def copy_to_device(self, batch: np.ndarray) -> np.ndarray:
    assert isinstance(batch, np.ndarray)
    return batch

  def half_precision(self, inference_result: np.ndarray) -> np.ndarray:
    assert isinstance(inference_result, np.ndarray)
    assert inference_result.dtype == np.float32
    if self._half_precision:
      inference_result = inference_result.astype(np.float16, copy=False)
      assert inference_result.dtype == np.float16
    return inference_result

  @final
  def predict(self, batch: np.ndarray) -> np.ndarray:
    assert self._prediction_idx is not None
    return self._infer(batch, self._prediction_idx)

  @final
  def encode(self, batch: np.ndarray) -> np.ndarray:
    self._prepare_encoding()
    assert self._embeddings_idx is not None
    return self._infer(batch, self._embeddings_idx)


class TorchBackend(Backend, ABC):
  def __init__(
    self,
    model_path: Path,
    device_name: str,
    half_precision: bool,
    **kwargs: object,
  ) -> None:
    super().__init__(model_path, device_name, half_precision)
    self._model: RecursiveScriptModule | None = None
    self._device: TorchDevice | None = None
    self._n_species: int | None = None
    self._cpu_threads = cast(int | None, kwargs.get(CPU_THREADS_ARG))

  @classmethod
  def name(cls) -> str:
    return MODEL_BACKEND_PT

  @final
  @classmethod
  def supports_cow(cls) -> bool:
    return False

  @classmethod
  @abstractmethod
  def prediction_out_idx(cls) -> int: ...

  @classmethod
  @abstractmethod
  def encoding_out_idx(cls) -> int | None: ...

  @classmethod
  @abstractmethod
  def probe_input_size_samples(cls) -> int: ...

  @classmethod
  def prediction_needs_sigmoid(cls) -> bool:
    """Whether the model's prediction head returns logits instead of sigmoid scores.

    The exported TorchScript modules are not consistent about this: the acoustic
    model applies the activation itself, the geo model does not. Backends whose
    model returns logits declare it here so that all backends of a model yield
    the same scores.
    """
    return False

  def load(self) -> None:
    assert self._model is None

    self._device = set_torch_device(self._device_name)
    if self._cpu_threads is not None and self._device.type == "cpu":
      import torch

      torch.set_num_threads(self._cpu_threads)
    self._model = load_torch_model(self._model_path, self._device)

  def unload(self) -> None:
    self._model = None
    self._device = None
    self._n_species = None

  @property
  def n_species(self) -> int:
    if self._n_species is None:
      probe_input = np.zeros((1, self.probe_input_size_samples()), dtype=np.float32)
      prediction = self.predict(self.copy_to_device(probe_input))
      self._n_species = int(self.copy_from_device(prediction).shape[1])
    return self._n_species

  def _infer(self, batch: TorchTensor, out_idx: int) -> TorchTensor:
    import torch

    assert self._model is not None
    with torch.inference_mode():
      outputs = self._model(batch)

    result: TorchTensor
    if isinstance(outputs, (tuple, list)):
      result = outputs[out_idx]
    elif out_idx == 0 and not self.supports_encoding():
      # Single-head models (e.g. the geo model) return the tensor directly.
      result = outputs
    else:
      raise ValueError(
        "PyTorch model is expected to return a tuple of (predictions, embeddings)."
      )
    if not isinstance(result, torch.Tensor):
      raise ValueError("PyTorch model output is expected to be a torch.Tensor.")
    assert result.dtype == torch.float32
    return result

  def copy_from_device(self, inference_result: TorchTensor) -> np.ndarray:
    import torch

    assert isinstance(inference_result, torch.Tensor)
    return inference_result.detach().cpu().numpy()

  def copy_to_device(self, batch: np.ndarray) -> TorchTensor:
    import torch

    assert isinstance(batch, np.ndarray)
    assert batch.dtype == np.float32
    assert self._device is not None
    return torch.from_numpy(batch).to(self._device)

  def half_precision(self, inference_result: TorchTensor) -> TorchTensor:
    import torch

    assert isinstance(inference_result, torch.Tensor)
    assert inference_result.dtype == torch.float32
    if self._half_precision:
      inference_result = inference_result.to(dtype=torch.float16)
      assert inference_result.dtype == torch.float16
    return inference_result

  @final
  def predict(self, batch: TorchTensor) -> TorchTensor:
    import torch

    result = self._infer(batch, self.prediction_out_idx())
    if self.prediction_needs_sigmoid():
      with torch.inference_mode():
        result = torch.sigmoid(result)
    return result

  @final
  def encode(self, batch: TorchTensor) -> TorchTensor:
    out_idx = self.encoding_out_idx()
    assert out_idx is not None
    return self._infer(batch, out_idx)


class OnnxBackend(Backend, ABC):
  def __init__(
    self,
    model_path: Path,
    device_name: str,
    half_precision: bool,
    **kwargs: object,
  ) -> None:
    super().__init__(model_path, device_name, half_precision)
    self._session: ort.InferenceSession | None = None
    self._input_name: str | None = None
    self._output_names: list[str] | None = None
    self._n_species: int | None = None
    self._cpu_threads = cast(int | None, kwargs.get(CPU_THREADS_ARG))

  @classmethod
  def name(cls) -> str:
    return MODEL_BACKEND_ONNX

  @final
  @classmethod
  def supports_cow(cls) -> bool:
    return False

  @classmethod
  @abstractmethod
  def prediction_out_idx(cls) -> int: ...

  @classmethod
  @abstractmethod
  def encoding_out_idx(cls) -> int | None: ...

  @classmethod
  @abstractmethod
  def probe_input_size_samples(cls) -> int: ...

  def load(self) -> None:
    assert self._session is None

    self._session = load_onnx_model(
      self._model_path, self._device_name, self._cpu_threads
    )
    self._input_name = self._session.get_inputs()[0].name
    self._output_names = [output.name for output in self._session.get_outputs()]

  def unload(self) -> None:
    self._session = None
    self._input_name = None
    self._output_names = None
    self._n_species = None

  @property
  def n_species(self) -> int:
    if self._n_species is None:
      assert self._session is not None
      pred_output = self._session.get_outputs()[self.prediction_out_idx()]
      pred_shape = pred_output.shape
      if len(pred_shape) > 1 and isinstance(pred_shape[1], int):
        self._n_species = pred_shape[1]
      else:
        probe_input = np.zeros((1, self.probe_input_size_samples()), dtype=np.float32)
        prediction = self.predict(self.copy_to_device(probe_input))
        self._n_species = int(self.copy_from_device(prediction).shape[1])
    return self._n_species

  def _infer(self, batch: np.ndarray, out_idx: int) -> np.ndarray:
    assert self._session is not None
    assert self._input_name is not None
    assert self._output_names is not None

    [result] = self._session.run(
      [self._output_names[out_idx]],
      {self._input_name: batch},
    )
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    return result

  def copy_from_device(self, inference_result: np.ndarray) -> np.ndarray:
    assert isinstance(inference_result, np.ndarray)
    return inference_result

  def copy_to_device(self, batch: np.ndarray) -> np.ndarray:
    assert isinstance(batch, np.ndarray)
    assert batch.dtype == np.float32
    return batch

  def half_precision(self, inference_result: np.ndarray) -> np.ndarray:
    assert isinstance(inference_result, np.ndarray)
    assert inference_result.dtype == np.float32
    if self._half_precision:
      inference_result = inference_result.astype(np.float16, copy=False)
      assert inference_result.dtype == np.float16
    return inference_result

  @final
  def predict(self, batch: np.ndarray) -> np.ndarray:
    return self._infer(batch, self.prediction_out_idx())

  @final
  def encode(self, batch: np.ndarray) -> np.ndarray:
    out_idx = self.encoding_out_idx()
    assert out_idx is not None
    return self._infer(batch, out_idx)


class PBBackend(Backend, ABC):
  def __init__(
    self,
    model_path: Path,
    device_name: str,
    half_precision: bool,
    **kwargs: object,
  ) -> None:
    super().__init__(model_path, device_name, half_precision)
    self._model: Any | None = None
    self._logical_device_name: str | None = None
    self._predict_fn: Callable | None = None
    self._encoding_fn: Callable | None = None
    self._cpu_threads = cast(int | None, kwargs.get(CPU_THREADS_ARG))

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
  def prediction_signature_name(cls) -> str: ...

  @classmethod
  @abstractmethod
  def prediction_key(cls) -> str: ...

  @classmethod
  @abstractmethod
  def encoding_signature_name(cls) -> str | None: ...

  @classmethod
  @abstractmethod
  def encoding_key(cls) -> str | None: ...

  @final
  def load(self) -> None:
    import_tf()
    self._set_logical_device()
    assert self._logical_device_name is not None
    self._model = load_pb_model(self._model_path, self._logical_device_name)
    self._predict_fn = self._signature(self.prediction_signature_name())
    if self.supports_encoding():
      encoding_sig_name = self.encoding_signature_name()
      assert encoding_sig_name is not None
      self._encoding_fn = self._signature(encoding_sig_name)

  def _signature(self, name: str) -> Callable:
    assert self._model is not None
    signatures = self._model.signatures
    if name not in signatures:
      available = ", ".join(sorted(signatures.keys())) or "none"
      raise ValueError(
        f"SavedModel '{self._model_path.absolute()}' has no '{name}' signature "
        f"(available: {available}). A Raven export from BirdNET-Analyzer carries "
        "the 'basic' signature and needs is_raven=True; the official SavedModel "
        "layout needs is_raven=False."
      )
    signature: Callable = signatures[name]
    return signature

  def unload(self) -> None:
    self._model = None
    self._logical_device_name = None
    self._predict_fn = None
    self._encoding_fn = None

  @property
  def n_species(self) -> int:
    assert self._predict_fn is not None
    n_species_in_model: int = (
      self._predict_fn.output_shapes[self.prediction_key()].dims[1].value  # type: ignore
    )
    return n_species_in_model

  def _set_logical_device(self) -> None:
    if "CPU" in self._device_name:
      self._logical_device_name = set_cpu_device_tf(self._cpu_threads)
    elif "GPU" in self._device_name:
      self._logical_device_name = set_gpu_device_tf(
        self._device_name, memory_growth=True
      )
    else:
      raise AssertionError()

  def copy_to_device(self, batch: np.ndarray) -> Tensor:
    from tensorflow import convert_to_tensor, device, float32

    assert self._logical_device_name is not None
    assert batch.dtype == np.float32

    with device(self._logical_device_name):  # type: ignore
      tensor = convert_to_tensor(batch, dtype=float32)
    return tensor

  @final
  def predict(self, batch: Tensor) -> Tensor:
    from tensorflow import device, float32

    assert self._logical_device_name is not None
    assert self._predict_fn is not None

    with device(self._logical_device_name):  # type: ignore
      prediction_result = self._predict_fn(**{self.input_key(): batch})
    predictions: Tensor = prediction_result[self.prediction_key()]
    assert predictions.dtype == float32
    return predictions

  @final
  def encode(self, batch: Tensor) -> Tensor:
    from tensorflow import device, float32

    assert self.supports_encoding()
    encoding_key = self.encoding_key()
    assert encoding_key is not None
    assert self._encoding_fn is not None
    assert self._logical_device_name is not None

    with device(self._logical_device_name):  # type: ignore
      encoding_result = self._encoding_fn(**{self.input_key(): batch})
    embeddings: Tensor = encoding_result[encoding_key]
    assert embeddings.dtype == float32
    return embeddings

  def copy_from_device(self, inference_result: Tensor) -> np.ndarray:
    from tensorflow import Tensor

    assert isinstance(inference_result, Tensor)
    inference_result_np = inference_result.numpy()
    return inference_result_np

  def half_precision(self, inference_result: Tensor) -> Tensor:
    from tensorflow import Tensor, cast, device, float16, float32

    assert self._logical_device_name is not None
    assert isinstance(inference_result, Tensor)
    assert inference_result.dtype == float32
    with device(self._logical_device_name):  # type: ignore
      inference_result = cast(inference_result, float16)
      assert inference_result.dtype == float16
    return inference_result


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

  def _load_backend(
    self, device_name: str, half_precision: bool
  ) -> VersionedBackendProtocol:
    assert self._backend is None
    backend = self._backend_type(
      model_path=self._model_path,
      device_name=device_name,
      half_precision=half_precision,
      **self._backend_kwargs,
    )
    backend.load()
    self._backend = backend
    return backend

  def load_backend_in_main_process_if_possible(
    self, devices: list[str], half_precision: bool, start_method: str
  ) -> None:
    # Only "fork" children inherit the parent's memory (copy-on-write), so
    # pre-loading the backend here only pays off then. The session's effective
    # start method is passed in because the global mp.get_start_method() can
    # differ from the context the pipeline actually uses.
    unique_devices = set(devices)
    same_device_for_all_workers = len(unique_devices) == 1
    if (
      same_device_for_all_workers
      and start_method == "fork"
      and self._backend_type.supports_cow()
    ):
      device_name = unique_devices.pop()
      self._load_backend(device_name, half_precision)

  def load_backend(
    self, device_name: str, half_precision: bool
  ) -> VersionedBackendProtocol:
    if self._backend is None:
      return self._load_backend(device_name, half_precision)
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
  ) -> int:
    loader = cls(model_path, backend_type, kwargs)
    loader.load_backend("CPU", half_precision=False)
    n_species_in_model = loader.backend.n_species
    return n_species_in_model

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
      n_species_in_model: int = run_in_subprocess(
        cls._get_n_species,
        (model_path, backend_type, kwargs),
        timeout_s=_SUBPROCESS_TIMEOUT_S,
      )
      return n_species_in_model
    except _SubprocessTimeout as e:
      raise ValueError("Failed to load model due to timeout.") from e
    except Exception as e:
      raise ValueError(f"Failed to load model: {e}") from e

  @classmethod
  def inspect_custom_tflite_model(
    cls,
    model_path: Path,
    library: LIBRARY_TYPES,
    legacy_prediction_indices: Collection[int],
  ) -> CustomTFLiteInfo:
    """Describe a custom TFLite classifier in a subprocess.

    The subprocess keeps TensorFlow (or LiteRT) out of the calling process, so
    an application can inspect a file before deciding how to load it.
    """
    from birdnet.acoustic.models.custom_tflite import inspect_custom_tflite_file

    try:
      info: CustomTFLiteInfo = run_in_subprocess(
        inspect_custom_tflite_file,
        (model_path, library, tuple(legacy_prediction_indices)),
        timeout_s=_SUBPROCESS_TIMEOUT_S,
      )
      return info
    except _SubprocessTimeout as e:
      raise ValueError("Failed to inspect model due to timeout.") from e
    except Exception as e:
      raise ValueError(
        f"Failed to inspect custom TFLite model '{model_path.absolute()}': {e}"
      ) from e


# The validity check and the inspection open the model in a subprocess; a
# model that takes longer than this is reported as a timeout.
_SUBPROCESS_TIMEOUT_S = 180


class _SubprocessTimeout(TimeoutError):
  """`run_in_subprocess` stopped waiting; a `TimeoutError` the function itself
  raised arrives as a plain one."""


def run_in_subprocess(
  fn: Callable[..., Any], args: tuple[Any, ...], timeout_s: float
) -> Any:  # noqa: ANN401
  """Run `fn(*args)` in a fresh process and return what it returns.

  A model that hangs while opening would otherwise hold up the caller, and a
  pool worker cannot be killed, so this is a process of its own: the timeout
  terminates it, and an exception raised inside is raised here. Only builtin
  exceptions arrive as themselves; any other becomes a `RuntimeError` naming
  its type, because unpickling it would import its module (TensorFlow, for a
  TensorFlow error) into this process.
  """
  # Pickled before anything is started: on Windows the process exists before
  # its target is pickled, and one whose start fails is left alive, waiting on
  # a pipe handle number the next start may reuse.
  call = pickle.dumps((fn, args))
  ctx: Any = multiprocessing.get_context(resolve_start_method())
  receiver, sender = ctx.Pipe(duplex=False)
  process = ctx.Process(target=_send_outcome, args=(sender, call), daemon=True)
  try:
    process.start()
    sender.close()
    if not receiver.poll(timeout_s):
      raise _SubprocessTimeout(
        f"{getattr(fn, '__name__', fn)} did not finish within {timeout_s:.0f} s"
      )
    try:
      outcome, payload = receiver.recv()
    except EOFError as err:
      process.join(5)
      raise RuntimeError(
        f"The subprocess exited with code {process.exitcode} before reporting."
      ) from err
  finally:
    if process.is_alive():
      process.kill()
      process.join(5)
    receiver.close()
    sender.close()
  if outcome == "error":
    raise payload
  return payload


def _send_outcome(sender: Any, call: bytes) -> None:  # noqa: ANN401
  try:
    fn, args = pickle.loads(call)
    outcome: tuple[str, Any] = ("ok", fn(*args))
  except BaseException as err:  # noqa: BLE001
    if type(err).__module__ != "builtins":
      err = RuntimeError(f"{type(err).__name__}: {err}")
    outcome = ("error", err)
  try:
    sender.send(outcome)
  except Exception:
    # Not everything pickles; the parent still gets to know what happened.
    sender.send(("error", RuntimeError(f"{type(outcome[1]).__name__}: {outcome[1]}")))
  finally:
    sender.close()


def import_tf() -> None:
  disable_tf_logging()
  # The absl banner is written by native code to fd 2 and is out of reach of
  # `disable_tf_logging`; only the descriptor redirect silences it. Lines that
  # TensorFlow emits later, at device creation, fall outside this block.
  with suppress_native_stderr():
    import tensorflow  # noqa: F401


def set_cpu_device_tf(cpu_threads: int | None = None) -> str:
  from tensorflow.config import list_logical_devices  # type: ignore

  # disable GPUs for TF when using CPU backend
  # because if GPU is available TF will try to use it by default
  # and raise: tensorflow.python.framework.errors_impl.InternalError:
  # cudaSetDevice() on GPU:0 failed. Status: out of memory
  # at call of tensorflow.config.list_logical_devices("CPU")
  os.environ["CUDA_VISIBLE_DEVICES"] = ""

  if cpu_threads is not None:
    from tensorflow.config import threading  # type: ignore

    try:
      threading.set_intra_op_parallelism_threads(cpu_threads)
      threading.set_inter_op_parallelism_threads(cpu_threads)
    except RuntimeError:
      # The runtime is already initialized in this process (a forked child of a
      # parent that ran TensorFlow); it keeps the pools it was created with.
      pass

  logical_devices = list_logical_devices("CPU")

  if len(logical_devices) == 0:
    raise ValueError(
      "No CPU found! "
      "Please check your TensorFlow installation and ensure that a CPU is "
      "available."
    )

  if len(logical_devices) > 1:
    raise ValueError(
      f"Multiple CPUs found ({len(logical_devices)}). "
      "Please ensure that only one CPU is available."
    )
  dev = logical_devices[0]
  return dev.name


def set_gpu_device_tf(device: str, memory_growth: bool) -> str:
  device_index = int(device.split(":")[1]) if ":" in device else 0
  os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)

  # Note: memory growth needs to be set before loading the model and
  # only once in the main process
  import tensorflow as tf

  if memory_growth:
    physical_devices = tf.config.list_physical_devices("GPU")
    assert len(physical_devices) == 1
    physical_device = physical_devices[0]
    tf.config.experimental.set_memory_growth(physical_device, True)

  logical_devices = tf.config.list_logical_devices("GPU")

  if len(logical_devices) == 0:
    raise ValueError(
      "No GPU found! "
      "Please check your TensorFlow installation and ensure that a GPU is "
      "available. Also ensure that birdnet is installed with GPU support "
      "(pip install birdnet[and-cuda])."
    )

  assert len(logical_devices) == 1
  dev = logical_devices[0]
  return dev.name


def disable_tf_logging() -> None:
  import absl.logging

  absl.logging.set_verbosity(absl.logging.ERROR)
  logging.getLogger("tensorflow").setLevel(logging.ERROR)
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_pb_model(model_path: Path, logical_device_name: str) -> Any:
  import tensorflow as tf

  start = time.perf_counter()
  with tf.device(logical_device_name):  # type: ignore
    model = tf.saved_model.load(str(model_path.absolute()))
  end = time.perf_counter()

  logger = get_logger_for_package(__name__)
  logger.debug(
    f"Model loaded from {model_path.absolute()} in {end - start:.2f} seconds."
  )
  return model


@overload
def load_tf_model(
  model_path: Path,
  library: Literal["tflite"],
  allocate_tensors: bool = False,
  preserve_all_tensors: bool = False,
) -> TFInterpreter: ...
@overload
def load_tf_model(
  model_path: Path,
  library: Literal["litert"],
  allocate_tensors: bool = False,
  preserve_all_tensors: bool = False,
) -> LiteRTInterpreter: ...
@overload
def load_tf_model(
  model_path: Path,
  library: LIBRARY_TYPES,
  allocate_tensors: bool = False,
  preserve_all_tensors: bool = False,
) -> TFInterpreter | LiteRTInterpreter: ...


def load_tf_model(
  model_path: Path,
  library: LIBRARY_TYPES,
  allocate_tensors: bool = False,
  preserve_all_tensors: bool = False,
):
  # preserve_all_tensors keeps intermediate tensors readable after invoke().
  # The interpreter otherwise reuses their memory, so reading one (a custom
  # 2.4 classifier's embeddings) returns whatever was computed there last.
  if library == LIBRARY_TFLITE:
    return load_lib_tf_model(
      model_path,
      allocate_tensors=allocate_tensors,
      preserve_all_tensors=preserve_all_tensors,
    )
  elif library == LIBRARY_LITERT:
    return load_lib_litert_model(
      model_path,
      allocate_tensors=allocate_tensors,
      preserve_all_tensors=preserve_all_tensors,
    )
  else:
    raise AssertionError()


def load_lib_tf_model(
  model_path: Path,
  allocate_tensors: bool = False,
  preserve_all_tensors: bool = False,
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
  # Importing it this way is not possible:
  # `import tensorflow.lite.python.interpreter as tflite`
  # This is where the tf backend first pulls TensorFlow in, so it is the import
  # that emits the native startup banner.
  with suppress_native_stderr():
    from tensorflow.lite.python import interpreter as tflite

  start = time.perf_counter()

  import warnings

  # tf.lite.Interpreter is deprecated but faster than LiteRT's (238 vs. 183 seg/s).
  with warnings.catch_warnings():
    warnings.filterwarnings(
      "ignore",
      message=r".*tf\.lite\.Interpreter is deprecated.*",
      category=UserWarning,
      module="tensorflow.lite.python.interpreter",
    )
    try:
      interp = tflite.Interpreter(
        str(model_path.absolute()),
        num_threads=1,
        experimental_op_resolver_type=tflite.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        experimental_preserve_all_tensors=preserve_all_tensors,
        # tensor#187 is a dynamic-sized tensor
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
  preserve_all_tensors: bool = False,
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
      experimental_preserve_all_tensors=preserve_all_tensors,
      # tensor#187 is a dynamic-sized tensor
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


def set_torch_device(device: str) -> TorchDevice:
  import torch

  if "CPU" in device:
    return torch.device("cpu")
  elif "GPU" in device:
    if not torch.cuda.is_available():
      raise ValueError(
        "No GPU found! Please check your PyTorch installation and ensure that a "
        "CUDA-capable GPU is available."
      )

    device_index = int(device.split(":")[1]) if ":" in device else 0
    if device_index >= torch.cuda.device_count():
      raise ValueError(
        f"Requested GPU index {device_index} is out of range for "
        f"{torch.cuda.device_count()} available CUDA device(s)."
      )
    return torch.device(f"cuda:{device_index}")
  else:
    raise AssertionError()


def load_torch_model(model_path: Path, device: TorchDevice) -> RecursiveScriptModule:
  assert model_path.is_file()
  assert torch_installed()

  import torch

  start = time.perf_counter()
  try:
    model = torch.jit.load(str(model_path.absolute()), map_location=device)
  except RuntimeError as e:
    raise ValueError(
      f"Failed to load model '{model_path.absolute()}' using 'torch'. "
      "Ensure it is a valid TorchScript model."
    ) from e
  model.eval()
  end = time.perf_counter()

  logger = get_logger_for_package(__name__)
  logger.debug(
    f"Model loaded from {model_path.absolute()} using 'torch' "
    f"in {end - start:.2f} seconds."
  )
  return model


def _get_onnx_session_providers(device: str) -> tuple[list[str], list[dict[str, str]]]:
  import onnxruntime as ort

  if "CPU" in device:
    return ["CPUExecutionProvider"], [{}]

  if "GPU" not in device:
    raise AssertionError()

  device_index = int(device.split(":")[1]) if ":" in device else 0
  available = ort.get_available_providers()
  if "CUDAExecutionProvider" in available:
    return ["CUDAExecutionProvider", "CPUExecutionProvider"], [
      {"device_id": str(device_index)},
      {},
    ]
  if "DmlExecutionProvider" in available:
    return ["DmlExecutionProvider", "CPUExecutionProvider"], [
      {"device_id": str(device_index)},
      {},
    ]

  raise ValueError(
    "No ONNX Runtime GPU execution provider found. Install an ONNX Runtime "
    "build with GPU support."
  )


def load_onnx_model(
  model_path: Path, device: str, cpu_threads: int | None = None
) -> ort.InferenceSession:
  assert model_path.is_file()
  assert onnxruntime_installed()

  import onnxruntime as ort

  providers, provider_options = _get_onnx_session_providers(device)
  sess_options = ort.SessionOptions()
  if cpu_threads is not None and "CPU" in device:
    sess_options.intra_op_num_threads = cpu_threads
  start = time.perf_counter()
  try:
    session = ort.InferenceSession(
      str(model_path.absolute()),
      sess_options=sess_options,
      providers=providers,
      provider_options=provider_options,
    )
  except Exception as e:
    raise ValueError(
      f"Failed to load model '{model_path.absolute()}' using 'onnxruntime'. "
      "Ensure it is a valid ONNX model."
    ) from e
  end = time.perf_counter()

  logger = get_logger_for_package(__name__)
  actual_provider = session.get_providers()[0] if session.get_providers() else "unknown"
  logger.debug(
    f"Model loaded from {model_path.absolute()} using 'onnxruntime' "
    f"with provider '{actual_provider}' in {end - start:.2f} seconds."
  )
  return session


def tf_installed() -> bool:
  import importlib.util

  return importlib.util.find_spec("tensorflow") is not None


def _get_major_minor_version(version: str) -> tuple[int, int]:
  version_parts = version.split(".")
  if len(version_parts) < 2:
    raise RuntimeError(f"Could not parse TensorFlow version {version!r}.")

  try:
    return int(version_parts[0]), int(version_parts[1])
  except ValueError as e:
    raise RuntimeError(f"Could not parse TensorFlow version {version!r}.") from e


def torch_installed() -> bool:
  import importlib.util

  return importlib.util.find_spec("torch") is not None


def onnxruntime_installed() -> bool:
  import importlib.util

  return importlib.util.find_spec("onnxruntime") is not None


def litert_installed() -> bool:
  import importlib.util

  return importlib.util.find_spec("ai_edge_litert") is not None
