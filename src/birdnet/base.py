import os
import time
from abc import ABC, abstractmethod
from multiprocessing import current_process
from pathlib import Path
from threading import current_thread
from typing import Self

import numpy as np
from ordered_set import OrderedSet

from birdnet.helper import get_float_dtype, get_hash, get_uint_dtype

NP_INPUTS_KEY = "inputs"
NP_INPUT_DURATIONS_KEY = "input_durations"
NP_SEGMENT_DURATION_S_KEY = "segment_duration_s"
NP_OVERLAP_DURATION_S_KEY = "overlap_duration_s"
NP_SPEED_KEY = "speed"
NP_MODEL_PATH_KEY = "model_path"
NP_MODEL_FMIN_KEY = "model_fmin"
NP_MODEL_FMAX_KEY = "model_fmax"
NP_MODEL_SR_KEY = "model_sr"
NP_MODEL_PRECISION_KEY = "model_precision"
NP_MODEL_VERSION_KEY = "model_version"


class ResultBase(ABC):
  def __init__(
    self,
    inputs: np.ndarray,
    input_durations: np.ndarray,
    segment_duration_s: int | float,
    overlap_duration_s: int | float,
    speed: int | float,
    model_path: Path,
    model_fmin: int,
    model_fmax: int,
    model_sr: int,
    model_precision: str,
    model_version: str,
  ) -> None:
    super().__init__()

    assert input_durations.dtype in (np.float16, np.float32, np.float64)

    self._inputs = inputs
    self._input_durations = input_durations

    self._segment_duration_s = np.array(
      [segment_duration_s], dtype=get_float_dtype(segment_duration_s)
    )
    self._overlap_duration_s = np.array(
      [overlap_duration_s], dtype=get_float_dtype(overlap_duration_s)
    )
    self._speed = np.array([speed], dtype=get_float_dtype(speed))
    self._model_path = np.array(
      [str(model_path)], dtype=np.dtype("U" + str(len(str(model_path.absolute()))))
    )
    self._model_fmin = np.array([model_fmin], dtype=get_uint_dtype(model_fmin))
    self._model_fmax = np.array([model_fmax], dtype=get_uint_dtype(model_fmax))
    self._model_sr = np.array([model_sr], dtype=get_uint_dtype(model_sr))
    self._model_precision = np.array(
      [model_precision],
      dtype=np.dtype("U" + str(len(model_precision))),
    )
    self._model_version = np.array(
      [model_version],
      dtype=np.dtype("U" + str(len(model_version))),
    )

  @property
  def segment_duration_s(self) -> float:
    return float(self._segment_duration_s[0])

  @property
  def overlap_duration_s(self) -> float:
    return float(self._overlap_duration_s[0])

  @property
  def speed(self) -> float:
    return float(self._speed[0])

  @property
  def inputs(self) -> np.ndarray:
    return self._inputs

  @property
  def n_inputs(self) -> int:
    return self._inputs.shape[0]

  @property
  def input_durations(self) -> np.ndarray:
    return self._input_durations

  @property
  def model_path(self) -> Path:
    return Path(self._model_path[0])

  @property
  def model_fmin(self) -> int:
    return int(self._model_fmin[0])

  @property
  def model_fmax(self) -> int:
    return int(self._model_fmax[0])

  @property
  def model_sr(self) -> int:
    return int(self._model_sr[0])

  @property
  def model_precision(self) -> str:
    return str(self._model_precision[0])

  @property
  def model_version(self) -> str:
    return str(self._model_version[0])

  def _get_extra_save_data(self) -> dict[str, np.ndarray]:
    return {}

  @classmethod
  @abstractmethod
  def _set_extra_load_data(cls, data: dict[str, np.ndarray]) -> None: ...

  def save(self, npz_out_path: os.PathLike | str, /, *, compress: bool = True) -> None:
    npz_out_path = Path(npz_out_path)
    if npz_out_path.suffix != ".npz":
      raise ValueError("Output path must have a .npz suffix")

    save_method = np.savez_compressed if compress else np.savez

    extra_data = self._get_extra_save_data()

    data = {
      NP_INPUTS_KEY: self._inputs,
      NP_INPUT_DURATIONS_KEY: self._input_durations,
      NP_SEGMENT_DURATION_S_KEY: self._segment_duration_s,
      NP_OVERLAP_DURATION_S_KEY: self._overlap_duration_s,
      NP_SPEED_KEY: self._speed,
      NP_MODEL_PATH_KEY: self._model_path,
      NP_MODEL_VERSION_KEY: self._model_version,
      NP_MODEL_FMIN_KEY: self._model_fmin,
      NP_MODEL_FMAX_KEY: self._model_fmax,
      NP_MODEL_SR_KEY: self._model_sr,
      NP_MODEL_PRECISION_KEY: self._model_precision,
    }

    assert extra_data.keys().isdisjoint(data.keys())
    data.update(extra_data)

    save_method(npz_out_path, **data)

  @classmethod
  def load(cls, path: os.PathLike | str) -> Self:
    result = cls.__new__(cls)
    with np.load(path, allow_pickle=True) as npz:
      data = {k: npz[k] for k in npz.files}

    result._inputs = data[NP_INPUTS_KEY]
    result._input_durations = data[NP_INPUT_DURATIONS_KEY]
    result._segment_duration_s = data[NP_SEGMENT_DURATION_S_KEY]
    result._overlap_duration_s = data[NP_OVERLAP_DURATION_S_KEY]
    result._speed = data[NP_SPEED_KEY]
    result._model_path = data[NP_MODEL_PATH_KEY]
    result._model_version = data[NP_MODEL_VERSION_KEY]
    result._model_fmin = data[NP_MODEL_FMIN_KEY]
    result._model_fmax = data[NP_MODEL_FMAX_KEY]
    result._model_sr = data[NP_MODEL_SR_KEY]
    result._model_precision = data[NP_MODEL_PRECISION_KEY]

    result._set_extra_load_data(data)

    return result

  @property
  def memory_size_mb(self) -> float:
    return (
      self._inputs.nbytes
      + self._input_durations.nbytes
      + self._segment_duration_s.nbytes
      + self._overlap_duration_s.nbytes
      + self._speed.nbytes
      + self._model_path.nbytes
      + self._model_fmin.nbytes
      + self._model_fmax.nbytes
      + self._model_sr.nbytes
      + self._model_precision.nbytes
    ) / 1024**2


class SessionBase(ABC):
  def __init__(self) -> None:
    self._session_id = get_session_id()

  @abstractmethod
  def __enter__(self) -> Self: ...

  @abstractmethod
  def __exit__(self, *args): ...

  @abstractmethod
  def run(self, *args, **kwargs) -> ResultBase: ...


def get_session_id() -> str:
  """
  Get a unique session ID based on the current process and thread.

  Example for two processes (fork):
    Process 1: 53554_127397175535424_1762165676846175803
    Process 2: 53555_127397175535424_1762165676846559511

  Example for two processes (spawn):
    Process 1: 54155_126834937165632_1762165717644505438
    Process 2: 54154_132842492557120_1762165717644777865

  Example for two threads in the same process:
    Thread 1: 53142_138235445503680_1762165643891762916
    Thread 2: 53142_138235453896384_1762165653498085145

  Example for same thread and process but different calls:
    Call 1: 50179_128078941120320_1762165462208616340
    Call 2: 50179_128078941120320_1762165485281125126
  """
  proc = current_process()
  thread = current_thread()
  timestamp = time.time_ns()
  result = f"{proc.ident}_{thread.ident}_{timestamp}"
  return result


def get_session_id_hash(session_id: str) -> str:
  hash_digest = get_hash(session_id)[:5]
  return hash_digest


class ModelBase(ABC):
  def __init__(
    self,
    model_path: Path,
    species_list: OrderedSet[str],
    use_custom_model: bool,
  ) -> None:
    super().__init__()
    self._model_path = model_path
    self._species_list = species_list
    self._use_custom_model = use_custom_model

  @property
  def model_path(self) -> Path:
    return self._model_path

  @property
  def species_list(self) -> OrderedSet[str]:
    return self._species_list

  @property
  def n_species(self) -> int:
    return len(self.species_list)

  @property
  def use_custom_model(self) -> bool:
    return self._use_custom_model

  @classmethod
  @abstractmethod
  def load(cls, *args, **kwargs) -> Self:  # noqa: ANN002, ANN003
    ...

  @classmethod
  @abstractmethod
  def load_custom(cls, *args, **kwargs) -> Self:  # noqa: ANN002, ANN003
    ...

  @classmethod
  @abstractmethod
  def predict(cls, *args, **kwargs) -> ResultBase:  # noqa: ANN002, ANN003
    ...

  @classmethod
  @abstractmethod
  def predict_session(cls, *args, **kwargs) -> SessionBase:  # noqa: ANN002, ANN003
    ...
