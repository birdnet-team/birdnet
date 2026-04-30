from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from multiprocessing import current_process
from pathlib import Path
from threading import current_thread
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from ordered_set import OrderedSet

from birdnet.core.base import ResultBase
from birdnet.utils.helper import (
  get_float_dtype,
  get_hash,
  get_hop_duration_s,
  get_uint_dtype,
)

if TYPE_CHECKING:
  import pandas as pd
  import pyarrow as pa

VAR_INPUT = "input"
VAR_START_TIME = "start_time"
VAR_END_TIME = "end_time"

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


class AcousticResultBase(ResultBase):
  """Base container for shared acoustic model result metadata and helpers."""

  def __init__(
    self,
    model_path: Path,
    model_version: str,
    model_precision: str,
    inputs: np.ndarray,
    input_durations: np.ndarray,
    segment_duration_s: int | float,
    overlap_duration_s: int | float,
    speed: int | float,
    model_fmin: int,
    model_fmax: int,
    model_sr: int,
  ) -> None:
    """Capture model metadata plus per-input timing information.

    Args:
      model_path: Path to the acoustic model binary used for inference.
      model_version: Version identifier of the model.
      model_precision: Precision string (float16/float32) backed by the model.
      inputs: Typed array of inputs (paths or indices) that were encoded/predicted.
      input_durations: Duration of each input in seconds.
      segment_duration_s: Segment length used by the inference pipeline.
      overlap_duration_s: Overlap between consecutive segments.
      speed: Speed multiplier applied to the inputs during preprocessing.
      model_fmin: Lower frequency bound used by the model.
      model_fmax: Upper frequency bound used by the model.
      model_sr: Sampling rate that the model expects.
    """
    super().__init__(
      model_path=model_path,
      model_version=model_version,
      model_precision=model_precision,
    )

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

    self._model_fmin = np.array([model_fmin], dtype=get_uint_dtype(model_fmin))
    self._model_fmax = np.array([model_fmax], dtype=get_uint_dtype(model_fmax))
    self._model_sr = np.array([model_sr], dtype=get_uint_dtype(model_sr))

  @property
  def _input_dtype(self) -> type:
    return self._inputs.dtype

  @property
  def segment_duration_s(self) -> float:
    """Segment duration as configured on the inference pipeline."""
    return float(self._segment_duration_s[0])

  @property
  def overlap_duration_s(self) -> float:
    """Overlap duration between sliding windows in seconds."""
    return float(self._overlap_duration_s[0])

  @property
  def speed(self) -> float:
    """Speed multiplier that was applied to the inputs."""
    return float(self._speed[0])

  @property
  def inputs(self) -> np.ndarray:
    """Identifiers for each input processed by the result."""
    return self._inputs

  @property
  def n_inputs(self) -> int:
    """Number of inputs in the result payload."""
    return self._inputs.shape[0]

  @property
  def input_durations(self) -> np.ndarray:
    """Durations of each input in seconds."""
    return self._input_durations

  @property
  def model_fmin(self) -> int:
    """Lower bound of the model's bandpass filter."""
    return int(self._model_fmin[0])

  @property
  def model_fmax(self) -> int:
    """Upper bound of the model's bandpass filter."""
    return int(self._model_fmax[0])

  @property
  def model_sr(self) -> int:
    """Sampling rate expected by the model."""
    return int(self._model_sr[0])

  @property
  def hop_duration_s(self) -> float:
    return get_hop_duration_s(
      self._segment_duration_s[0], self._overlap_duration_s[0], self._speed[0]
    )

  def _get_extra_save_data(self) -> dict[str, np.ndarray]:
    return {
      NP_INPUTS_KEY: self._inputs,
      NP_INPUT_DURATIONS_KEY: self._input_durations,
      NP_SEGMENT_DURATION_S_KEY: self._segment_duration_s,
      NP_OVERLAP_DURATION_S_KEY: self._overlap_duration_s,
      NP_SPEED_KEY: self._speed,
      NP_MODEL_FMIN_KEY: self._model_fmin,
      NP_MODEL_FMAX_KEY: self._model_fmax,
      NP_MODEL_SR_KEY: self._model_sr,
    }

  @classmethod
  def _set_extra_load_data(cls, data: dict[str, np.ndarray]) -> None:
    cls._inputs = data[NP_INPUTS_KEY]
    cls._input_durations = data[NP_INPUT_DURATIONS_KEY]
    cls._segment_duration_s = data[NP_SEGMENT_DURATION_S_KEY]
    cls._overlap_duration_s = data[NP_OVERLAP_DURATION_S_KEY]
    cls._speed = data[NP_SPEED_KEY]
    cls._model_fmin = data[NP_MODEL_FMIN_KEY]
    cls._model_fmax = data[NP_MODEL_FMAX_KEY]
    cls._model_sr = data[NP_MODEL_SR_KEY]

  @property
  def memory_size_MiB(self) -> float:
    """Memory usage for the base result metadata.

    Returns:
      float: Memory used by metadata buffers in mebibytes.
    """
    return super().memory_size_MiB + (
      (
        self._inputs.nbytes
        + self._input_durations.nbytes
        + self._segment_duration_s.nbytes
        + self._overlap_duration_s.nbytes
        + self._speed.nbytes
        + self._model_fmin.nbytes
        + self._model_fmax.nbytes
        + self._model_sr.nbytes
      )
      / 1024**2
    )

  @abstractmethod
  def to_structured_array(self) -> np.ndarray: ...

  @abstractmethod
  def to_arrow_table(self) -> pa.Table: ...

  @abstractmethod
  def to_csv(
    self,
    path: os.PathLike | str,
    *,
    encoding: str = "utf-8",
    buffer_size_kb: int = 1024,
    silent: bool = False,
  ) -> None: ...

  def to_dataframe(self) -> pd.DataFrame:
    """Convert the structured array into a pandas DataFrame."""
    import pandas as pd

    structured = self.to_structured_array()
    df_data: dict[str, object] = {}

    dtype_names = structured.dtype.names
    if dtype_names is None:
      return pd.DataFrame(structured, copy=True)

    for name in dtype_names:
      column = structured[name]
      df_data[name] = column.tolist() if column.ndim > 1 else column

    return pd.DataFrame(df_data)

  def to_parquet(
    self,
    path: os.PathLike | str,
    *,
    compression: Literal["none", "snappy", "gzip", "brotli", "lz4", "zstd"] = "snappy",
    compression_level: int | None = None,
    silent: bool = False,
  ) -> None:
    """Write the contents to disk as an Arrow Parquet file."""
    import pyarrow.parquet as pq

    path = Path(path)
    if path.suffix != ".parquet":
      raise ValueError("Output path must have a .parquet suffix")

    if not silent:
      print("Creating Arrow table...")  # noqa: T201

    table = self.to_arrow_table()

    if not silent:
      print(f"Writing Parquet to {path.absolute()} ...")  # noqa: T201

    pq.write_table(
      table,
      path,
      compression=compression,
      compression_level=compression_level,
    )

    if not silent:
      file_size = path.stat().st_size / 1024**2
      original_size = table.nbytes / 1024**2
      compression_ratio = original_size / file_size if file_size > 0 else 0
      print(f"Parquet file: {file_size:.1f} MB (compression: {compression_ratio:.1f}x)")  # noqa: T201

  def __iter__(self) -> Iterable[np.ndarray]:
    """Iterate over the structured records via the iterator protocol."""
    yield from self.to_structured_array()


class SessionBase(ABC):
  def __init__(self) -> None:
    self._session_id = get_session_id()

  @abstractmethod
  def __enter__(self) -> Self: ...

  @abstractmethod
  def __exit__(self, *args: object) -> None: ...

  @abstractmethod
  def run(self, *args: object, **kwargs: object) -> ResultBase: ...


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
    is_custom_model: bool,
  ) -> None:
    super().__init__()
    self._model_path = model_path
    self._species_list = species_list
    self._is_custom_model = is_custom_model

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
  def is_custom_model(self) -> bool:
    return self._is_custom_model

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
