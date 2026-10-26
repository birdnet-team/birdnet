from __future__ import annotations

import os
from abc import abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from tqdm import tqdm

from birdnet.core.base import ResultBase
from birdnet.utils.helper import (
  apply_speed_to_duration,
  get_hop_duration_s,
  get_lossless_float_dtype,
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
      model_precision: Precision of the model ("int8", "fp16" or "fp32").
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

    # Scalar config: lossless dtype, since these feed hop_duration_s and any
    # rounding here accumulates across all segments.
    self._segment_duration_s = np.array(
      [segment_duration_s], dtype=get_lossless_float_dtype(segment_duration_s)
    )
    self._overlap_duration_s = np.array(
      [overlap_duration_s], dtype=get_lossless_float_dtype(overlap_duration_s)
    )
    self._speed = np.array([speed], dtype=get_lossless_float_dtype(speed))

    self._input_durations = input_durations

    self._model_fmin = np.array([model_fmin], dtype=get_uint_dtype(model_fmin))
    self._model_fmax = np.array([model_fmax], dtype=get_uint_dtype(model_fmax))
    self._model_sr = np.array([model_sr], dtype=get_uint_dtype(model_sr))

  @staticmethod
  def _inputs_from_files(files: list[Path]) -> np.ndarray:
    all_files = [str(file.absolute()) for file in files]
    max_len = max(map(len, all_files))
    return np.asarray(all_files, dtype=f"<U{max_len}")

  @staticmethod
  def _inputs_from_count(n: int) -> np.ndarray:
    return np.arange(n, dtype=get_uint_dtype(n - 1))

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

  @property
  def _time_dtype(self) -> np.dtype:
    # Force at least float32 for timing columns. The bulk _input_durations
    # array is stored in a magnitude-based dtype (float16 for files <= 2**11 s),
    # which is too coarse for accumulated i*hop products and would also produce
    # Arrow halffloat that some implementations (e.g. R) cannot read.
    return np.result_type(self._input_durations.dtype, np.float32)

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

  def _set_extra_load_data(self, data: dict[str, np.ndarray]) -> None:
    self._inputs = data[NP_INPUTS_KEY]
    self._input_durations = data[NP_INPUT_DURATIONS_KEY]
    self._segment_duration_s = data[NP_SEGMENT_DURATION_S_KEY]
    self._overlap_duration_s = data[NP_OVERLAP_DURATION_S_KEY]
    self._speed = data[NP_SPEED_KEY]
    self._model_fmin = data[NP_MODEL_FMIN_KEY]
    self._model_fmax = data[NP_MODEL_FMAX_KEY]
    self._model_sr = data[NP_MODEL_SR_KEY]

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

  def _fill_shared_columns(
    self,
    structured: np.ndarray,
    file_idx_flat: np.ndarray,
    chunk_idx_flat: np.ndarray,
  ) -> None:
    time_dtype = structured.dtype[VAR_START_TIME]
    start_times = chunk_idx_flat.astype(time_dtype) * self.hop_duration_s

    structured[VAR_START_TIME] = start_times
    structured[VAR_END_TIME] = np.minimum(
      start_times
      + apply_speed_to_duration(self._segment_duration_s[0], self._speed[0]),
      self._input_durations[file_idx_flat].astype(time_dtype),
    )
    del start_times
    structured[VAR_INPUT] = self._inputs[file_idx_flat]

  @abstractmethod
  def to_arrow_table(self) -> pa.Table: ...

  def _arrow_shared_columns(
    self, structured: np.ndarray
  ) -> dict[str, pa.Array | pa.ChunkedArray]:
    import pyarrow as pa

    arrow_arrays: dict[str, pa.Array | pa.ChunkedArray] = {}
    # An empty object column would infer `null`, which Parquet refuses in a
    # non-nullable field.
    input_type = pa.string() if structured[VAR_INPUT].dtype == object else None
    arrow_arrays[VAR_INPUT] = pa.array(
      structured[VAR_INPUT], type=input_type
    ).dictionary_encode()
    arrow_arrays[VAR_START_TIME] = pa.array(
      structured[VAR_START_TIME],
      type=pa.from_numpy_dtype(structured[VAR_START_TIME].dtype),
    )
    arrow_arrays[VAR_END_TIME] = pa.array(
      structured[VAR_END_TIME],
      type=pa.from_numpy_dtype(structured[VAR_END_TIME].dtype),
    )
    return arrow_arrays

  @staticmethod
  def _arrow_fields(
    arrays: Mapping[str, pa.Array | pa.ChunkedArray],
  ) -> list[pa.Field]:
    import pyarrow as pa

    return [
      pa.field(name, array.type, nullable=False) for name, array in arrays.items()
    ]

  @abstractmethod
  def to_csv(
    self,
    path: os.PathLike | str,
    *,
    encoding: str = "utf-8",
    buffer_size_kb: int = 1024,
    silent: bool = False,
  ) -> None: ...

  @abstractmethod
  def _format_csv_record(self, record: np.void) -> str: ...

  def _write_csv(
    self,
    path: os.PathLike | str,
    *,
    header: str,
    unit: str,
    encoding: str,
    buffer_size_kb: int,
    silent: bool,
  ) -> None:
    if not silent:
      print("Preparing CSV export...")  # noqa: T201

    structured = self.to_structured_array()

    buffer_bytes = buffer_size_kb * 1024
    output_path = Path(path)

    if output_path.suffix != ".csv":
      raise ValueError("Output path must have a .csv suffix")

    with output_path.open("w", encoding=encoding, buffering=buffer_bytes) as f:
      f.write(f"{header}\n")

      block: list[str] = []
      block_size_bytes = 0
      total_size_bytes = 0
      collected_size_bytes = 0
      update_size_every = 1024**2 * 100

      with tqdm(
        total=len(structured),
        desc="Writing CSV",
        unit=unit,
        disable=silent,
      ) as pbar:
        for record in structured:
          line = self._format_csv_record(record)

          block.append(line)
          block_size_bytes += len(line.encode(encoding))

          if block_size_bytes >= buffer_bytes:
            f.writelines(block)
            pbar.update(len(block))
            block.clear()
            collected_size_bytes += block_size_bytes
            block_size_bytes = 0

            if collected_size_bytes >= update_size_every:
              total_size_bytes += collected_size_bytes
              collected_size_bytes = 0

              if not silent:
                pbar.set_postfix({"CSV": f"{total_size_bytes / 1024**2:.0f} MB"})

        if block:
          f.writelines(block)
          pbar.update(len(block))

        if len(structured) > 0 and not silent:
          total_size_bytes += collected_size_bytes
          pbar.set_postfix({"CSV": f"{total_size_bytes / 1024**2:.0f} MB"})

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
