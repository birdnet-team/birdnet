from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from ordered_set import OrderedSet
from tqdm import tqdm

from birdnet.acoustic_models.inference.prediction.tensor import AcousticPredictionTensor
from birdnet.acoustic_models.inference.result_base import AcousticResultBase
from birdnet.helper import (
  apply_speed_to_duration,
  get_hop_duration_s,
  get_uint_dtype,
)

if TYPE_CHECKING:
  import pandas as pd
  import pyarrow as pa

VAR_INPUT = "input"
VAR_START_TIME = "start_time"
VAR_END_TIME = "end_time"
VAR_SPECIES_NAME = "species_name"
VAR_CONFIDENCE = "confidence"

NP_SPECIES_IDS_KEY = "species_ids"
NP_SPECIES_PROBS_KEY = "species_probs"
NP_SPECIES_MASKED_KEY = "species_masked"
NP_SPECIES_LIST_KEY = "species_list"
NP_UNPROCESSABLE_INPUTS_KEY = "unprocessable_inputs"


class AcousticPredictionResultBase(AcousticResultBase):
  def __init__(
    self,
    inputs: np.ndarray,
    input_durations: np.ndarray,
    model_path: Path,
    model_fmin: int,
    model_fmax: int,
    model_sr: int,
    model_precision: str,
    model_version: str,
    species_list: OrderedSet[str],
    segment_duration_s: int | float,
    overlap_duration_s: int | float,
    speed: int | float,
    tensor: AcousticPredictionTensor,
  ) -> None:
    super().__init__(
      inputs=inputs,
      input_durations=input_durations,
      segment_duration_s=segment_duration_s,
      overlap_duration_s=overlap_duration_s,
      speed=speed,
      model_path=model_path,
      model_fmin=model_fmin,
      model_fmax=model_fmax,
      model_sr=model_sr,
      model_precision=model_precision,
      model_version=model_version,
    )
    assert tensor._species_ids.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
    assert tensor._species_probs.dtype in (np.float16, np.float32)
    assert tensor._species_masked.dtype == bool
    assert tensor.unprocessable_inputs.dtype in (
      np.uint8,
      np.uint16,
      np.uint32,
      np.uint64,
    )

    max_len = max(map(len, species_list))
    self._species_list = np.array(list(species_list), dtype=f"<U{max_len}")
    self._species_probs = tensor._species_probs
    self._species_ids = tensor._species_ids
    self._species_masked = tensor._species_masked
    self._unprocessable_inputs = tensor.unprocessable_inputs

  @property
  def memory_size_mb(self) -> float:
    return super().memory_size_mb + (
      (
        self._species_ids.nbytes
        + self._species_probs.nbytes
        + self._species_masked.nbytes
        + self._species_list.nbytes
        + self._unprocessable_inputs.nbytes
      )
      / 1024**2
    )

  @property
  def species_list(self) -> np.ndarray:
    return self._species_list

  @property
  def species_ids(self) -> np.ndarray:
    return self._species_ids

  @property
  def species_probs(self) -> np.ndarray:
    return self._species_probs

  @property
  def species_masked(self) -> np.ndarray:
    return self._species_masked

  @property
  def n_species(self) -> int:
    return len(self._species_list)

  @property
  def max_n_segments(self) -> int:
    return self._species_ids.shape[1]

  @property
  def top_k(self) -> int:
    return self._species_ids.shape[2]

  @property
  def unprocessable_inputs(self) -> np.ndarray:
    return self._unprocessable_inputs

  def _get_extra_save_data(self) -> dict[str, np.ndarray]:
    return super()._get_extra_save_data() | {
      NP_SPECIES_IDS_KEY: self._species_ids,
      NP_SPECIES_PROBS_KEY: self._species_probs,
      NP_SPECIES_MASKED_KEY: self._species_masked,
      NP_SPECIES_LIST_KEY: self._species_list,
      NP_UNPROCESSABLE_INPUTS_KEY: self._unprocessable_inputs,
    }

  @classmethod
  def _set_extra_load_data(cls, data: dict[str, np.ndarray]) -> None:
    super()._set_extra_load_data(data)
    cls._species_ids = data[NP_SPECIES_IDS_KEY]
    cls._species_probs = data[NP_SPECIES_PROBS_KEY]
    cls._species_masked = data[NP_SPECIES_MASKED_KEY]
    cls._species_list = data[NP_SPECIES_LIST_KEY]
    cls._unprocessable_inputs = data[NP_UNPROCESSABLE_INPUTS_KEY]

  @property
  def _input_dtype(self) -> type:
    return self._inputs.dtype

  def to_structured_array(self) -> np.ndarray:
    valid_mask = ~self._species_masked
    valid_indices = np.where(valid_mask)
    del valid_mask

    n_predictions = len(valid_indices[0])
    dtype = [
      (VAR_INPUT, self._input_dtype),
      (VAR_START_TIME, self._input_durations.dtype),
      (VAR_END_TIME, self._input_durations.dtype),
      (VAR_SPECIES_NAME, object),
      (VAR_CONFIDENCE, self._species_probs.dtype),
    ]

    structured_array = np.empty(n_predictions, dtype=dtype)
    del dtype

    if n_predictions == 0:
      return structured_array
    del n_predictions

    file_idx_flat = valid_indices[0]
    chunk_idx_flat = valid_indices[1]
    confidences_flat = self._species_probs[valid_indices]

    sort_keys = (
      -confidences_flat,
      chunk_idx_flat,
      file_idx_flat,
    )
    sort_indices = np.lexsort(sort_keys)
    del sort_keys
    del confidences_flat

    file_idx_flat = file_idx_flat[sort_indices]
    chunk_idx_flat = chunk_idx_flat[sort_indices]
    valid_indices = (
      valid_indices[0][sort_indices],
      valid_indices[1][sort_indices],
      valid_indices[2][sort_indices],
    )
    del sort_indices

    hop_duration_s = get_hop_duration_s(
      self._segment_duration_s[0], self._overlap_duration_s[0], self._speed[0]
    )
    start_times = chunk_idx_flat.astype(self._input_durations.dtype) * hop_duration_s
    del hop_duration_s
    del chunk_idx_flat

    structured_array[VAR_START_TIME] = start_times
    structured_array[VAR_END_TIME] = np.minimum(
      start_times
      + apply_speed_to_duration(self._segment_duration_s[0], self._speed[0]),
      self._input_durations[file_idx_flat],
    )
    del start_times
    structured_array[VAR_INPUT] = self._inputs[file_idx_flat]
    del file_idx_flat
    structured_array[VAR_SPECIES_NAME] = self._species_list[
      self._species_ids[valid_indices]
    ]
    structured_array[VAR_CONFIDENCE] = self._species_probs[valid_indices]
    del valid_indices

    return structured_array

  def to_arrow_table(self) -> pa.Table:
    import pyarrow as pa

    structured = self.to_structured_array()

    arrow_arrays = {}
    arrow_arrays[VAR_INPUT] = pa.array(structured[VAR_INPUT]).dictionary_encode()
    arrow_arrays[VAR_START_TIME] = pa.array(
      structured[VAR_START_TIME],
      type=pa.from_numpy_dtype(structured[VAR_START_TIME].dtype),
    )
    arrow_arrays[VAR_END_TIME] = pa.array(
      structured[VAR_END_TIME], type=pa.from_numpy_dtype(structured[VAR_END_TIME].dtype)
    )
    arrow_arrays[VAR_SPECIES_NAME] = pa.array(
      structured[VAR_SPECIES_NAME]
    ).dictionary_encode()
    arrow_arrays[VAR_CONFIDENCE] = pa.array(
      structured[VAR_CONFIDENCE],
      type=pa.from_numpy_dtype(structured[VAR_CONFIDENCE].dtype),
    )

    fields = [
      pa.field(VAR_INPUT, arrow_arrays[VAR_INPUT].type, nullable=False),
      pa.field(VAR_START_TIME, arrow_arrays[VAR_START_TIME].type, nullable=False),
      pa.field(VAR_END_TIME, arrow_arrays[VAR_END_TIME].type, nullable=False),
      pa.field(VAR_SPECIES_NAME, arrow_arrays[VAR_SPECIES_NAME].type, nullable=False),
      pa.field(VAR_CONFIDENCE, arrow_arrays[VAR_CONFIDENCE].type, nullable=False),
    ]

    metadata: dict[bytes | str, bytes | str] | None = {
      "segment_duration_s": str(self._segment_duration_s[0]),
      "overlap_duration_s": str(self._overlap_duration_s[0]),
      "speed": str(self._speed[0]),
      "n_inputs": str(self.n_inputs),
      "n_species": str(self.n_species),
      "model_path": str(self._model_path[0]),
      "model_version": str(self._model_version[0]),
      "model_fmin": str(self._model_fmin[0]),
      "model_fmax": str(self._model_fmax[0]),
      "model_sr": str(self._model_sr[0]),
      "model_precision": str(self._model_precision[0]),
    }
    schema_with_metadata = pa.schema(fields, metadata=metadata)
    table = pa.table(arrow_arrays, schema=schema_with_metadata)
    return table

  def _format_input_for_csv(self, input_value: Any) -> str:  # noqa: ANN401
    return f'"{input_value}"'

  def to_csv(
    self,
    path: os.PathLike | str,
    *,
    encoding: str = "utf-8",
    buffer_size_kb: int = 1024,
    silent: bool = False,
  ) -> None:
    if not silent:
      print("Preparing CSV export...")

    structured = self.to_structured_array()

    buffer_bytes = buffer_size_kb * 1024

    output_path = Path(path)
    if output_path.suffix != ".csv":
      raise ValueError("Output path must have a .csv suffix")

    with output_path.open("w", encoding=encoding, buffering=buffer_bytes) as f:
      # Header
      f.write(
        f"{VAR_INPUT},{VAR_START_TIME},{VAR_END_TIME},{VAR_SPECIES_NAME},{VAR_CONFIDENCE}\n"
      )

      block = []
      block_size_bytes = 0
      total_size_bytes = 0
      collected_size_bytes = 0
      update_size_every = 1024**2 * 100  # Update every 100 MB

      with tqdm(
        total=len(structured),
        desc="Writing CSV",
        unit="predictions",
        disable=silent,
      ) as pbar:
        for record in structured:
          line = f'{self._format_input_for_csv(record[VAR_INPUT])},"{hms_centis_fast(record[VAR_START_TIME])}","{hms_centis_fast(record[VAR_END_TIME])}","{record[VAR_SPECIES_NAME]}",{record[VAR_CONFIDENCE]:.6f}\n'

          block.append(line)
          block_size_bytes += len(line.encode(encoding))

          # Gepufferte I/O
          if block_size_bytes >= buffer_bytes:
            f.writelines(block)
            block.clear()
            collected_size_bytes += block_size_bytes
            block_size_bytes = 0

          pbar.update(1)
          # show file size in GB after every GB of data written
          if collected_size_bytes >= update_size_every or pbar.n == pbar.total:
            total_size_bytes += collected_size_bytes
            collected_size_bytes = 0
            if not silent:
              pbar.set_postfix({"CSV": f"{total_size_bytes / 1024**2:.0f} MB"})

        # Final flush
        if block:
          f.writelines(block)

  def to_dataframe(self) -> pd.DataFrame:
    import pandas as pd

    df = pd.DataFrame(self.to_structured_array(), copy=True)
    return df

  def to_parquet(
    self,
    path: os.PathLike | str,
    *,
    compression: Literal["none", "snappy", "gzip", "brotli", "lz4", "zstd"] = "snappy",
    compression_level: int | None = None,
    silent: bool = False,
  ) -> None:
    import pyarrow.parquet as pq

    path = Path(path)
    if path.suffix != ".parquet":
      raise ValueError("Output path must have a .parquet suffix")

    if not silent:
      print("Creating Arrow table...")

    table = self.to_arrow_table()

    if not silent:
      print(f"Writing Parquet to {path.absolute()} ...")

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
      print(f"Parquet file: {file_size:.1f} MB (compression: {compression_ratio:.1f}x)")


class AcousticFilePredictionResult(AcousticPredictionResultBase):
  def __init__(
    self,
    tensor: AcousticPredictionTensor,
    files: list[Path],
    species_list: OrderedSet[str],
    file_durations: np.ndarray,
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
    all_files = [str(file.absolute()) for file in files]
    max_len = max(map(len, all_files))
    inputs = np.asarray(all_files, dtype=f"<U{max_len}")

    super().__init__(
      tensor=tensor,
      species_list=species_list,
      input_durations=file_durations,
      segment_duration_s=segment_duration_s,
      overlap_duration_s=overlap_duration_s,
      speed=speed,
      inputs=inputs,
      model_path=model_path,
      model_fmin=model_fmin,
      model_fmax=model_fmax,
      model_sr=model_sr,
      model_precision=model_precision,
      model_version=model_version,
    )

  def get_unprocessed_files(self) -> set[Path]:
    inputs = self._inputs[self._unprocessable_inputs]
    inputs_paths = {Path(input_str) for input_str in inputs}
    return inputs_paths

  @property
  def _input_dtype(self) -> type:
    # NOTE: use dtype object for paths and species because these strings repeat often
    # -> pointer to python string is more efficient
    return object

  def _format_input_for_csv(self, input_value: str) -> str:
    return f'"{input_value}"'


class AcousticDataPredictionResult(AcousticPredictionResultBase):
  def __init__(
    self,
    tensor: AcousticPredictionTensor,
    species_list: OrderedSet[str],
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
    n_arrays = len(input_durations)
    array_indices = np.arange(n_arrays, dtype=get_uint_dtype(n_arrays - 1))

    super().__init__(
      tensor=tensor,
      species_list=species_list,
      input_durations=input_durations,
      segment_duration_s=segment_duration_s,
      overlap_duration_s=overlap_duration_s,
      speed=speed,
      inputs=array_indices,
      model_path=model_path,
      model_fmin=model_fmin,
      model_fmax=model_fmax,
      model_sr=model_sr,
      model_precision=model_precision,
      model_version=model_version,
    )

  def _format_input_for_csv(self, input_value: Any) -> str:
    return f"{input_value}"


def hms_centis_fast(v: float) -> str:
  h, rem = divmod(v, 3600)
  m, s = divmod(rem, 60)
  result = f"{int(h):02}:{int(m):02}:{s:05.2f}"
  return result
