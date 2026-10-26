from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ordered_set import OrderedSet

from birdnet.acoustic.inference.core.prediction.prediction_tensor import (
  AcousticPredictionTensor,
)
from birdnet.acoustic.inference.core.result_base import (
  VAR_END_TIME,
  VAR_INPUT,
  VAR_START_TIME,
  AcousticResultBase,
)
from birdnet.utils.helper import format_input_for_csv, hms_centis_fast

if TYPE_CHECKING:
  import pyarrow as pa

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
    species_list_array: np.ndarray | None = None,
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
    species_ids, species_probs, species_masked = tensor.arrays()
    assert species_ids.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
    assert species_probs.dtype in (np.float16, np.float32)
    assert species_masked.dtype == bool
    assert tensor.unprocessable_inputs.dtype in (
      np.uint8,
      np.uint16,
      np.uint32,
      np.uint64,
    )

    # A caller may pass a prebuilt species-name array to avoid rebuilding it per
    # result (e.g. the per-file completion path constructs one result per file
    # from the same species list). The array is immutable and identical for all
    # results of a run.
    if species_list_array is not None:
      self._species_list = species_list_array
    else:
      max_len = max(map(len, species_list))
      self._species_list = np.array(list(species_list), dtype=f"<U{max_len}")
    self._species_probs = species_probs
    self._species_ids = species_ids
    self._species_masked = species_masked
    self._unprocessable_inputs = tensor.unprocessable_inputs

  @property
  def memory_size_MiB(self) -> float:
    return super().memory_size_MiB + (
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
    return int(self._species_ids.shape[1])

  @property
  def top_k(self) -> int:
    return int(self._species_ids.shape[2])

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

  def _set_extra_load_data(self, data: dict[str, np.ndarray]) -> None:
    super()._set_extra_load_data(data)
    self._species_ids = data[NP_SPECIES_IDS_KEY]
    self._species_probs = data[NP_SPECIES_PROBS_KEY]
    self._species_masked = data[NP_SPECIES_MASKED_KEY]
    self._species_list = data[NP_SPECIES_LIST_KEY]
    self._unprocessable_inputs = data[NP_UNPROCESSABLE_INPUTS_KEY]

  def to_structured_array(self) -> np.ndarray:
    valid_mask = ~self._species_masked
    valid_indices = np.where(valid_mask)
    del valid_mask

    n_predictions = len(valid_indices[0])
    time_dtype = self._time_dtype
    dtype = [
      (VAR_INPUT, self._input_dtype),
      (VAR_START_TIME, time_dtype),
      (VAR_END_TIME, time_dtype),
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

    self._fill_shared_columns(structured_array, file_idx_flat, chunk_idx_flat)
    del file_idx_flat
    del chunk_idx_flat
    structured_array[VAR_SPECIES_NAME] = self._species_list[
      self._species_ids[valid_indices]
    ]
    structured_array[VAR_CONFIDENCE] = self._species_probs[valid_indices]
    del valid_indices

    return structured_array

  def to_arrow_table(self) -> pa.Table:
    import pyarrow as pa

    structured = self.to_structured_array()

    arrow_arrays = self._arrow_shared_columns(structured)
    arrow_arrays[VAR_SPECIES_NAME] = pa.array(
      structured[VAR_SPECIES_NAME], type=pa.string()
    ).dictionary_encode()
    arrow_arrays[VAR_CONFIDENCE] = pa.array(
      structured[VAR_CONFIDENCE],
      type=pa.from_numpy_dtype(structured[VAR_CONFIDENCE].dtype),
    )

    fields = self._arrow_fields(arrow_arrays)

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
    table = pa.table(list(arrow_arrays.values()), schema=schema_with_metadata)
    return table

  def to_csv(
    self,
    path: os.PathLike | str,
    *,
    encoding: str = "utf-8",
    buffer_size_kb: int = 1024,
    silent: bool = False,
  ) -> None:
    self._write_csv(
      path,
      header=(
        f"{VAR_INPUT},{VAR_START_TIME},{VAR_END_TIME},"
        f"{VAR_SPECIES_NAME},{VAR_CONFIDENCE}"
      ),
      unit="predictions",
      encoding=encoding,
      buffer_size_kb=buffer_size_kb,
      silent=silent,
    )

  def _format_csv_record(self, record: np.void) -> str:
    return (
      f"{format_input_for_csv(record[VAR_INPUT])},"
      f'"{hms_centis_fast(record[VAR_START_TIME])}",'
      f'"{hms_centis_fast(record[VAR_END_TIME])}",'
      f"{format_input_for_csv(record[VAR_SPECIES_NAME])},"
      f"{record[VAR_CONFIDENCE]:.6f}\n"
    )


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
    species_list_array: np.ndarray | None = None,
  ) -> None:
    inputs = self._inputs_from_files(files)

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
      species_list_array=species_list_array,
    )

  def get_unprocessed_files(self) -> set[Path]:
    inputs = self._inputs[self._unprocessable_inputs]
    inputs_paths = {Path(input_str) for input_str in inputs}
    return inputs_paths

  @property
  def _input_dtype(self) -> type:
    # object dtype: repeated path/species strings share one Python object.
    return object


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
    array_indices = self._inputs_from_count(len(input_durations))

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
