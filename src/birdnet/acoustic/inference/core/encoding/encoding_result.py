from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from birdnet.acoustic.inference.core.encoding.encoding_tensor import (
  AcousticEncodingTensor,
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

VAR_EMBEDDING = "embedding"

# A `list` column addresses its values with int32 offsets; a larger export is
# split into chunks, as `pa.array` splits a list of lists.
_MAX_VALUES_PER_CHUNK = np.iinfo(np.int32).max

NP_EMB_KEY = "embeddings"
NP_EMB_MASKED_KEY = "embeddings_masked"
NP_UNPROCESSABLE_INPUTS_KEY = "unprocessable_inputs"


class AcousticEncodingResultBase(AcousticResultBase):
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
    segment_duration_s: int | float,
    overlap_duration_s: int | float,
    speed: int | float,
    tensor: AcousticEncodingTensor,
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
    assert tensor._emb.dtype in (np.float16, np.float32)
    assert tensor._emb_masked.dtype == bool

    self._embeddings = tensor._emb
    self._embeddings_masked = tensor._emb_masked
    self._unprocessable_inputs = tensor.unprocessable_inputs

  @property
  def memory_size_MiB(self) -> float:
    """Return the total result memory usage including embeddings buffers.

    Returns:
      float: Memory size in mebibytes.
    """
    return super().memory_size_MiB + (
      (
        self._embeddings.nbytes
        + self._embeddings_masked.nbytes
        + self._unprocessable_inputs.nbytes
      )
      / 1024**2
    )

  @property
  def embeddings(self) -> np.ndarray:
    """Return the raw embedding tensor produced by the encoder.

    Returns:
      np.ndarray: Embeddings with shape `(n_inputs, n_segments, emb_dim)`.
    """
    return self._embeddings

  @property
  def embeddings_masked(self) -> np.ndarray:
    """Return the mask that marks relevant segments across files.

    Returns:
      np.ndarray: Boolean mask of the same shape as `embeddings`.
    """
    return self._embeddings_masked

  @property
  def emb_dim(self) -> int:
    """Return the embedding dimensionality.

    Returns:
      int: Number of coefficients per embedding vector.
    """
    return self._embeddings.shape[-1]

  @property
  def max_n_segments(self) -> int:
    """Return the maximum segment count reserved per input.

    Returns:
      int: Number of overlapping windows available per file.
    """
    return self._embeddings.shape[1]

  def to_structured_array(self) -> np.ndarray:
    """Convert the embeddings and timing metadata into a structured array.

    Returns:
      np.ndarray: Array with fields for input path, start/end times, and embedding.
    """
    valid_mask_per_segment = ~(self._embeddings_masked).all(axis=2)
    valid_file_idx, valid_seg_idx = np.where(valid_mask_per_segment)
    n_embeddings = len(valid_file_idx)

    embeddings_selected = self.embeddings[valid_file_idx, valid_seg_idx]

    time_dtype = self._time_dtype

    dtype = [
      (VAR_INPUT, self._input_dtype),
      (VAR_START_TIME, time_dtype),
      (VAR_END_TIME, time_dtype),
      (VAR_EMBEDDING, self._embeddings.dtype, self.emb_dim),
    ]

    structured_array = np.empty(n_embeddings, dtype=dtype)
    del dtype

    if n_embeddings == 0:
      return structured_array
    del n_embeddings

    sort_keys = (
      valid_seg_idx,
      valid_file_idx,
    )
    sort_indices = np.lexsort(sort_keys)
    del sort_keys

    file_idx_flat = valid_file_idx[sort_indices]
    chunk_idx_flat = valid_seg_idx[sort_indices]
    emb_flat = embeddings_selected[sort_indices]
    del embeddings_selected
    del sort_indices

    self._fill_shared_columns(structured_array, file_idx_flat, chunk_idx_flat)
    del file_idx_flat
    del chunk_idx_flat

    structured_array[VAR_EMBEDDING] = emb_flat
    del emb_flat

    return structured_array

  def to_arrow_table(self) -> pa.Table:
    """Produce a PyArrow table that serializes each embedding with timing metadata.

    Returns:
      pa.Table: Table containing dictionary-encoded inputs and embeddings lists.
    """
    import pyarrow as pa

    structured = self.to_structured_array()

    arrow_arrays = self._arrow_shared_columns(structured)
    fields = self._arrow_fields(arrow_arrays)

    embedding_element_type = pa.from_numpy_dtype(self._embeddings.dtype)
    embedding_type = pa.list_(embedding_element_type)
    arrow_arrays[VAR_EMBEDDING] = _embedding_column(structured[VAR_EMBEDDING])
    fields.append(pa.field(VAR_EMBEDDING, embedding_type, nullable=False))

    metadata: dict[bytes | str, bytes | str] = {
      "segment_duration_s": str(self._segment_duration_s[0]),
      "overlap_duration_s": str(self._overlap_duration_s[0]),
      "speed": str(self._speed[0]),
      "n_inputs": str(self.n_inputs),
      "model_path": str(self._model_path[0]),
      "model_version": str(self._model_version[0]),
      "model_fmin": str(self._model_fmin[0]),
      "model_fmax": str(self._model_fmax[0]),
      "model_sr": str(self._model_sr[0]),
      "model_precision": str(self._model_precision[0]),
      "embedding_dim": str(self.emb_dim),
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
    """Dump the structured embeddings to a CSV file for downstream analysis.

    Args:
      path: File path where the CSV will be written (must end with .csv).
      encoding: Text encoding for the output file.
      buffer_size_kb: Buffer size used when writing the file.
      silent: Suppress progress messages when True.
    """
    self._write_csv(
      path,
      header=f"{VAR_INPUT},{VAR_START_TIME},{VAR_END_TIME},{VAR_EMBEDDING}",
      unit="embeddings",
      encoding=encoding,
      buffer_size_kb=buffer_size_kb,
      silent=silent,
    )

  def _format_csv_record(self, record: np.void) -> str:
    return (
      f"{format_input_for_csv(record[VAR_INPUT])},"
      f'"{hms_centis_fast(record[VAR_START_TIME])}",'
      f'"{hms_centis_fast(record[VAR_END_TIME])}",'
      f"{_format_embedding_for_csv(record[VAR_EMBEDDING])}\n"
    )

  def unprocessable_inputs(self) -> np.ndarray:
    """Return the indices of inputs that could not be processed.

    Returns:
      np.ndarray: Boolean mask or indices for skipped inputs.
    """
    return self._unprocessable_inputs

  def _get_extra_save_data(self) -> dict[str, np.ndarray]:
    return super()._get_extra_save_data() | {
      NP_EMB_KEY: self._embeddings,
      NP_EMB_MASKED_KEY: self._embeddings_masked,
      NP_UNPROCESSABLE_INPUTS_KEY: self._unprocessable_inputs,
    }

  def _set_extra_load_data(self, data: dict[str, np.ndarray]) -> None:
    super()._set_extra_load_data(data)
    self._embeddings = data[NP_EMB_KEY]
    self._embeddings_masked = data[NP_EMB_MASKED_KEY]
    self._unprocessable_inputs = data[NP_UNPROCESSABLE_INPUTS_KEY]


class AcousticFileEncodingResult(AcousticEncodingResultBase):
  def __init__(
    self,
    tensor: AcousticEncodingTensor,
    files: list[Path],
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
    inputs = self._inputs_from_files(files)

    super().__init__(
      tensor=tensor,
      inputs=inputs,
      input_durations=file_durations,
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

  @property
  def _input_dtype(self) -> type:
    # object dtype: repeated path strings share one Python object.
    return object


class AcousticDataEncodingResult(AcousticEncodingResultBase):
  def __init__(
    self,
    tensor: AcousticEncodingTensor,
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
      inputs=array_indices,
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


def _format_embedding_for_csv(embedding: np.ndarray, decimals: int = 6) -> str:
  fmt = f"{{:.{decimals}f}}"
  # Python floats format about twice as fast as NumPy scalars.
  formatted = ",".join(fmt.format(value) for value in embedding.tolist())
  return f'"{formatted}"'


def _embedding_column(embeddings: np.ndarray) -> pa.ChunkedArray:
  import pyarrow as pa

  n_embeddings, emb_dim = embeddings.shape
  rows_per_chunk = max(1, _MAX_VALUES_PER_CHUNK // max(1, emb_dim))
  chunks: list[pa.ListArray] = []
  # An empty result still gets one (empty) chunk.
  for start in range(0, max(1, n_embeddings), rows_per_chunk):
    rows = embeddings[start : start + rows_per_chunk]
    offsets = np.arange(0, (len(rows) + 1) * emb_dim, emb_dim, dtype=np.int32)
    # One contiguous copy: the field is a strided view into the structured array.
    values = np.ascontiguousarray(rows).reshape(-1)
    # No type=: pyarrow 7 lacks it, and the default child field is already "item".
    chunks.append(
      pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), pa.array(values))
    )
  return pa.chunked_array(chunks)
