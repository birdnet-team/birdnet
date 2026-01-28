from __future__ import annotations

from pathlib import Path

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
from birdnet.utils.helper import (
  apply_speed_to_duration,
  get_hop_duration_s,
  get_uint_dtype,
)

VAR_EMBEDDING = "embedding"

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
  def memory_size_mb(self) -> float:
    return super().memory_size_mb + (
      (
        self._embeddings.nbytes
        + self._embeddings_masked.nbytes
        + self._unprocessable_inputs.nbytes
      )
      / 1024**2
    )

  @property
  def embeddings(self) -> np.ndarray:
    return self._embeddings

  @property
  def embeddings_masked(self) -> np.ndarray:
    return self._embeddings_masked

  @property
  def emd_dim(self) -> int:
    return self._embeddings.shape[-1]

  @property
  def max_n_segments(self) -> int:
    return self._embeddings.shape[1]

  def test() -> None:
    pass

  def to_structured_array(self) -> np.ndarray:
    valid_mask_per_segment = ~(self._embeddings_masked).all(axis=2)
    valid_file_idx, valid_seg_idx = np.where(valid_mask_per_segment)
    n_embeddings = len(valid_file_idx)

    embeddings_selected = self.embeddings[valid_file_idx, valid_seg_idx]

    dtype = [
      (VAR_INPUT, self._input_dtype),
      (VAR_START_TIME, self._input_durations.dtype),
      (VAR_END_TIME, self._input_durations.dtype),
      (VAR_EMBEDDING, self._embeddings.dtype, self.emd_dim),
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

    structured_array[VAR_EMBEDDING] = emb_flat
    del emb_flat

    return structured_array

  def unprocessable_inputs(self) -> np.ndarray:
    return self._unprocessable_inputs

  def _get_extra_save_data(self) -> dict[str, np.ndarray]:
    return super()._get_extra_save_data() | {
      NP_EMB_KEY: self._embeddings,
      NP_EMB_MASKED_KEY: self._embeddings_masked,
      NP_UNPROCESSABLE_INPUTS_KEY: self._unprocessable_inputs,
    }

  @classmethod
  def _set_extra_load_data(cls, data: dict[str, np.ndarray]) -> None:
    super()._set_extra_load_data(data)
    cls._embeddings = data[NP_EMB_KEY]
    cls._embeddings_masked = data[NP_EMB_MASKED_KEY]
    cls._unprocessable_inputs = data[NP_UNPROCESSABLE_INPUTS_KEY]


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
    all_files = [str(file.absolute()) for file in files]
    max_len = max(map(len, all_files))
    inputs = np.asarray(all_files, dtype=f"<U{max_len}")

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
    # NOTE: use dtype object for paths and species because these strings repeat often
    # -> pointer to python string is more efficient
    return object

  def _format_input_for_csv(self, input_value: str) -> str:
    return f'"{input_value}"'


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
    n_arrays = len(input_durations)
    array_indices = np.arange(n_arrays, dtype=get_uint_dtype(n_arrays - 1))

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
