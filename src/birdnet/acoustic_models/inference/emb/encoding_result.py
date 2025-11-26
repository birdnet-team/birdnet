from __future__ import annotations

import os
from pathlib import Path
from typing import Self

import numpy as np

from birdnet.acoustic_models.inference.emb.tensor import EmbeddingsTensor
from birdnet.base import ResultBase
from birdnet.helper import get_float_dtype, get_uint_dtype


class EncodingResultBase(ResultBase):
  def __init__(
    self,
    tensor: EmbeddingsTensor,
    inputs: np.ndarray,
    input_durations: np.ndarray,
    segment_duration_s: int | float,
    overlap_duration_s: int | float,
    speed: int | float,
  ) -> None:
    assert input_durations.dtype in (np.float16, np.float32, np.float64)
    assert tensor._emb.dtype in (np.float16, np.float32)
    assert tensor._emb_masked.dtype == bool

    self._inputs = inputs

    self._segment_duration_s = np.array(
      [segment_duration_s], dtype=get_float_dtype(segment_duration_s)
    )
    self._overlap_duration_s = np.array(
      [overlap_duration_s], dtype=get_float_dtype(overlap_duration_s)
    )
    self._speed = np.array([speed], dtype=get_float_dtype(speed))

    self._embeddings = tensor._emb
    self._embeddings_masked = tensor._emb_masked
    self._file_durations = input_durations

  @property
  def memory_size_mb(self) -> float:
    return (
      self._embeddings.nbytes
      + self._embeddings_masked.nbytes
      + self._inputs.nbytes
      + self._segment_duration_s.nbytes
      + self._overlap_duration_s.nbytes
      + self._speed.nbytes
      + self._file_durations.nbytes
    ) / 1024**2

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
  def file_durations(self) -> np.ndarray:
    return self._file_durations

  @property
  def embeddings(self) -> np.ndarray:
    return self._embeddings

  @property
  def embeddings_masked(self) -> np.ndarray:
    return self._embeddings_masked

  @property
  def inputs(self) -> np.ndarray:
    return self._inputs

  @property
  def n_inputs(self) -> int:
    return len(self._inputs)

  @property
  def emd_dim(self) -> int:
    return self._embeddings.shape[-1]

  @property
  def max_n_segments(self) -> int:
    return self._embeddings.shape[1]

  def save(self, npz_out_path: os.PathLike | str, /, *, compress: bool = True) -> None:
    npz_out_path = Path(npz_out_path)
    if npz_out_path.suffix != ".npz":
      raise ValueError("Output path must have a .npz suffix")

    save_method = np.savez_compressed if compress else np.savez

    save_method(
      npz_out_path,
      embeddings=self._embeddings,
      embeddings_masked=self._embeddings_masked,
      inputs=self._inputs,
      segment_duration_s=self._segment_duration_s,
      overlap_duration_s=self._overlap_duration_s,
      speed=self._speed,
      file_durations=self._file_durations,
    )

  @classmethod
  def load(cls, path: os.PathLike | str) -> Self:
    result = cls.__new__(cls)
    with np.load(path, allow_pickle=True) as npz:
      data = {k: npz[k] for k in npz.files}

    result._embeddings = data["embeddings"]
    result._embeddings_masked = data["embeddings_masked"]
    result._inputs = data["inputs"]
    result._segment_duration_s = data["segment_duration_s"]
    result._overlap_duration_s = data["overlap_duration_s"]
    result._speed = data["speed"]
    result._file_durations = data["file_durations"]
    return result


class FileEncodingResult(EncodingResultBase):
  def __init__(
    self,
    tensor: EmbeddingsTensor,
    files: list[Path],
    file_durations: np.ndarray,
    segment_duration_s: int | float,
    overlap_duration_s: int | float,
    speed: int | float,
  ) -> None:
    all_files = [str(file.absolute()) for file in files]
    max_len = max(map(len, all_files))
    inputs = np.asarray(all_files, dtype=f"<U{max_len}")

    super().__init__(
      tensor,
      inputs,
      file_durations,
      segment_duration_s,
      overlap_duration_s,
      speed,
    )

  @property
  def _input_dtype(self) -> type:
    # NOTE: use dtype object for paths and species because these strings repeat often
    # -> pointer to python string is more efficient
    return object

  def _format_input_for_csv(self, input_value: str) -> str:
    return f'"{input_value}"'


class DataEncodingResult(EncodingResultBase):
  def __init__(
    self,
    tensor: EmbeddingsTensor,
    input_durations: np.ndarray,
    segment_duration_s: int | float,
    overlap_duration_s: int | float,
    speed: int | float,
  ) -> None:
    n_arrays = len(input_durations)
    array_indices = np.arange(n_arrays, dtype=get_uint_dtype(n_arrays - 1))

    super().__init__(
      tensor,
      array_indices,
      input_durations,
      segment_duration_s,
      overlap_duration_s,
      speed,
    )
