from __future__ import annotations  # seit Py 3.7, ab Py 3.11 Standard

import os
from pathlib import Path
from typing import Self

import numpy as np  # alles, was du ohnehin brauchst
from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.emb.tensor import EmbeddingsTensor
from birdnet.base import PredictionResultBase
from birdnet.helper import get_float_dtype


class EncodingResult(PredictionResultBase):
  def __init__(
    self,
    tensor: EmbeddingsTensor,
    files: OrderedSet[Path],
    file_durations: np.ndarray,
    segment_duration_s: int | float,
    overlap_duration_s: int | float,
  ) -> None:
    assert file_durations.dtype in (np.float16, np.float32, np.float64)
    assert tensor._emb.dtype in (np.float16, np.float32)
    assert tensor._emb_masked.dtype == bool

    all_files = [str(file.absolute()) for file in files]
    max_len = max(map(len, all_files))
    self._files = np.asarray(all_files, dtype=f"<U{max_len}")
    self._segment_duration_s = np.array(
      [segment_duration_s], dtype=get_float_dtype(segment_duration_s)
    )
    self._overlap_duration_s = np.array(
      [overlap_duration_s], dtype=get_float_dtype(overlap_duration_s)
    )

    self._embeddings = tensor._emb
    self._embeddings_masked = tensor._emb_masked
    self._file_durations = file_durations

  @property
  def memory_size_mb(self) -> float:
    return (
      self._embeddings.nbytes
      + self._embeddings_masked.nbytes
      + self._files.nbytes
      + self._segment_duration_s.nbytes
      + self._overlap_duration_s.nbytes
      + self._file_durations.nbytes
    ) / 1024**2

  @property
  def segment_duration_s(self) -> float:
    return float(self._segment_duration_s)

  @property
  def overlap_duration_s(self) -> float:
    return float(self._overlap_duration_s)

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
  def files(self) -> np.ndarray:
    return self._files

  @property
  def n_files(self) -> int:
    return len(self._files)

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
      files=self._files,
      segment_duration_s=self._segment_duration_s,
      overlap_duration_s=self._overlap_duration_s,
      file_durations=self._file_durations,
    )

  @classmethod
  def load(cls, path: os.PathLike | str) -> Self:
    result = cls.__new__(cls)
    with np.load(path, allow_pickle=True) as npz:
      data = {k: npz[k] for k in npz.files}

    result._embeddings = data["embeddings"]
    result._embeddings_masked = data["embeddings_masked"]
    result._files = data["files"]
    result._segment_duration_s = data["segment_duration_s"]
    result._overlap_duration_s = data["overlap_duration_s"]
    result._file_durations = data["file_durations"]
    return result
