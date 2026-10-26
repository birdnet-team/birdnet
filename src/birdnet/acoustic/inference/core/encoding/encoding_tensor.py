from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike

from birdnet.acoustic.inference.core.tensor import (
  AcousticArrayTensorBase,
  AcousticTensorBase,
)
from birdnet.utils.helper import UIntCType


class AcousticEncodingTensor(AcousticArrayTensorBase):
  _ARRAYS = (("_emb", 0), ("_emb_masked", True))

  _emb: np.ndarray
  _emb_masked: np.ndarray

  def __init__(
    self,
    session_id: str,
    n_inputs: int,
    emb_dim: int,
    half_precision: bool,
    input_indices_dtype: DTypeLike,
    segment_indices_dtype: DTypeLike,
    max_segment_index: UIntCType,
  ) -> None:
    emb_dtype = np.float16 if half_precision else np.float32
    super().__init__(
      session_id,
      n_inputs,
      input_indices_dtype,
      segment_indices_dtype,
      max_segment_index,
      array_specs=(("_emb", emb_dim, emb_dtype), ("_emb_masked", emb_dim, bool)),
    )

  def write_block(
    self,
    file_indices: np.ndarray,
    segment_indices: np.ndarray,
    emb: np.ndarray,  # 2dim
  ) -> None:
    assert emb.dtype == self._emb.dtype
    self._prepare_block(file_indices, segment_indices)
    self._emb[file_indices, segment_indices] = emb
    self._emb_masked[file_indices, segment_indices] = False


class PrebuiltEncodingTensor(AcousticEncodingTensor):
  """Minimal tensor holder wrapping already-materialised per-file embeddings.

  Used to build a single-file ``AcousticFileEncodingResult`` from the slice
  copied out of the shared result tensor, without re-running any inference.
  """

  def __init__(self, emb: np.ndarray, emb_masked: np.ndarray) -> None:
    AcousticTensorBase.__init__(self)
    self._emb = emb
    self._emb_masked = emb_masked

  def write_block(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
    raise NotImplementedError("PrebuiltEncodingTensor is read-only.")
