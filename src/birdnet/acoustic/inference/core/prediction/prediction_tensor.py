from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike

from birdnet.acoustic.inference.core.tensor import (
  AcousticArrayTensorBase,
  AcousticTensorBase,
)
from birdnet.utils.helper import UIntCType, get_uint_dtype


class AcousticPredictionTensor(AcousticArrayTensorBase):
  _ARRAYS = (("_species_ids", 0), ("_species_probs", 0.0), ("_species_masked", True))

  _species_ids: np.ndarray
  _species_probs: np.ndarray
  _species_masked: np.ndarray

  def __init__(
    self,
    session_id: str,
    n_inputs: int,
    top_k: int,
    n_species: int,
    half_precision: bool,
    input_indices_dtype: DTypeLike,
    segment_indices_dtype: DTypeLike,
    max_segment_index: UIntCType,  # TODO: track max_n_segments
  ) -> None:
    species_probs_dtype = np.float16 if half_precision else np.float32
    super().__init__(
      session_id,
      n_inputs,
      input_indices_dtype,
      segment_indices_dtype,
      max_segment_index,
      array_specs=(
        ("_species_ids", top_k, get_uint_dtype(max(0, n_species - 1))),
        ("_species_probs", top_k, species_probs_dtype),
        ("_species_masked", top_k, bool),
      ),
    )

  def write_block(
    self,
    input_indices: np.ndarray,
    segment_indices: np.ndarray,
    top_k_species: np.ndarray,  # 2dim
    top_k_scores: np.ndarray,  # 2dim
    top_k_mask: np.ndarray,  # 2dim
  ) -> None:
    assert top_k_species.dtype == self._species_ids.dtype
    assert top_k_scores.dtype == self._species_probs.dtype
    assert top_k_mask.dtype == self._species_masked.dtype
    self._prepare_block(input_indices, segment_indices)
    self._species_ids[input_indices, segment_indices] = top_k_species
    self._species_probs[input_indices, segment_indices] = top_k_scores
    self._species_masked[input_indices, segment_indices] = top_k_mask


class PrebuiltPredictionTensor(AcousticPredictionTensor):
  """Minimal tensor holder wrapping already-materialised per-file arrays.

  Used to build a single-file ``AcousticFilePredictionResult`` from the slice
  copied out of the shared result tensor, without re-running any inference.
  """

  def __init__(
    self,
    species_ids: np.ndarray,
    species_probs: np.ndarray,
    species_masked: np.ndarray,
  ) -> None:
    AcousticTensorBase.__init__(self)
    self._species_ids = species_ids
    self._species_probs = species_probs
    self._species_masked = species_masked

  def write_block(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
    raise NotImplementedError("PrebuiltPredictionTensor is read-only.")
