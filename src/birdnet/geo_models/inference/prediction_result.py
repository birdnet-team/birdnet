import os
from pathlib import Path
from typing import Self, final

import numpy as np
from numpy.typing import DTypeLike
from ordered_set import OrderedSet


class PredictionResult:
  def __init__(
    self,
    species_masked: np.ndarray,
    species_ids: np.ndarray,
    species_probs: np.ndarray,
    species_list: OrderedSet[str],
  ) -> None:
    assert species_ids.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
    assert species_probs.dtype in (np.float16, np.float32)
    assert species_masked.dtype == bool
    assert (
      species_masked.shape
      == species_ids.shape
      == species_probs.shape
      == (len(species_list),)
    )

    # Direkte String-Konvertierung ohne Zwischenlisten
    max_len = max(map(len, species_list))
    self._species_list = np.array(list(species_list), dtype=f"<U{max_len}")
    self._species_probs = species_probs
    self._species_ids = species_ids
    self._species_masked = species_masked

  @property
  def memory_size_mb(self) -> float:
    return (
      self._species_ids.nbytes
      + self._species_probs.nbytes
      + self._species_masked.nbytes
      + self._species_list.nbytes
    ) / 1024**2

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

  def save(self, npz_out_path: os.PathLike | str, /, *, compress: bool = True) -> None:
    npz_out_path = Path(npz_out_path)
    if npz_out_path.suffix != ".npz":
      raise ValueError("Output path must have a .npz suffix")

    save_method = np.savez_compressed if compress else np.savez

    save_method(
      npz_out_path,
      species_ids=self._species_ids,
      species_probs=self._species_probs,
      species_masked=self._species_masked,
      species_list=self._species_list,
    )

  @classmethod
  def load(cls, path: os.PathLike | str) -> Self:
    result = cls.__new__(cls)
    with np.load(path, allow_pickle=True) as npz:
      data = {k: npz[k] for k in npz.files}

    result._species_ids = data["species_ids"]
    result._species_probs = data["species_probs"]
    result._species_masked = data["species_masked"]
    result._species_list = data["species_list"]
    return result
