from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from ordered_set import OrderedSet

from birdnet.base import ResultBase

VAR_SPECIES_NAME = "species_name"
VAR_CONFIDENCE = "confidence"

NP_SPECIES_IDS_KEY = "species_ids"
NP_SPECIES_PROBS_KEY = "species_probs"
NP_SPECIES_MASKED_KEY = "species_masked"
NP_SPECIES_LIST_KEY = "species_list"

NP_LATITUDE_KEY = "latitude"
NP_LONGITUDE_KEY = "longitude"
NP_WEEK_KEY = "week"

if TYPE_CHECKING:
  import pandas as pd
  import pyarrow as pa


class GeoPredictionResult(ResultBase):
  def __init__(
    self,
    model_path: Path,
    model_version: str,
    model_precision: str,
    latitude: float,
    longitude: float,
    week: int,
    species_masked: np.ndarray,
    species_ids: np.ndarray,
    species_probs: np.ndarray,
    species_list: OrderedSet[str],
  ) -> None:
    super().__init__(
      model_path=model_path,
      model_version=model_version,
      model_precision=model_precision,
    )

    assert species_ids.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
    assert species_probs.dtype in (np.float16, np.float32)
    assert species_masked.dtype == bool
    assert (
      species_masked.shape
      == species_ids.shape
      == species_probs.shape
      == (len(species_list),)
    )

    max_len = max(map(len, species_list))
    self._species_list = np.array(list(species_list), dtype=f"<U{max_len}")
    self._species_probs = species_probs
    self._species_ids = species_ids
    self._species_masked = species_masked

    self._latitude = np.array([latitude], dtype=np.float32)
    self._longitude = np.array([longitude], dtype=np.float32)

    assert week == -1 or (1 <= week <= 48)
    self._week = np.array([week], dtype=np.int8)

  @property
  def memory_size_mb(self) -> float:
    return super().memory_size_mb + (
      (
        self._species_ids.nbytes
        + self._species_probs.nbytes
        + self._species_masked.nbytes
        + self._species_list.nbytes
        + self._latitude.nbytes
        + self._longitude.nbytes
        + self._week.nbytes
      )
      / 1024**2
    )

  @property
  def latitude(self) -> int:
    return int(self._latitude[0])

  @property
  def longitude(self) -> int:
    return int(self._longitude[0])

  @property
  def week(self) -> int:
    return int(self._week[0])

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

  def _get_extra_save_data(self) -> dict[str, np.ndarray]:
    return {
      NP_SPECIES_IDS_KEY: self._species_ids,
      NP_SPECIES_PROBS_KEY: self._species_probs,
      NP_SPECIES_MASKED_KEY: self._species_masked,
      NP_SPECIES_LIST_KEY: self._species_list,
      NP_LATITUDE_KEY: self._latitude,
      NP_LONGITUDE_KEY: self._longitude,
      NP_WEEK_KEY: self._week,
    }

  @classmethod
  def _set_extra_load_data(cls, data: dict[str, np.ndarray]) -> None:
    cls._species_ids = data[NP_SPECIES_IDS_KEY]
    cls._species_probs = data[NP_SPECIES_PROBS_KEY]
    cls._species_masked = data[NP_SPECIES_MASKED_KEY]
    cls._species_list = data[NP_SPECIES_LIST_KEY]
    cls._latitude = data[NP_LATITUDE_KEY]
    cls._longitude = data[NP_LONGITUDE_KEY]
    cls._week = data[NP_WEEK_KEY]

  def to_structured_array(
    self,
    sort_by: Literal["species", "confidences"] | None = "species",
  ) -> np.ndarray:
    unmasked_indices = ~self._species_masked
    unmasked_probs = self._species_probs[unmasked_indices]
    unmasked_species = self._species_list[unmasked_indices]

    n_predictions = len(unmasked_species)

    max_len = max(map(len, unmasked_species))
    dtype = [
      (VAR_SPECIES_NAME, f"<U{max_len}"),
      (VAR_CONFIDENCE, self._species_probs.dtype),
    ]

    structured_array = np.empty(n_predictions, dtype=dtype)

    if n_predictions == 0:
      return structured_array

    if sort_by is not None:
      if sort_by == "species":
        sorted_indices = np.argsort(unmasked_species)
      elif sort_by == "confidences":
        sorted_indices = np.argsort(unmasked_probs)[::-1]
      else:
        raise ValueError("sort_by must be either None, 'species' or 'confidences'")
      unmasked_species = unmasked_species[sorted_indices]
      unmasked_probs = unmasked_probs[sorted_indices]

    structured_array[VAR_SPECIES_NAME] = unmasked_species
    structured_array[VAR_CONFIDENCE] = unmasked_probs

    return structured_array

  def to_arrow_table(
    self,
    sort_by: Literal["species", "confidences"] | None = "species",
  ) -> pa.Table:
    import pyarrow as pa

    structured = self.to_structured_array(sort_by)

    arrow_arrays = {}
    arrow_arrays[VAR_SPECIES_NAME] = pa.array(
      structured[VAR_SPECIES_NAME], type=pa.string()
    )
    arrow_arrays[VAR_CONFIDENCE] = pa.array(
      structured[VAR_CONFIDENCE],
      type=pa.from_numpy_dtype(self._species_probs.dtype),
    )

    fields = [
      pa.field(VAR_SPECIES_NAME, arrow_arrays[VAR_SPECIES_NAME].type, nullable=False),
      pa.field(VAR_CONFIDENCE, arrow_arrays[VAR_CONFIDENCE].type, nullable=False),
    ]

    metadata: dict[bytes | str, bytes | str] | None = {
      "n_species": str(self.n_species),
    }

    schema_with_metadata = pa.schema(fields, metadata=metadata)
    table = pa.table(arrow_arrays, schema=schema_with_metadata)
    return table

  def to_dataframe(
    self, sort_by: Literal["species", "confidences"] | None = "species"
  ) -> pd.DataFrame:
    import pandas as pd

    df = pd.DataFrame(self.to_structured_array(sort_by), copy=True)
    return df

  def to_set(self) -> set[str]:
    structured = self.to_structured_array(sort_by=None)
    result = set(structured[VAR_SPECIES_NAME].tolist())
    return result

  def to_txt(
    self,
    txt_out_path: os.PathLike[str] | str,
    sort_by: Literal["species", "confidences"] | None = "species",
    encoding: str = "utf8",
  ) -> None:
    txt_out_path = Path(txt_out_path)
    if txt_out_path.suffix != ".txt":
      raise ValueError("Output path must have a .txt suffix")

    structured = self.to_structured_array(sort_by)

    with txt_out_path.open("w", encoding=encoding) as f:
      f.write("\n".join(structured[VAR_SPECIES_NAME]))
      f.write("\n")

  def to_csv(
    self,
    csv_out_path: os.PathLike[str] | str,
    sort_by: Literal["species", "confidences"] | None = "species",
    encoding: str = "utf8",
  ) -> None:
    csv_out_path = Path(csv_out_path)
    if csv_out_path.suffix != ".csv":
      raise ValueError("Output path must have a .csv suffix")

    structured = self.to_structured_array(sort_by)

    with csv_out_path.open("w", encoding=encoding) as f:
      f.write(f"{VAR_SPECIES_NAME},{VAR_CONFIDENCE}\n")
      for record in structured:
        f.write(f"{record[VAR_SPECIES_NAME]},{record[VAR_CONFIDENCE]}\n")
