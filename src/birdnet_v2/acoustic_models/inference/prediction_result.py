from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from ordered_set import OrderedSet

from birdnet_v2.acoustic_models.inference.species_tensor import SpeciesTensor


class PredictionResult:
  def __init__(
    self,
    tensor: SpeciesTensor,
    files: OrderedSet[Path],
    chunk_duration_s: int | float,
    overlap_duration_s: int | float,
    species_list: OrderedSet[str],
  ) -> None:
    self._tensor = tensor
    self._files = files
    self._chunk_duration_s = chunk_duration_s
    self._overlap_duration_s = overlap_duration_s
    self._species_list = species_list

  def to_dataframe(self) -> pd.DataFrame:
    return convert_tensor_to_dataframe(
      self._tensor,
      self._files,
      self._chunk_duration_s,
      self._overlap_duration_s,
      self._species_list,
    )


def convert_tensor_to_dataframe(
  tensor: SpeciesTensor,
  files: OrderedSet[Path],
  chunk_duration_s: int | float,
  overlap_duration_s: int | float,
  species_list: OrderedSet[str],
) -> pd.DataFrame:
  max_chunks = tensor.current_n_chunks
  n_files = len(files)
  chunks = []
  resulting_lines = []
  for i in range(max_chunks):
    start = i * chunk_duration_s - (i * overlap_duration_s)
    end = start + chunk_duration_s
    chunks.append((start, end))
  for i in range(n_files):
    for j in range(max_chunks):
      species_ids = tensor._species_ids[i, j]
      species_probs = tensor._species_probs[i, j]
      valid = ~tensor._species_masked[i, j]
      for k in range(tensor._top_k):
        if valid[k]:
          species_id = species_ids[k]
          species_name: str = species_list[species_id]
          scientific_name = species_name
          common_name = ""
          if "_" in species_name:
            parts = species_name.split("_", 1)
            scientific_name = parts[0]
            common_name = parts[1]

          start_sec = int(chunks[j][0])
          end_sec = int(chunks[j][1])

          row = {
            "file": str(files[i].absolute()),
            "start": time.strftime("%H:%M:%S", time.gmtime(start_sec)),
            "end": time.strftime("%H:%M:%S", time.gmtime(end_sec)),
            "scientific_name": scientific_name,
            "common_name": common_name,
            "confidence": species_probs[k],
          }
          resulting_lines.append(row)
        else:
          break
  df = pd.DataFrame.from_records(resulting_lines)

  if len(df.index) > 0:
    # sorting with float16 is not supported by pandas DataFrame
    df["_confidence32"] = df["confidence"].astype(np.float32, copy=False)
    df = (
      df.sort_values(
        by=["file", "start", "_confidence32"], ascending=[True, True, False]
      )
      .drop(columns="_confidence32")
      .reset_index(drop=True)
    )

  return df
