from __future__ import annotations  # seit Py 3.7, ab Py 3.11 Standard

import csv
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np  # alles, was du ohnehin brauchst
from ordered_set import OrderedSet
from tqdm import tqdm

from birdnet.acoustic_models.inference.species_tensor import SpeciesTensor
from birdnet.globals import PKG_NAME
from birdnet.helper import get_max_n_segments_array
from birdnet.local_data import get_package_version

if TYPE_CHECKING:
  import pandas as pd
  import pyarrow as pa

VAR_FILE_PATH = "file_path"
VAR_START_TIME = "start_time"
VAR_END_TIME = "end_time"
VAR_SPECIES_NAME = "species_name"
VAR_CONFIDENCE = "confidence"


class PredictionResult:
  def __init__(
    self,
    tensor: SpeciesTensor,
    files: OrderedSet[Path],
    species_list: OrderedSet[str],
    file_durations: np.ndarray,
    segment_duration_s: int | float,
    overlap_duration_s: int | float,
  ) -> None:
    assert file_durations.dtype == np.float16
    assert tensor._species_ids.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
    assert tensor._species_probs.dtype in (np.float16, np.float32)
    assert tensor._species_masked.dtype == bool

    # Direkte String-Konvertierung ohne Zwischenlisten
    all_files = [str(file.absolute()) for file in files]
    max_len = max(map(len, all_files))
    self._files = np.asarray(all_files, dtype=f"<U{max_len}")
    self._segment_duration_s = np.float16(segment_duration_s)
    self._overlap_duration_s = np.float16(overlap_duration_s)

    max_len = max(map(len, species_list))
    self._species_list = np.array(list(species_list), dtype=f"<U{max_len}")
    self._species_probs = tensor._species_probs
    self._species_ids = tensor._species_ids
    self._species_masked = tensor._species_masked
    self._file_durations = file_durations

  @property
  def memory_size_mb(self) -> float:
    return (
      self._species_ids.nbytes
      + self._species_probs.nbytes
      + self._species_masked.nbytes
      + self._files.nbytes
      + self._segment_duration_s.nbytes
      + self._overlap_duration_s.nbytes
      + self._species_list.nbytes
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
  def files(self) -> np.ndarray:
    return self._files

  @property
  def n_files(self) -> int:
    return len(self._files)

  @property
  def n_species(self) -> int:
    return len(self._species_list)

  @property
  def max_n_segments(self) -> int:
    return self._species_ids.shape[1]

  @property
  def top_k(self) -> int:
    return self._species_ids.shape[2]

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
      files=self._files,
      segment_duration_s=self._segment_duration_s,
      overlap_duration_s=self._overlap_duration_s,
      species_list=self._species_list,
      file_durations=self._file_durations,
    )

  @classmethod
  def load(cls, path: os.PathLike | str) -> Self:
    result = cls.__new__(cls)
    with np.load(path, allow_pickle=True) as npz:
      data = {k: npz[k] for k in npz.files}

    result._species_ids = data["species_ids"]
    result._species_probs = data["species_probs"]
    result._species_masked = data["species_masked"]
    result._files = data["files"]
    result._segment_duration_s = data["segment_duration_s"]
    result._overlap_duration_s = data["overlap_duration_s"]
    result._species_list = data["species_list"]
    result._file_durations = data["file_durations"]
    return result

  def to_structured_array(self) -> np.ndarray:
    valid_mask = ~self._species_masked
    valid_indices = np.where(valid_mask)
    del valid_mask

    n_predictions = len(valid_indices[0])
    # NOTE: use object for paths and species because strings repeat often -> pointer is more efficient
    dtype = [
      (VAR_FILE_PATH, object),
      (VAR_START_TIME, self._file_durations.dtype),
      (VAR_END_TIME, self._file_durations.dtype),
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

    hop_duration = self._segment_duration_s - self._overlap_duration_s
    start_times = chunk_idx_flat.astype(self._file_durations.dtype) * hop_duration
    del hop_duration
    del chunk_idx_flat

    structured_array[VAR_START_TIME] = start_times
    structured_array[VAR_END_TIME] = np.minimum(
      start_times + self._segment_duration_s, self._file_durations[file_idx_flat]
    )
    del start_times
    structured_array[VAR_FILE_PATH] = self._files[file_idx_flat]
    del file_idx_flat
    structured_array[VAR_SPECIES_NAME] = self._species_list[
      self._species_ids[valid_indices]
    ]
    structured_array[VAR_CONFIDENCE] = self._species_probs[valid_indices]
    del valid_indices

    return structured_array

  def to_arrow_table(self) -> pa.Table:
    import pyarrow as pa

    structured = self.to_structured_array()

    file_paths = structured[VAR_FILE_PATH]
    start_times = structured[VAR_START_TIME]
    end_times = structured[VAR_END_TIME]
    species_names = structured[VAR_SPECIES_NAME]
    confidences = structured[VAR_CONFIDENCE]

    arrow_arrays = {
      VAR_FILE_PATH: pa.array(file_paths).dictionary_encode(),
      VAR_START_TIME: pa.array(
        start_times, type=pa.from_numpy_dtype(start_times.dtype)
      ),
      VAR_END_TIME: pa.array(end_times, type=pa.from_numpy_dtype(end_times.dtype)),
      VAR_SPECIES_NAME: pa.array(species_names).dictionary_encode(),
      VAR_CONFIDENCE: pa.array(
        confidences, type=pa.from_numpy_dtype(confidences.dtype)
      ),
    }

    fields = [
      pa.field(VAR_FILE_PATH, arrow_arrays[VAR_FILE_PATH].type, nullable=False),
      pa.field(VAR_START_TIME, arrow_arrays[VAR_START_TIME].type, nullable=False),
      pa.field(VAR_END_TIME, arrow_arrays[VAR_END_TIME].type, nullable=False),
      pa.field(VAR_SPECIES_NAME, arrow_arrays[VAR_SPECIES_NAME].type, nullable=False),
      pa.field(VAR_CONFIDENCE, arrow_arrays[VAR_CONFIDENCE].type, nullable=False),
    ]

    metadata: dict[bytes | str, bytes | str] | None = {
      "segment_duration_s": str(self._segment_duration_s),
      "overlap_duration_s": str(self._overlap_duration_s),
      "n_files": str(self.n_files),
      "n_species": str(self.n_species),
    }

    schema_with_metadata = pa.schema(fields, metadata=metadata)
    table = pa.table(arrow_arrays, schema=schema_with_metadata)

    return table

  def to_csv(
    self,
    path: os.PathLike | str,
    *,
    encoding: str = "utf-8",
    buffer_size_kb: int = 1024,
    silent: bool = False,
  ) -> None:
    if not silent:
      print("Preparing CSV export...")

    structured = self.to_structured_array()

    buffer_bytes = buffer_size_kb * 1024

    with Path(path).open("w", encoding=encoding, buffering=buffer_bytes) as f:
      # Header
      f.write(
        f"{VAR_FILE_PATH},{VAR_START_TIME},{VAR_END_TIME},{VAR_SPECIES_NAME},{VAR_CONFIDENCE}\n"
      )

      block = []
      block_size_bytes = 0
      total_size_bytes = 0
      collected_size_bytes = 0
      update_size_every = 1024**2 * 100  # Update every 100 MB

      with tqdm(
        total=len(structured),
        desc="Writing CSV",
        unit="predictions",
        disable=silent,
      ) as pbar:
        for record in structured:
          line = f'"{record[VAR_FILE_PATH]}","{hms_centis_fast(record[VAR_START_TIME])}","{hms_centis_fast(record[VAR_END_TIME])}","{record[VAR_SPECIES_NAME]}",{record[VAR_CONFIDENCE]:.6f}\n'

          block.append(line)
          block_size_bytes += len(line.encode(encoding))

          # Gepufferte I/O
          if block_size_bytes >= buffer_bytes:
            f.writelines(block)
            block.clear()
            collected_size_bytes += block_size_bytes
            block_size_bytes = 0

          pbar.update(1)
          # show file size in GB after every GB of data written
          if collected_size_bytes >= update_size_every:
            total_size_bytes += collected_size_bytes
            collected_size_bytes = 0
            if not silent:
              pbar.set_postfix({"CSV": f"{total_size_bytes / 1024**2:.0f} MB"})

        # Final flush
        if block:
          f.writelines(block)

  def to_dataframe(self) -> pd.DataFrame:
    return convert_tensor_to_dataframe(
      self._species_ids,
      self._species_probs,
      self._species_masked,
      self._files,
      self.segment_duration_s,
      self.overlap_duration_s,
      self._species_list,
    )


def convert_tensor_to_dataframe(
  species_ids: np.ndarray,
  species_probs: np.ndarray,
  species_masked: np.ndarray,
  files: np.ndarray,
  segment_duration_s: int | float,
  overlap_duration_s: int | float,
  species_list: np.ndarray,
  /,
  *,
  silent: bool = False,
) -> pd.DataFrame:
  import pandas as pd

  top_k = species_probs.shape[2]
  max_segments = species_probs.shape[1]
  n_files = len(files)
  segments = []
  resulting_lines = []
  for i in range(max_segments):
    start = i * segment_duration_s - (i * overlap_duration_s)
    end = start + segment_duration_s
    segments.append((start, end))
  non_masked_entry_count = np.count_nonzero(~species_masked)
  with tqdm(
    total=non_masked_entry_count,
    desc="Creating DataFrame",
    unit="segment",
    disable=silent,
  ) as pbar:
    for i in range(n_files):
      for j in range(max_segments):
        spec_ids = species_ids[i, j]
        spec_probs = species_probs[i, j]
        valid = ~species_masked[i, j]
        for k in range(top_k):
          if valid[k]:
            species_id = spec_ids[k]
            species_name: str = species_list[species_id]
            scientific_name = species_name
            common_name = ""
            if "_" in species_name:
              parts = species_name.split("_", 1)
              scientific_name = parts[0]
              common_name = parts[1]

            start_sec = int(segments[j][0])
            end_sec = int(segments[j][1])

            row = {
              "file": files[i],
              "start": time.strftime("%H:%M:%S", time.gmtime(start_sec)),
              "end": time.strftime("%H:%M:%S", time.gmtime(end_sec)),
              "scientific_name": scientific_name,
              "common_name": common_name,
              VAR_CONFIDENCE: spec_probs[k],
            }
            resulting_lines.append(row)
            pbar.update(1)
          else:
            break
  df = pd.DataFrame.from_records(resulting_lines)

  if len(df.index) > 0:
    # sorting with float16 is not supported by pandas DataFrame
    df["_confidence32"] = df[VAR_CONFIDENCE].astype(np.float32, copy=False)
    df = (
      df.sort_values(
        by=["file", "start", "_confidence32"], ascending=[True, True, False]
      )
      .drop(columns="_confidence32")
      .reset_index(drop=True)
    )

  return df


def format_time_hms(seconds: float) -> str:
  """
  Formats a time in seconds to a byte string in the format HH:MM:SS.
  """
  result = time.strftime("%H:%M:%S", time.gmtime(seconds))
  return result


def hms_centis_fast(v: float) -> str:
  h, rem = divmod(v, 3600)
  m, s = divmod(rem, 60)  # s bleibt Float
  return f"{int(h):02}:{int(m):02}:{s:05.2f}"


def load_prediction_data(in_path: os.PathLike | str) -> dict[str, Any]:
  with np.load(Path(in_path), allow_pickle=True) as npz:
    result = {k: npz[k] for k in npz.files}
    return result
