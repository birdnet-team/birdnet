from __future__ import annotations  # seit Py 3.7, ab Py 3.11 Standard

import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import numpy as np  # alles, was du ohnehin brauchst
from ordered_set import OrderedSet
from tqdm import tqdm

from birdnet.acoustic_models.inference.scores.tensor import ScoresTensor
from birdnet.base import PredictionResultBase
from birdnet.helper import get_float_dtype

if TYPE_CHECKING:
  import pandas as pd
  import pyarrow as pa

VAR_FILE_PATH = "file_path"
VAR_START_TIME = "start_time"
VAR_END_TIME = "end_time"
VAR_SPECIES_NAME = "species_name"
VAR_CONFIDENCE = "confidence"


class PredictionResult(PredictionResultBase):
  def __init__(
    self,
    tensor: ScoresTensor,
    files: OrderedSet[Path],
    species_list: OrderedSet[str],
    file_durations: np.ndarray,
    segment_duration_s: int | float,
    overlap_duration_s: int | float,
  ) -> None:
    assert file_durations.dtype in (np.float16, np.float32, np.float64)
    assert tensor._species_ids.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
    assert tensor._species_probs.dtype in (np.float16, np.float32)
    assert tensor._species_masked.dtype == bool

    # Direkte String-Konvertierung ohne Zwischenlisten
    all_files = [str(file.absolute()) for file in files]
    max_len = max(map(len, all_files))
    self._files = np.asarray(all_files, dtype=f"<U{max_len}")
    self._segment_duration_s = np.array(
      [segment_duration_s], dtype=get_float_dtype(segment_duration_s)
    )
    self._overlap_duration_s = np.array(
      [overlap_duration_s], dtype=get_float_dtype(overlap_duration_s)
    )

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
    # NOTE: use dtype object for paths and species because these strings repeat often -> pointer to python string is more efficient
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

    arrow_arrays = {}
    arrow_arrays[VAR_FILE_PATH] = pa.array(
      structured[VAR_FILE_PATH]
    ).dictionary_encode()
    arrow_arrays[VAR_START_TIME] = pa.array(
      structured[VAR_START_TIME],
      type=pa.from_numpy_dtype(structured[VAR_START_TIME].dtype),
    )
    arrow_arrays[VAR_END_TIME] = pa.array(
      structured[VAR_END_TIME], type=pa.from_numpy_dtype(structured[VAR_END_TIME].dtype)
    )
    arrow_arrays[VAR_SPECIES_NAME] = pa.array(
      structured[VAR_SPECIES_NAME]
    ).dictionary_encode()
    arrow_arrays[VAR_CONFIDENCE] = pa.array(
      structured[VAR_CONFIDENCE],
      type=pa.from_numpy_dtype(structured[VAR_CONFIDENCE].dtype),
    )

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

    output_path = Path(path)
    if output_path.suffix != ".csv":
      raise ValueError("Output path must have a .csv suffix")

    with output_path.open("w", encoding=encoding, buffering=buffer_bytes) as f:
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
          if collected_size_bytes >= update_size_every or pbar.n == pbar.total:
            total_size_bytes += collected_size_bytes
            collected_size_bytes = 0
            if not silent:
              pbar.set_postfix({"CSV": f"{total_size_bytes / 1024**2:.0f} MB"})

        # Final flush
        if block:
          f.writelines(block)

  def to_dataframe(self) -> pd.DataFrame:
    import pandas as pd

    df = pd.DataFrame(self.to_structured_array(), copy=True)
    return df

  def to_parquet(
    self,
    path: os.PathLike | str,
    *,
    compression: Literal["none", "snappy", "gzip", "brotli", "lz4", "zstd"] = "snappy",
    compression_level: int | None = None,
    silent: bool = False,
  ) -> None:
    import pyarrow.parquet as pq

    path = Path(path)
    if path.suffix != ".parquet":
      raise ValueError("Output path must have a .parquet suffix")

    if not silent:
      print("Creating Arrow table...")

    table = self.to_arrow_table()

    if not silent:
      print(f"Writing Parquet to {path.absolute()} ...")

    pq.write_table(
      table,
      path,
      compression=compression,
      compression_level=compression_level,
    )

    if not silent:
      file_size = path.stat().st_size / 1024**2
      original_size = table.nbytes / 1024**2
      compression_ratio = original_size / file_size if file_size > 0 else 0
      print(f"Parquet file: {file_size:.1f} MB (compression: {compression_ratio:.1f}x)")


def hms_centis_fast(v: float) -> str:
  h, rem = divmod(v, 3600)
  m, s = divmod(rem, 60)
  result = f"{int(h):02}:{int(m):02}:{s:05.2f}"
  return result


def assert_species_masked_pattern(species_masked: np.ndarray) -> None:
  """
  Assert that species_masked has False entries first, then only True entries.
  For 3D arrays (n_files, n_segments, top_k), checks the pattern along the top_k axis
  for each (file, segment) combination.
  """
  if species_masked.size == 0:
    return

  assert species_masked.ndim == 3

  # Reshape to (n_files * n_segments, top_k) for vectorized processing
  n_files, n_segments, top_k = species_masked.shape
  reshaped = species_masked.reshape(-1, top_k)

  # For each row (file, segment combination), find first True
  # Using argmax on the mask gives us the first True position
  # If no True exists, argmax returns 0, but we handle this separately
  has_true = np.any(reshaped, axis=1)
  first_true_pos = np.argmax(reshaped, axis=1)

  # Only check rows that have at least one True
  if np.any(has_true):
    valid_rows = np.where(has_true)[0]

    for row_idx in valid_rows:
      row = reshaped[row_idx]
      first_true = first_true_pos[row_idx]

      # Quick check: all before first_true should be False, all after should be True
      if not (np.all(~row[:first_true]) and np.all(row[first_true:])):
        file_idx, seg_idx = divmod(row_idx, n_segments)
        raise AssertionError(
          f"Invalid mask pattern at file {file_idx}, segment {seg_idx}: "
          f"expected False...False,True...True pattern"
        )
