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
    data = load_prediction_data(path)

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

    n_predictions = len(valid_indices[0])
    dtype = [
      ("file_path", self._files.dtype),
      ("start_time", self._file_durations.dtype),
      ("end_time", self._file_durations.dtype),
      ("species_name", self._species_list.dtype),
      ("confidence", self._species_probs.dtype),
    ]

    structured_array = np.empty(n_predictions, dtype=dtype)

    if n_predictions == 0:
      return structured_array

    file_idx_flat = valid_indices[0]
    chunk_idx_flat = valid_indices[1]
    confidences_flat = self._species_probs[valid_indices]

    sort_keys = (
      -confidences_flat,
      chunk_idx_flat,
      file_idx_flat,
    )
    sort_indices = np.lexsort(sort_keys)

    file_idx_flat = file_idx_flat[sort_indices]
    chunk_idx_flat = chunk_idx_flat[sort_indices]
    valid_indices = (
      valid_indices[0][sort_indices],
      valid_indices[1][sort_indices],
      valid_indices[2][sort_indices],
    )

    hop_duration = self._segment_duration_s - self._overlap_duration_s
    start_times = chunk_idx_flat.astype(self._file_durations.dtype) * hop_duration
    end_times = start_times + self._segment_duration_s

    file_durations_flat = self._file_durations[file_idx_flat]
    end_times = np.minimum(end_times, file_durations_flat)

    species_ids_flat = self._species_ids[valid_indices]
    confidences_flat = self._species_probs[valid_indices]

    file_paths = self._files[file_idx_flat]
    species_names = self._species_list[species_ids_flat]

    structured_array["file_path"] = file_paths
    structured_array["start_time"] = start_times
    structured_array["end_time"] = end_times
    structured_array["species_name"] = species_names
    structured_array["confidence"] = confidences_flat

    return structured_array

  def to_arrow_table(self) -> pa.Table:
    import pyarrow as pa

    structured = self.to_structured_array()

    file_paths = structured["file_path"]
    start_times = structured["start_time"]
    end_times = structured["end_time"]
    species_names = structured["species_name"]
    confidences = structured["confidence"]

    arrow_arrays = {
      "file_path": pa.array(file_paths).dictionary_encode(),
      "start_time": pa.array(start_times, type=pa.from_numpy_dtype(start_times.dtype)),
      "end_time": pa.array(end_times, type=pa.from_numpy_dtype(end_times.dtype)),
      "species_name": pa.array(species_names).dictionary_encode(),
      "confidence": pa.array(confidences, type=pa.from_numpy_dtype(confidences.dtype)),
    }

    fields = [
      pa.field("file_path", arrow_arrays["file_path"].type, nullable=False),
      pa.field("start_time", arrow_arrays["start_time"].type, nullable=False),
      pa.field("end_time", arrow_arrays["end_time"].type, nullable=False),
      pa.field("species_name", arrow_arrays["species_name"].type, nullable=False),
      pa.field("confidence", arrow_arrays["confidence"].type, nullable=False),
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

  def to_csv(
    self, path: os.PathLike | str, /, *, encoding: str = "utf-8", silent: bool = False
  ) -> None:
    fast_save_tensor_to_csv(self, path, encoding=encoding, silent=silent)


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
              "confidence": spec_probs[k],
            }
            resulting_lines.append(row)
            pbar.update(1)
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


def fast_save_tensor_to_csv(
  result: PredictionResult,
  out_path: os.PathLike | str,
  *,
  buf_KiB: int = 256,
  encoding: str = "utf-8",
  silent: bool = False,
) -> None:
  sid = result.species_ids
  sprob = result.species_probs
  smask = ~result.species_masked
  top_k = result.species_probs.shape[2]
  n_files = len(result.files)
  n_segments = result.species_probs.shape[1]
  hop = result.segment_duration_s - result.overlap_duration_s

  segm_starts = np.arange(n_segments, dtype=np.float64) * hop
  segm_ends = segm_starts + result.segment_duration_s
  segm_starts_fmt = np.array(
    [hms_centis_fast(v).encode(encoding) for v in segm_starts],
    dtype="S11",
  )
  segm_ends_fmt = np.array(
    [hms_centis_fast(v).encode(encoding) for v in segm_ends],
    dtype="S11",
  )

  max_segments = get_max_n_segments_array(
    result.file_durations, result.segment_duration_s, result.overlap_duration_s
  )

  file_ends_fmt_files = np.array(
    [hms_centis_fast(x).encode(encoding) for x in result.file_durations],
    dtype="S11",
  )
  hdr = b"file,start,end,species,confidence\n"

  buf_bytes = buf_KiB * 1024
  block: list[bytes] = []
  block_size = 0

  with open(out_path, "wb", buffering=buf_bytes) as fh:
    fh.write(hdr)
    non_masked_entry_count = np.count_nonzero(smask)
    with tqdm(
      total=non_masked_entry_count,
      desc="Writing predictions to CSV",
      unit="segment",
      disable=silent,
    ) as pbar:
      for file_index in range(n_files):
        file_b = result.files[file_index].encode(encoding)
        species_ids = sid[file_index]  # View [segments, top_k]
        species_probs = sprob[file_index]
        species_masks = smask[file_index]
        max_segments_for_file = max_segments[file_index]
        file_end = result.file_durations[file_index]

        for segment_index in range(max_segments_for_file):
          # Bool-Maske für gültige Spezies
          valid = species_masks[segment_index]
          if not valid.any():
            continue

          start_time = segm_starts_fmt[segment_index]
          if segm_ends[segment_index] <= file_end:
            end_time = segm_ends_fmt[segment_index]
          else:
            end_time = file_ends_fmt_files[file_index]

          # Slice ohne Python-Loop
          k_lim = np.argmax(~valid, axis=0) if not valid.all() else top_k
          ids = species_ids[segment_index, :k_lim]
          confidences = species_probs[segment_index, :k_lim]

          # → Strings zusammenbauen (bytes, weil schneller)
          for k in range(k_lim):
            name: str = result.species_list[ids[k]]
            # sci = name
            # common = ""
            # if "_" in name:
            #   sci, _, common = name.partition("_")
            line = (
              file_b
              + b","
              + start_time
              + b","
              + end_time
              + b","
              + name.encode(encoding)
              + b","
              + f"{confidences[k]:.6f}".encode(encoding)
              + b"\n"
            )

            block.append(line)
            pbar.update(1)
            block_size += len(line)

            # Sobald der Puffer > ~256 KiB ist → einmalig flushen
            if block_size >= buf_bytes:
              fh.writelines(block)
              block.clear()
              block_size = 0

    # Tail flush
    if block:
      fh.writelines(block)


def save_tensor_to_csv(
  species_ids: np.ndarray,
  species_probs: np.ndarray,
  species_masked: np.ndarray,
  files: np.ndarray,
  segment_duration_s: float,
  overlap_duration_s: float,
  species_list: np.ndarray,
  out_path: os.PathLike | str,
  *,
  encoding: str = "utf-8",
  newline: str = "",
) -> None:
  """
  Stream-writes the prediction tensor directly to *out_path* (CSV).

  •  zero-copy on the tensor: we only read views (.astype is **never** called)
  •  no pandas / DataFrame → almost no extra RAM
  •  rows come out already grouped     (file ↑  →  start ↑  → confidence ↓)

  """

  # ---------– helpers (local for speed) ------------------------------------
  fmt_hms = time.strftime  # local binding (tight inner loop)
  gmtime = time.gmtime
  top_k = species_ids.shape[2]
  n_segments = species_ids.shape[1]
  n_files = len(files)
  step = segment_duration_s - overlap_duration_s  # “effective hop”

  # fast pre-compute the start/end list once
  starts = np.arange(n_segments, dtype=np.float64) * step
  ends = starts + segment_duration_s

  header = "file,start,end,scientific_name,common_name,confidence"

  # --- stream write --------------------------------------------------------
  with open(out_path, "w", encoding=encoding, newline=newline) as fh:
    writer = csv.writer(fh)
    writer.writerow(header.split(","))  # one tiny allocation

    for fi in range(n_files):
      file_str = files[fi]
      for ci in range(n_segments):
        # tensor already sorted per-segment by score ↓
        valid_mask = ~species_masked[fi, ci]
        if not valid_mask.any():
          continue  # no detections here

        start_txt = fmt_hms("%H:%M:%S", gmtime(int(starts[ci])))
        end_txt = fmt_hms("%H:%M:%S", gmtime(int(ends[ci])))

        species_ids = species_ids[fi, ci]
        species_prob = species_probs[fi, ci]

        # iterate only over valid positions (≤ top_k, early break)
        for ki in range(top_k):
          if not valid_mask[ki]:
            break

          sp_id = species_ids[ki]
          sp_name = species_list[
            sp_id
          ]  # e.g. "Poecile_atricapillus_Black-capped Chickadee"
          sci = sp_name
          common = ""
          if "_" in sp_name:
            sci, _, common = sp_name.partition("_")

          writer.writerow(
            (
              file_str,
              start_txt,
              end_txt,
              sci,
              common,  # may be ""
              f"{species_prob[ki]:.6f}",
            )
          )


def load_prediction_data(in_path: os.PathLike | str) -> dict[str, Any]:
  with np.load(Path(in_path), allow_pickle=True) as npz:
    result = {k: npz[k] for k in npz.files}
    return result
