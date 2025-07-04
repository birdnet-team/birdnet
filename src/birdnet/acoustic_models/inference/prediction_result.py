from __future__ import annotations

import csv
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from ordered_set import OrderedSet
from tqdm import tqdm

from birdnet.acoustic_models.inference.species_tensor import SpeciesTensor


class PredictionResult:
  def __init__(
    self,
    tensor: SpeciesTensor,
    files: OrderedSet[Path],
    species_list: OrderedSet[str],
    chunk_duration_s: int | float,
    overlap_duration_s: int | float,
  ) -> None:
    all_files = [str(file.absolute()) for file in files]
    max_len = max(map(len, all_files))
    self._files = np.asarray([str(p) for p in files], dtype=f"<U{max_len}")
    # self._files = np.array([str(file.absolute()) for file in files], dtype=object)
    self._chunk_duration_s = np.float32(chunk_duration_s)
    self._overlap_duration_s = np.float32(overlap_duration_s)
    self._species_list = np.array(list(species_list), dtype=object)
    self._species_probs = tensor._species_probs
    self._species_ids = tensor._species_ids
    self._species_masked = tensor._species_masked

  @property
  def memory_size_mb(self) -> float:
    return (
      self._species_ids.nbytes
      + self._species_probs.nbytes
      + self._species_masked.nbytes
      + self._files.nbytes
      + self._chunk_duration_s.nbytes
      + self._overlap_duration_s.nbytes
      + self._species_list.nbytes
    ) / 1024**2

  @classmethod
  def load(cls, path: os.PathLike | str) -> Self:
    result = cls.__new__(cls)
    data = load_prediction_data(path)

    result._species_ids = data["species_ids"]
    result._species_probs = data["species_probs"]
    result._species_masked = data["species_masked"]
    result._files = data["files"]
    result._chunk_duration_s = data["chunk_duration_s"]
    result._overlap_duration_s = data["overlap_duration_s"]
    result._species_list = data["species_list"]
    return result

  @property
  def chunk_duration_s(self) -> float:
    return float(self._chunk_duration_s)

  @property
  def overlap_duration_s(self) -> float:
    return float(self._overlap_duration_s)

  def dump(self, npz_out_path: os.PathLike | str, /, *, compress: bool = True) -> None:
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
      chunk_duration_s=self._chunk_duration_s,
      overlap_duration_s=self._overlap_duration_s,
      species_list=self._species_list,
    )

  def to_dataframe(self) -> pd.DataFrame:
    return convert_tensor_to_dataframe(
      self._species_ids,
      self._species_probs,
      self._species_masked,
      self._files,
      self.chunk_duration_s,
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
  chunk_duration_s: int | float,
  overlap_duration_s: int | float,
  species_list: np.ndarray,
  /,
  *,
  silent: bool = False,
) -> pd.DataFrame:
  top_k = species_probs.shape[2]
  max_chunks = species_probs.shape[1]
  n_files = len(files)
  chunks = []
  resulting_lines = []
  for i in range(max_chunks):
    start = i * chunk_duration_s - (i * overlap_duration_s)
    end = start + chunk_duration_s
    chunks.append((start, end))
  non_masked_entry_count = np.count_nonzero(~species_masked)
  with tqdm(
    total=non_masked_entry_count,
    desc="Creating DataFrame",
    unit="chunk",
    disable=silent,
  ) as pbar:
    for i in range(n_files):
      for j in range(max_chunks):
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

            start_sec = int(chunks[j][0])
            end_sec = int(chunks[j][1])

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


def fast_save_tensor_to_csv(
  result: PredictionResult,
  out_path: os.PathLike | str,
  *,
  buf_KiB: int = 256,  # Schreibpuffer in KiB
  encoding="utf-8",
  silent: bool = False,
) -> None:
  """Schreibt *tensor* extrem speicherschonend & schnellen CSV‐Dump."""

  sid = result._species_ids
  sprob = result._species_probs
  smask = result._species_masked
  top_k = result._species_probs.shape[2]
  n_files = len(result._files)
  n_chunks = result._species_probs.shape[1]
  hop = result.chunk_duration_s - result.overlap_duration_s

  sec = np.arange(n_chunks, dtype=np.float64) * hop
  start_fmt = np.array(
    [time.strftime("%H:%M:%S", time.gmtime(int(v))).encode(encoding) for v in sec],
    dtype="S8",
  )
  end_fmt = np.array(
    [
      time.strftime("%H:%M:%S", time.gmtime(int(v + result.chunk_duration_s))).encode(
        encoding
      )
      for v in sec
    ],
    dtype="S8",
  )

  hdr = b"file,start,end,scientific_name,common_name,confidence\n"

  # Großer roher Binär-Puffer; wir umgehen csv.writer komplett
  buf_bytes = buf_KiB * 1024
  block: list[bytes] = []
  block_size = 0

  with open(out_path, "wb", buffering=buf_bytes) as fh:
    fh.write(hdr)
    non_masked_entry_count = np.count_nonzero(~smask)
    with tqdm(
      total=non_masked_entry_count,
      desc="Writing CSV",
      unit="chunk",
      disable=silent,
    ) as pbar:
      for fi in range(n_files):
        file_b = result._files[fi].encode(encoding)
        ids = sid[fi]  # View [chunks, top_k]
        probs = sprob[fi]
        masks = smask[fi]

        for ci in range(n_chunks):
          # Bool-Maske für gültige Spezies
          valid = ~masks[ci]
          if not valid.any():
            continue

          s8 = start_fmt[ci]
          e8 = end_fmt[ci]

          # Slice ohne Python-Loop
          k_lim = valid.argmax() if not valid.all() else top_k
          sp_ids = ids[ci, :k_lim]
          sp_conf = probs[ci, :k_lim]

          # → Strings zusammenbauen (bytes, weil schneller)
          for k in range(k_lim):
            name: str = result._species_list[sp_ids[k]]
            sci = name
            common = ""
            if "_" in name:
              sci, _, common = name.partition("_")
            line = (
              file_b
              + b","
              + s8
              + b","
              + e8
              + b","
              + sci.encode(encoding)
              + b","
              + common.encode(encoding)
              + b","
              + f"{sp_conf[k]:.6f}".encode(encoding)
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
  chunk_duration_s: float,
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
  n_chunks = species_ids.shape[1]
  n_files = len(files)
  step = chunk_duration_s - overlap_duration_s  # “effective hop”

  # fast pre-compute the start/end list once
  starts = np.arange(n_chunks, dtype=np.float64) * step
  ends = starts + chunk_duration_s

  header = "file,start,end,scientific_name,common_name,confidence"

  # --- stream write --------------------------------------------------------
  with open(out_path, "w", encoding=encoding, newline=newline) as fh:
    writer = csv.writer(fh)
    writer.writerow(header.split(","))  # one tiny allocation

    for fi in range(n_files):
      file_str = files[fi]
      for ci in range(n_chunks):
        # tensor already sorted per-chunk by score ↓
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


if __name__ == "__main__":
  res = PredictionResult.load(
    Path(tempfile.gettempdir()) / "predictions.npz"
  )  # Example usage

  df = res.to_csv(
    Path(tempfile.gettempdir()) / "predictions.csv",
    encoding="utf-8",
  )

  res = load_prediction_data(Path(tempfile.gettempdir()) / "predictions.npz")
  print(res)
