# birdnet_batch_inference.py – raw‑audio version
from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import queue
import shutil
import sys
import tempfile
import time
import zipfile
from collections.abc import Generator
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event, Semaphore
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Sequence, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import soundfile as sf  # pip install soundfile
from numpy.lib.stride_tricks import as_strided
from numpy.typing import DTypeLike
from ordered_set import OrderedSet

# try:
#   import tflite_runtime.interpreter as tflite
# except ImportError:  # fallback to full TF (heavier)
from tensorflow.lite.python import interpreter as tflite
from tensorflow.lite.python.interpreter import Interpreter

from birdnet.utils import download_file_tqdm, get_species_from_file
from birdnet_v2.acoustic_models.v2_4.base import AcousticModelBaseV2_4
from birdnet_v2.globals import APP_DIR, WRITE_FLAG
from birdnet_v2.helper import (
  RingField,
  create_shm_ring,
  get_max_n_chunks,
  uint_dtype_for,
)
from birdnet_v2.inference.consumer import Consumer
from birdnet_v2.inference.producer import (
  Producer,
  get_chunks_with_overlap,  # type: ignore
  load_audio_in_chunks_with_overlap,
  shm_ring_from_name,
)
from birdnet_v2.inference.species_tensor import SpeciesTensor
from birdnet_v2.inference.worker import ChildWorker, Worker
from birdnet_v2.model_downloader import ModelDownloader


class AcousticTFModelV2_4(AcousticModelBaseV2_4):
  def __init__(self, lang_id: str) -> None:
    super().__init__("tf")

    model_path, species_list = ModelDownloader.get_model_path_and_labels(
      "acoustic", "v2.4", "tf", lang_id, download_if_not_available=True
    )

    self._model_path = model_path
    self._species_list = species_list
    self._use_custom_model = False

  @property
  def n_species(self) -> int:
    return len(self._species_list)

  def use_custom_model(self, model_path: Path, species_list: Path) -> None:
    if not model_path.is_file():
      raise ValueError(f"Model file '{model_path.absolute()}' does not exist!")

    if not species_list.is_file():
      raise ValueError(f"Species list file '{species_list.absolute()}' does not exist!")

    try:
      interp = tflite.Interpreter(str(model_path.absolute()), num_threads=1)
    except ValueError as e:
      raise ValueError(
        f"Failed to load model '{model_path.absolute()}'. Ensure it is a valid TFLite model."
      ) from e

    loaded_species_list: OrderedSet[str]
    try:
      loaded_species_list = get_species_from_file(species_list, encoding="utf8")
    except Exception as e:
      raise ValueError(
        f"Failed to read species list from '{species_list.absolute()}'. Ensure it is a valid text file."
      ) from e

    output_size = interp.get_output_details()[0]["index"]
    if output_size != len(loaded_species_list):
      raise ValueError(
        f"Model '{model_path.absolute()}' has {output_size} outputs, but species list '{species_list.absolute()}' has {len(loaded_species_list)} species!"
      )

    self._model_path = model_path
    self._species_list = loaded_species_list
    self._use_custom_model = True

  def analyze(
    self,
    files: List[Path] | List[str],
    top_k: int = 5,
    n_jobs: int = 4,
    batch_size: int = 50,
    n_slots_factor: int = 2,
    overlap_duration_s: float = 0,
    default_confidence_threshold: float = 0.1,
    custom_confidence_thresholds: Optional[dict[str, float]] = None,
    use_bandpass: bool = False,
    bandpass_fmin: Optional[int] = None,
    bandpass_fmax: Optional[int] = None,
    apply_sigmoid: bool = True,
    sigmoid_sensitivity: Optional[float] = 1.0,
    custom_species_list: Optional[set[str]] = None,
    half_precision: bool = True,
    max_audio_duration_min: Optional[float] = None,
  ):
    logger = logging.getLogger(__name__)

    file_paths: OrderedSet[Path] = OrderedSet([])
    for file in files:
      if isinstance(file, str):
        file = Path(file)
      if not file.is_file():
        raise ValueError(f"File '{file.absolute()}' does not exist!")
      file_paths.append(file)

    species_whitelist: np.ndarray
    if custom_species_list is not None:
      if len(custom_species_list) == 0:
        raise ValueError("Custom species list is empty!")
      species_ids_whitelist = np.empty(len(custom_species_list), dtype=int)
      for i, species_name in enumerate(custom_species_list):
        if species_name not in self._species_list:
          raise ValueError(
            f"Species '{species_name}' is not in the model's species list! Available species: {', '.join(self._species_list)}"
          )
        species_id = self._species_list.index(species_name)
        species_ids_whitelist[i] = species_id

      species_whitelist = np.full(self.n_species, fill_value=False, dtype=bool)
      species_whitelist[species_ids_whitelist] = True
    else:
      species_whitelist = np.full(self.n_species, fill_value=True, dtype=bool)
    species_whitelist.setflags(write=False)

    thresholds = np.full(self.n_species, default_confidence_threshold, np.float32)

    if custom_confidence_thresholds:
      for species_name, threshold in custom_confidence_thresholds.items():
        if species_name not in self._species_list:
          raise ValueError(
            f"Species '{species_name}' is not in the model's species list! Available species: {', '.join(self._species_list)}"
          )
        species_id = self._species_list.index(species_name)
        thresholds[species_id] = threshold
    thresholds.setflags(write=False)

    # chunks_dtype for max file duration:
    # hopsize 1s = overlap 2s:
    # ---
    # uint8 = 4 ¼ min
    # uint16 = 18 h 12 min 16 s
    # uint32 = 1,19 mio hours = 136 years
    # ---
    # hopsize 3s = overlap 0s:
    # ---
    # uint8 = 12 min 48 s
    # uint16 = 54 h 36 m 48 s
    reserve_n_chunks = 0
    chunks_dtype = np.dtype(np.uint16)
    if max_audio_duration_min is not None:
      reserve_n_chunks = get_max_n_chunks(
        max_audio_duration_min, self.chunk_size_s, overlap_duration_s
      )
      chunks_dtype = uint_dtype_for(max(0, reserve_n_chunks - 1))

    prob_dtype: DTypeLike = np.float16 if half_precision else np.float32

    n_species = len(self._species_list)
    n_files = len(file_paths)

    n_slots = n_jobs * n_slots_factor

    sem_free_slots = mp.Semaphore(n_slots)
    sem_filled_slots = mp.Semaphore(0)
    logger.debug(f"FILL: {sem_filled_slots}, FREE: {sem_free_slots}")

    rf_file_indices = RingField(
      "bnet_ring_file_indices",
      dtype=uint_dtype_for(max(0, n_files - 1)),
      shape=(n_slots, batch_size),
    )

    rf_chunk_indices = RingField(
      "bnet_ring_chunk_indices",
      dtype=chunks_dtype,
      shape=(n_slots, batch_size),
    )

    rf_audio_samples = RingField(
      "bnet_ring_audio_samples",
      dtype=np.dtype(np.float32),
      shape=(n_slots, batch_size, self.chunk_size_samples),
    )

    assert batch_size > 0
    rf_batch_sizes = RingField(
      "bnet_ring_batch_sizes",
      dtype=uint_dtype_for(batch_size),
      shape=(n_slots,),
    )

    rf_flags = RingField(
      "bnet_ring_flags",
      dtype=np.dtype(np.uint8),
      shape=(n_slots,),
    )

    result = SpeciesTensor(
      n_files,
      n_chunks=reserve_n_chunks,
      top_k=top_k,
      n_species=n_species,
      prob_dtype=prob_dtype,
      chunk_indices_dtype=rf_chunk_indices.dtype,
      files_dtype=rf_file_indices.dtype,
    )

    with (
      create_shm_ring(rf_file_indices),
      create_shm_ring(rf_chunk_indices),
      create_shm_ring(rf_audio_samples),
      create_shm_ring(rf_batch_sizes),
      create_shm_ring(rf_flags) as shm_ring_flags,
    ):
      logger.debug("Shared memory initialized.")

      flags = rf_flags.get_array(shm_ring_flags)
      flags[:] = WRITE_FLAG

      prod = mp.Process(
        target=Producer(
          file_paths,
          batch_size=batch_size,
          n_slots=n_slots,
          rf_file_indices=rf_file_indices,
          rf_chunk_indices=rf_chunk_indices,
          rf_audio_samples=rf_audio_samples,
          rf_batch_sizes=rf_batch_sizes,
          rf_flags=rf_flags,
          n_jobs=n_jobs,
          sem_free_slots=sem_free_slots,
          sem_filled_slots=sem_filled_slots,
          chunk_duration_s=self.chunk_size_s,
          overlap_duration_s=overlap_duration_s,
          target_sample_rate=self.sample_rate,
          use_bandpass=use_bandpass,
          bandpass_fmax=bandpass_fmax,
          bandpass_fmin=bandpass_fmin,
          fmax=self.sig_fmax,
          fmin=self.sig_fmin,
          max_supported_n_chunks=reserve_n_chunks
          if max_audio_duration_min is not None
          else None,
        ),
        daemon=True,
      )
      prod.start()

      species_blacklist = ~species_whitelist[np.newaxis, :]
      species_blacklist.setflags(write=False)
      species_thresholds = thresholds[np.newaxis, :]
      species_thresholds.setflags(write=False)
      worker_queue = mp.Queue()
      slot_ptr = mp.Value("I", 0, lock=True)

      workers = [
        mp.Process(
          target=ChildWorker(
            model_path=self._model_path,
            top_k=top_k,
            species_thresholds=species_thresholds,
            species_blacklist=species_blacklist,
            batch_size=batch_size,
            n_slots=n_slots,
            slot_ptr=slot_ptr,
            chunk_duration_samples=self.chunk_size_samples,
            out_q=worker_queue,
            rf_file_indices=rf_file_indices,
            rf_chunk_indices=rf_chunk_indices,
            rf_audio_samples=rf_audio_samples,
            rf_batch_sizes=rf_batch_sizes,
            rf_flags=rf_flags,
            sem_fill=sem_filled_slots,
            sem_free=sem_free_slots,
            apply_sigmoid=apply_sigmoid,
            prob_dtype=prob_dtype,
            sigmoid_sensitivity=sigmoid_sensitivity,
            num_threads=1,  # more than one is not possible with multiprocessing in this tflite version
          ),
          daemon=True,
        )
        for _ in range(n_jobs)
      ]

      for w in workers:
        w.start()

      consumer = Consumer(n_jobs, worker_queue, result)
      consumer()

      prod.join()
      logger.debug("Producer finished.")

      for w in workers:
        w.join()
        logger.debug(f"Worker {w.pid} finished.")
      logger.debug("All workers finished.")

    df = convert_tensor_to_dataframe(
      result, file_paths, self.chunk_size_s, overlap_duration_s, self._species_list
    )
    return df


class PredictionResult:
  def __init__(self, tensor: SpeciesTensor):
    pass

  def to_dataframe(self):
    """
    Convert the prediction result to a pandas DataFrame.
    """
    pass


def convert_tensor_to_dataframe(
  tensor: SpeciesTensor,
  files: OrderedSet[Path],
  chunk_duration_s: Union[int, float],
  overlap_duration_s: Union[int, float],
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
            "start": time.strftime("%-H:%M:%S", time.gmtime(start_sec)),
            "end": time.strftime("%-H:%M:%S", time.gmtime(end_sec)),
            "scientific_name": scientific_name,
            "common_name": common_name,
            "confidence": species_probs[k],
          }
          resulting_lines.append(row)
        else:
          break
  df = pd.DataFrame.from_records(resulting_lines)

  # sorting with float16 is not supported by pandas DataFrame
  df["_confidence32"] = df["confidence"].astype(np.float32, copy=False)
  df = (
    df.sort_values(by=["file", "start", "_confidence32"], ascending=[True, True, False])
    .drop(columns="_confidence32")
    .reset_index(drop=True)
  )
  return df
