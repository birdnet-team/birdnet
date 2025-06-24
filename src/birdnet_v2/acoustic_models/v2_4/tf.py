# birdnet_batch_inference.py – raw‑audio version
from __future__ import annotations

import ctypes
import json
import logging
import logging.handlers
import math
import multiprocessing
import multiprocessing as mp
import os
import platform
import queue
import shutil
import sys
import tempfile
import time
import zipfile
from collections import OrderedDict, deque
from collections.abc import Generator

# You'll need these imports in your own code
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event, Semaphore
from pathlib import Path

# Next two import lines for this demo only
from random import choice, random
from typing import (
  Any,
  Callable,
  Iterable,
  List,
  Literal,
  Optional,
  Sequence,
  Set,
  Tuple,
  Union,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
import psutil
import soundfile as sf  # pip install soundfile
from numpy.lib.stride_tricks import as_strided
from numpy.typing import DTypeLike
from ordered_set import OrderedSet

# try:
#   import tflite_runtime.interpreter as tflite
# except ImportError:  # fallback to full TF (heavier)
from tensorflow.lite.python import interpreter as tflite
from tensorflow.lite.python.interpreter import Interpreter

import birdnet_v2.logging_utils as bn_logging
from birdnet.utils import download_file_tqdm, get_species_from_file
from birdnet_v2.acoustic_models.v2_4.base import AcousticModelBaseV2_4
from birdnet_v2.globals import APP_DIR, WRITE_FLAG
from birdnet_v2.helper import (
  RingField,
  code_from_dtype,
  create_shm_ring,
  get_max_n_chunks,
  max_value_for_uint_dtype,
  uint_ctype_from_dtype,
  uint_dtype_for,
)
from birdnet_v2.inference.consumer import Consumer
from birdnet_v2.inference.perf_tracker import PerformanceTracker
from birdnet_v2.inference.producer import (
  FilesAnalyzer,
  Producer,
  get_chunks_with_overlap,  # type: ignore
  load_audio_in_chunks_with_overlap,
)
from birdnet_v2.inference.species_tensor import SpeciesTensor
from birdnet_v2.inference.worker import ChildWorker
from birdnet_v2.logging_utils import (
  QueueFileWriter,
  get_package_logging_level,
)
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
    files: list[Path] | list[str],
    top_k: int = 5,
    n_jobs: int = 4,
    batch_size: int = 50,
    n_slots_factor: int = 2,
    overlap_duration_s: float = 0,
    default_confidence_threshold: float = 0.1,
    custom_confidence_thresholds: dict[str, float] | None = None,
    use_bandpass: bool = False,
    bandpass_fmin: int | None = None,
    bandpass_fmax: int | None = None,
    apply_sigmoid: bool = True,
    sigmoid_sensitivity: float | None = 1.0,
    custom_species_list: set[str] | None = None,
    half_precision: bool = True,
    max_audio_duration_min: float | None = None,
    track_performance: bool = True,
  ):
    pid = os.getpid()
    parent = psutil.Process(pid)
    ramp_up_here = time.time() - parent.create_time()
    start = time.perf_counter()
    # if multiprocessing.get_start_method() not in ("spawn", "fork"):
    #   raise ValueError("Multiprocessing start method must be 'spawn' or 'fork'!")
    logging_level = get_package_logging_level()
    logging_queue = multiprocessing.Queue()
    logging_listener = multiprocessing.Process(
      target=QueueFileWriter(logging_queue, logging_level), daemon=True
    )
    logging_listener.start()

    queue_handler = bn_logging.add_queue_handler(logging_queue)

    logger = bn_logging.get_logger(__name__)
    logger.info("Starting analysis...")

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
    # hopsize 3s & overlap 0s: n-chunks ÷ 1200
    # ---
    # uint8 = 255 chunks = 0 m 12 s
    # uint16 = 65 535 chunks = 54 m 36 s
    # uint32 = 4 294 967 295 chunks = 2 485 days = 59 652 h
    reserve_n_chunks = 0
    chunks_dtype = np.dtype(np.uint32)
    if max_audio_duration_min is not None:
      reserve_n_chunks = get_max_n_chunks(
        max_audio_duration_min * 60, self.chunk_size_s, overlap_duration_s
      )
      chunks_dtype = uint_dtype_for(max(0, reserve_n_chunks - 1))

    chunks_code_type = uint_ctype_from_dtype(chunks_dtype)
    max_chunk_idx_ptr = mp.RawValue(chunks_code_type, max(0, reserve_n_chunks - 1))  # type: ignore

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
      dtype=np.dtype(np.uint8),  # 4 Values
      shape=(n_slots,),
    )

    rf_file_indices.cleanup()
    rf_chunk_indices.cleanup()
    rf_audio_samples.cleanup()
    rf_batch_sizes.cleanup()
    rf_flags.cleanup()

    result = SpeciesTensor(
      n_files,
      n_chunks=reserve_n_chunks,
      top_k=top_k,
      n_species=n_species,
      prob_dtype=prob_dtype,
      chunk_indices_dtype=rf_chunk_indices.dtype,
      files_dtype=rf_file_indices.dtype,
    )

    species_blacklist = ~species_whitelist[np.newaxis, :]
    species_blacklist.setflags(write=False)
    species_thresholds = thresholds[np.newaxis, :]
    species_thresholds.setflags(write=False)
    worker_queue = mp.Queue()
    slot_ptr = mp.Value(
      uint_ctype_from_dtype(uint_dtype_for(n_slots - 1)), 0, lock=True
    )  # type: ignore

    pred_dur_queue = mp.SimpleQueue()
    analyzer_queue = mp.SimpleQueue()
    perf_res: mp.SimpleQueue | None = None
    stop_event = mp.Event()
    stop_time = mp.RawValue(ctypes.c_float, 0.0)  # float32
    tot_n_chunks_ptr = mp.RawValue(ctypes.c_uint64, 0)

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

      file_analyzer_proc = mp.Process(
        target=FilesAnalyzer(
          files=file_paths,
          logging_level=logging_level,
          logging_queue=logging_queue,
          chunk_duration_s=self.chunk_size_s,
          overlap_duration_s=overlap_duration_s,
          max_chunk_idx_ptr=max_chunk_idx_ptr,
          rf_chunk_indices=rf_chunk_indices,
          analyzing_result=analyzer_queue,
          tot_n_chunks=tot_n_chunks_ptr,
        )
      )
      file_analyzer_proc.start()

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
          logging_queue=logging_queue,
          logging_level=logging_level,
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
          max_chunk_idx_ptr=max_chunk_idx_ptr,
        ),
        daemon=True,
      )
      prod.start()

      inference_tracker = None
      if track_performance:
        perf_res = mp.SimpleQueue()
        inference_tracker = mp.Process(
          target=PerformanceTracker(
            pred_dur_queue,
            stop_event,
            update_interval=0.5,
            print_interval=1,
            print_last_n=500,
            stop_time=stop_time,
            start=start,
            chunk_size_s=self.chunk_size_s,
            logging_queue=logging_queue,
            logging_level=logging_level,
            perf_res=perf_res,
            parent_process_id=os.getpid(),
            rf_flags=rf_flags,
            tot_n_chunks_ptr=tot_n_chunks_ptr,
          ),
          daemon=True,
        )
        inference_tracker.start()

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
            logging_queue=logging_queue,
            logging_level=logging_level,
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
            pred_dur_queue=pred_dur_queue,
            track_performance=track_performance,
            num_threads=1,  # more than one is not possible with multiprocessing in this tflite version
          ),
          daemon=True,
        )
        for _ in range(n_jobs)
      ]

      for w in workers:
        w.start()

      consumer = Consumer(
        n_jobs=n_jobs,
        worker_queue=worker_queue,
        species_tensor=result,
        max_chunk_index=max_chunk_idx_ptr,
        pred_dur_queue=pred_dur_queue,
      )
      consumer()

      file_analyzer_proc.join()
      logger.debug("File analyzer finished.")

      prod.join()
      logger.debug("Producer finished.")

      for w in workers:
        w.join()
        logger.debug(f"Worker {w.pid} finished.")
      logger.debug("All workers finished.")

      if track_performance:
        stop = time.perf_counter()
        stop_time.value = stop
        stop_event.set()
        assert inference_tracker is not None
        assert perf_res is not None
        inference_tracker.join()
        perf_result = perf_res.get()

        wall_time_s = stop - start

        analyzer_res: dict = analyzer_queue.get()
        file_durations_s = analyzer_res["file_durations_s"]
        max_chunk_index = analyzer_res["max_chunk_index"]
        tot_n_chunks = analyzer_res["tot_n_chunks"]

        total_chunks_processed = perf_result["total_chunks_processed"]
        cpu_time_s = perf_result["summed_prediction_duration_s"]
        memory_usages_mb = perf_result["memory_usages_mb"]
        cpu_usages_pct = perf_result["cpu_usages_pct"]
        free_slots = perf_result["free_slots"]
        filled_slots = perf_result["filled_slots"]
        busy_slots = perf_result["busy_slots"]
        preloaded_slots = perf_result["preloaded_slots"]
        ramp_up_time_until_first_pred_s = perf_result["ramp_up_time_until_first_pred_s"]

        model_pred_ms_per_chunk = cpu_time_s / total_chunks_processed * 1000
        pc_chunks_per_s = total_chunks_processed / wall_time_s
        pc_audio_min_per_s = pc_chunks_per_s * self.chunk_size_s / 60
        n_files = len(file_paths)

        meta = OrderedDict()
        # Timestamp
        meta["date"] = datetime.now().isoformat(timespec="seconds")
        # Hardware
        meta["host"] = platform.node()
        meta["cpu"] = platform.processor()
        meta["cpu_cores"] = psutil.cpu_count(logical=False)
        meta["cpu_logical_cores"] = psutil.cpu_count(logical=True)
        meta["ram_GiB"] = psutil.virtual_memory().total / 1024**3
        meta["n_jobs"] = n_jobs
        meta["start_method"] = multiprocessing.get_start_method()
        # Software
        meta["os"] = f"{platform.system()} {platform.release()}"
        meta["python"] = platform.python_version()
        meta["birdnet"] = "2.0.0"
        # Model
        meta["model_type"] = "tf"
        meta["model_version"] = "2.4"
        meta["custom_model"] = self._use_custom_model
        meta["model_path"] = str(self._model_path.absolute())
        meta["model_n_species"] = len(self._species_list)
        # Dataset
        meta["n_files"] = len(file_paths)
        meta["tot_file_duration_h"] = sum(file_durations_s) / 60**2
        # meta["max_chunk_index"] = max_chunk_index
        meta["max_n_chunks"] = max_chunk_idx_ptr.value + 1
        meta["tot_n_chunks"] = tot_n_chunks
        # Parameter
        meta["chunk_s"] = self.chunk_size_s
        meta["overlap_s"] = overlap_duration_s
        meta["batch_size"] = batch_size
        meta["top_k"] = top_k
        # meta["n_slots_factor"] = n_slots_factor
        meta["ringsize"] = n_slots
        meta["apply_sigmoid"] = apply_sigmoid
        meta["sigmoid_sensitivity"] = sigmoid_sensitivity if apply_sigmoid else None
        meta["use_bandpass"] = use_bandpass
        meta["bandpass_fmin"] = bandpass_fmin
        meta["bandpass_fmax"] = bandpass_fmax
        meta["half_precision"] = half_precision
        meta["default_confidence_threshold"] = default_confidence_threshold
        meta["n_custom_species"] = (
          len(custom_species_list) if custom_species_list else 0
        )
        meta["n_custom_confidence_thresholds"] = (
          len(custom_confidence_thresholds) if custom_confidence_thresholds else 0
        )
        # Metrics
        meta["rampup_first_line_s"] = ramp_up_here
        meta["max_audio_duration_min"] = -1  # TODO
        meta["wall_time_s"] = wall_time_s
        meta["cpu_time_s"] = cpu_time_s
        meta["rampup_time_s"] = ramp_up_time_until_first_pred_s
        meta["n_chunks_processed"] = total_chunks_processed
        meta["model_pred_ms_per_chunk"] = model_pred_ms_per_chunk
        meta["pc_chunks_per_s"] = pc_chunks_per_s
        meta["pc_audio_min_per_s"] = pc_audio_min_per_s
        meta["result_memory_usage_MiB"] = result.memory_usage_mb
        assert len(memory_usages_mb) == len(cpu_usages_pct)
        n_usage_recordings = len(memory_usages_mb)
        meta["n_usage_recordings"] = n_usage_recordings
        meta["max_memory_usages_MiB"] = max(memory_usages_mb, default=np.nan)
        meta["avg_memory_usage_MiB"] = (
          sum(memory_usages_mb) / len(memory_usages_mb)
          if len(memory_usages_mb) > 0
          else np.nan
        )
        meta["max_cpu_usages_pct"] = max(cpu_usages_pct, default=np.nan)
        meta["avg_cpu_usage_pct"] = (
          sum(cpu_usages_pct) / len(cpu_usages_pct)
          if len(cpu_usages_pct) > 0
          else np.nan
        )
        meta["avg_free_slots"] = (
          sum(free_slots) / len(free_slots) if len(free_slots) > 0 else np.nan
        )
        meta["avg_filled_slots"] = (
          sum(filled_slots) / len(filled_slots) if len(filled_slots) > 0 else np.nan
        )
        meta["avg_busy_slots"] = (
          sum(busy_slots) / len(busy_slots) if len(busy_slots) > 0 else np.nan
        )
        meta["avg_preloaded_slots"] = (
          sum(preloaded_slots) / len(preloaded_slots)
          if len(preloaded_slots) > 0
          else np.nan
        )

        meta_out = Path(tempfile.gettempdir()) / "meta.json"
        with open(meta_out, "w", encoding="utf8") as f:
          json.dump(meta, f, indent=2, ensure_ascii=False)
        meta_df_out = Path(tempfile.gettempdir()) / "meta.csv"
        meta_df = pd.DataFrame.from_records([meta])
        meta_df.to_csv(
          meta_df_out, mode="a", header=not meta_df_out.exists(), index=False
        )
        logger.info(f"Meta data JSON written to: {meta_out.absolute()}")
        logger.info(f"Meta data CSV written to: {meta_df_out.absolute()}")

    logging_queue.put_nowait(None)
    logging_listener.join()
    bn_logging.remove_queue_handler(queue_handler)

    res = PredictionResult(
      tensor=result,
      files=file_paths,
      chunk_duration_s=self.chunk_size_s,
      overlap_duration_s=overlap_duration_s,
      species_list=self._species_list,
    )
    return res


class PredictionResult:
  def __init__(
    self,
    tensor: SpeciesTensor,
    files: OrderedSet[Path],
    chunk_duration_s: int | float,
    overlap_duration_s: int | float,
    species_list: OrderedSet[str],
  ):
    self._tensor = tensor
    self._files = files
    self._chunk_duration_s = chunk_duration_s
    self._overlap_duration_s = overlap_duration_s
    self._species_list = species_list

  def to_dataframe(self):
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
