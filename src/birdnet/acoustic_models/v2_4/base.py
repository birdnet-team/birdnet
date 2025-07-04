# birdnet_batch_inference.py – raw‑audio version
from __future__ import annotations

import ctypes
import importlib.metadata
import json
import multiprocessing
import multiprocessing as mp
import os
import platform
import time
from collections import OrderedDict
from collections.abc import Iterable

# You'll need these imports in your own code
from datetime import datetime, timedelta
from pathlib import Path

# Next two import lines for this demo only
# backend_protocol.py
from typing import Literal, final

import numpy as np
import pandas as pd
import psutil
from numpy.typing import DTypeLike
from ordered_set import OrderedSet

import birdnet.logging_utils as bn_logging
from birdnet.acoustic_models.base import AcousticModelBase
from birdnet.acoustic_models.inference.consumer import Consumer
from birdnet.acoustic_models.inference.files_analyzer import FilesAnalyzer
from birdnet.acoustic_models.inference.perf_tracker import PerformanceTracker

# try:
#   import tflite_runtime.interpreter as tflite
# except ImportError:  # fallback to full TF (heavier)
from birdnet.acoustic_models.inference.prediction_result import PredictionResult
from birdnet.acoustic_models.inference.producer import ChildProducer
from birdnet.acoustic_models.inference.species_tensor import SpeciesTensor
from birdnet.acoustic_models.inference.worker import ChildWorker
from birdnet.base import (
  MODEL_TYPE_ACOUSTIC,
  MODEL_TYPES,
  MODEL_VERSION_V2_4,
  MODEL_VERSIONS,
)
from birdnet.globals import PKG_NAME, WRITABLE_FLAG
from birdnet.helper import (
  RingField,
  create_shm_ring,
  get_max_n_chunks,
  get_supported_audio_files,
  uint_ctype_from_dtype,
  uint_dtype_for,
)
from birdnet.local_data import get_benchmark_dir
from birdnet.logging_utils import QueueFileWriter, get_package_logging_level


class AcousticModelBaseV2_4(AcousticModelBase):
  def __init__(self) -> None:
    super().__init__()
    self._model_path: Path | None = None
    self._species_list: OrderedSet[str] | None = None
    self._use_custom_model: bool | None = None

  @property
  def n_species(self) -> int:
    return len(self.species_list)

  @property
  def model_path(self) -> Path:
    assert self._model_path is not None
    return self._model_path

  @property
  def species_list(self) -> OrderedSet[str]:
    assert self._species_list is not None
    return self._species_list

  @property
  def use_custom_model(self) -> bool:
    assert self._use_custom_model is not None
    return self._use_custom_model

  @classmethod
  @final
  def get_version(cls) -> MODEL_VERSIONS:
    return MODEL_VERSION_V2_4

  @classmethod
  @final
  def get_model_type(cls) -> MODEL_TYPES:
    return MODEL_TYPE_ACOUSTIC

  @classmethod
  @final
  def get_sig_fmin(cls) -> int:
    return 0

  @classmethod
  @final
  def get_sig_fmax(cls) -> int:
    return 15_000

  @classmethod
  @final
  def get_sample_rate(cls) -> int:
    return 48_000

  @classmethod
  @final
  def get_chunk_size_s(cls) -> float:
    return 3.0

  @classmethod
  @final
  def get_chunk_size_samples(cls) -> int:
    return 144_000  # 3.0 * 48_000

  def analyze(
    self,
    inp: Path | str | Iterable[Path | str],
    /,
    *,
    top_k: int | None = 5,
    n_producers: int = 1,
    n_workers: int = 4,
    batch_size: int = 1,
    n_slots_factor: int = 1,
    overlap_duration_s: float = 0,
    default_confidence_threshold: float | None = 0.1,
    custom_confidence_thresholds: dict[str, float] | None = None,
    use_bandpass: bool = False,
    bandpass_fmin: int | None = None,
    bandpass_fmax: int | None = None,
    apply_sigmoid: bool = True,
    sigmoid_sensitivity: float | None = 1.0,
    custom_species_list: set[str] | None = None,
    half_precision: bool = True,
    max_audio_duration_min: float | None = None,
    show_stats: Literal["no", "minimal", "progress", "benchmark"] = "progress",
    device: str | list[str] = "CPU",
  ):
    start = time.perf_counter()
    start_time = time.time()
    start_timepoint = datetime.now()

    if top_k is not None and top_k > len(self.species_list):
      raise ValueError(
        f"top_k cannot be larger than the number of species ({len(self.species_list)})."
      )

    if top_k is None:
      top_k = len(self.species_list)

    track_performance = show_stats in ("progress", "benchmark")

    # for i, kwargs in enumerate(backend_kwargs):
    #   kwargs["device"] =kwargs["device"].replace("0", str(i)) # Assign different CPU cores
    if isinstance(device, list) and len(device) != n_workers:
      raise ValueError(
        f"Device list length ({len(device)}) does not match number of workers ({n_workers})."
      )

    logging_level = get_package_logging_level()
    logging_queue = multiprocessing.Queue()
    logging_listener = multiprocessing.Process(
      target=QueueFileWriter(logging_queue, logging_level), daemon=True
    )
    logging_listener.start()

    queue_handler = bn_logging.add_queue_handler(logging_queue)

    logger = bn_logging.get_logger(__name__)

    logger.info("Getting input files...")
    parsed_audio_paths = set()
    if isinstance(inp, (Path, str)):
      inp = (Path(inp),)

    if isinstance(inp, Iterable):
      for inp_audio in inp:
        if isinstance(inp_audio, (Path, str)):
          inp_path = Path(inp_audio)
          if inp_path.is_file():
            parsed_audio_paths.add(inp_path.absolute())
          elif inp_path.is_dir():
            parsed_audio_paths.update(get_supported_audio_files(inp_path))
          else:
            raise ValueError(f"Input path '{inp_path}' was not found.")
        else:
          raise ValueError(f"Unsupported input type: {type(inp)}")
    else:
      raise ValueError(f"Unsupported input type: {type(inp)}")

    file_paths: OrderedSet[Path] = OrderedSet(sorted(set(parsed_audio_paths)))
    n_files = len(file_paths)

    logger.info(f"Got {len(file_paths)} audio files for analysis.")

    n_producers = min(n_producers, n_files)
    logger.debug(f"Using {n_producers} producer(s) for {n_files} file(s).")
    logger.info("Starting analysis...")

    species_whitelist: np.ndarray
    if custom_species_list is not None:
      if len(custom_species_list) == 0:
        raise ValueError("Custom species list is empty!")
      species_ids_whitelist = np.empty(len(custom_species_list), dtype=int)
      for i, species_name in enumerate(custom_species_list):
        if species_name not in self.species_list:
          raise ValueError(
            f"Species '{species_name}' is not in the model's species list!"
          )
        species_id = self.species_list.index(species_name)
        species_ids_whitelist[i] = species_id

      species_whitelist = np.full(self.n_species, fill_value=False, dtype=bool)
      species_whitelist[species_ids_whitelist] = True
    else:
      species_whitelist = np.full(self.n_species, fill_value=True, dtype=bool)
    species_whitelist.setflags(write=False)

    if default_confidence_threshold is None:
      default_confidence_threshold = -np.inf
    thresholds = np.full(self.n_species, default_confidence_threshold, np.float32)

    if custom_confidence_thresholds:
      for species_name, threshold in custom_confidence_thresholds.items():
        if species_name not in self.species_list:
          raise ValueError(
            f"Species '{species_name}' is not in the model's species list!"
          )
        species_id = self.species_list.index(species_name)
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
        max_audio_duration_min * 60, self.get_chunk_size_s(), overlap_duration_s
      )
      chunks_dtype = uint_dtype_for(max(0, reserve_n_chunks - 1))

    chunks_code_type = uint_ctype_from_dtype(chunks_dtype)
    max_chunk_idx_ptr = mp.RawValue(chunks_code_type, max(0, reserve_n_chunks - 1))  # type: ignore

    prob_dtype: DTypeLike = np.float16 if half_precision else np.float32

    n_species = self.n_species

    n_slots = n_workers * n_slots_factor

    sem_free_slots = mp.Semaphore(n_slots)
    sem_filled_slots = mp.Semaphore(0)
    sem_active_workers = mp.Semaphore(0)
    logger.debug(f"FILL: {sem_filled_slots}, FREE: {sem_free_slots}")

    rf_file_indices = RingField(
      "bn_ring_file_indices",
      dtype=uint_dtype_for(max(0, n_files - 1)),
      shape=(n_slots, batch_size),
    )

    rf_chunk_indices = RingField(
      "bn_ring_chunk_indices",
      dtype=chunks_dtype,
      shape=(n_slots, batch_size),
    )

    rf_audio_samples = RingField(
      "bn_ring_audio_samples",
      dtype=np.dtype(np.float32),
      shape=(n_slots, batch_size, self.get_chunk_size_samples()),
    )

    assert batch_size > 0
    rf_batch_sizes = RingField(
      "bn_ring_batch_sizes",
      dtype=uint_dtype_for(batch_size),
      shape=(n_slots,),
    )

    rf_flags = RingField(
      "bn_ring_flags",
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
    worker_slot_ptr = mp.Value(
      uint_ctype_from_dtype(uint_dtype_for(n_slots - 1)),  # type: ignore
      0,
      lock=True,  # Lock = false?
    )  # type: ignore
    producer_slot_ptr = mp.Value(
      uint_ctype_from_dtype(uint_dtype_for(n_slots - 1)),  # type: ignore
      0,
      lock=True,
    )  # type: ignore

    pred_dur_queue = mp.SimpleQueue()
    analyzer_queue = mp.SimpleQueue()
    perf_res: mp.SimpleQueue | None = None
    perf_stop_event = mp.Event()
    cancel_event = mp.Event()
    tot_n_chunks_ptr = mp.RawValue(ctypes.c_uint64, 0)
    files_queue = mp.Queue()
    for file_idx, file_path in enumerate(file_paths):
      files_queue.put((file_idx, file_path), block=False)
    for _ in range(n_producers):
      files_queue.put(None, block=False)
    prod_done_ptr = mp.Value(
      uint_ctype_from_dtype(uint_dtype_for(n_producers)),  # type: ignore
      0,
      lock=True,
    )  # type: ignore

    with (
      create_shm_ring(rf_file_indices),
      create_shm_ring(rf_chunk_indices),
      create_shm_ring(rf_audio_samples),
      create_shm_ring(rf_batch_sizes),
      create_shm_ring(rf_flags) as shm_ring_flags,
    ):
      logger.debug("Shared memory initialized.")

      flags = rf_flags.get_array(shm_ring_flags)
      flags[:] = WRITABLE_FLAG

      file_analyzer_proc = mp.Process(
        target=FilesAnalyzer(
          files=file_paths,
          logging_level=logging_level,
          logging_queue=logging_queue,
          chunk_duration_s=AcousticModelBaseV2_4.get_chunk_size_s(),
          overlap_duration_s=overlap_duration_s,
          max_chunk_idx_ptr=max_chunk_idx_ptr,
          rf_chunk_indices=rf_chunk_indices,
          analyzing_result=analyzer_queue,
          tot_n_chunks=tot_n_chunks_ptr,
          cancel_event=cancel_event,
        ),
        daemon=True,
      )
      file_analyzer_proc.start()

      producer_processes: list[mp.Process] = [
        mp.Process(
          target=ChildProducer(
            files_queue=files_queue,
            slot_ptr=producer_slot_ptr,
            batch_size=batch_size,
            n_slots=n_slots,
            rf_file_indices=rf_file_indices,
            rf_chunk_indices=rf_chunk_indices,
            rf_audio_samples=rf_audio_samples,
            rf_batch_sizes=rf_batch_sizes,
            rf_flags=rf_flags,
            logging_queue=logging_queue,
            logging_level=logging_level,
            sem_free_slots=sem_free_slots,
            sem_filled_slots=sem_filled_slots,
            chunk_duration_s=AcousticModelBaseV2_4.get_chunk_size_s(),
            overlap_duration_s=overlap_duration_s,
            target_sample_rate=AcousticModelBaseV2_4.get_sample_rate(),
            use_bandpass=use_bandpass,
            bandpass_fmax=bandpass_fmax,
            bandpass_fmin=bandpass_fmin,
            fmax=AcousticModelBaseV2_4.get_sig_fmax(),
            fmin=AcousticModelBaseV2_4.get_sig_fmin(),
            max_chunk_idx_ptr=max_chunk_idx_ptr,
            prod_done_ptr=prod_done_ptr,
            n_prods=n_producers,
            cancel_event=cancel_event,
          ),
          daemon=True,
        )
        for _ in range(n_producers)
      ]
      for p in producer_processes:
        p.start()

      perf_tracker = None
      if track_performance:
        perf_res = mp.SimpleQueue()
        perf_tracker = mp.Process(
          target=PerformanceTracker(
            pred_dur_queue,
            perf_stop_event,
            update_interval=0.5,
            print_interval=1,
            use_stats_from_last_seconds=30,
            n_workers=n_workers,
            start=start,
            chunk_size_s=AcousticModelBaseV2_4.get_chunk_size_s(),
            logging_queue=logging_queue,
            logging_level=logging_level,
            perf_res=perf_res,
            parent_process_id=os.getpid(),
            rf_flags=rf_flags,
            tot_n_chunks_ptr=tot_n_chunks_ptr,
            cancel_event=cancel_event,
            sem_active_workers=sem_active_workers,
          ),
          daemon=True,
        )
        perf_tracker.start()

      backend_kwargs = [self.get_backend_args() for _ in range(n_workers)]

      devices = device if isinstance(device, list) else [device] * n_workers

      worker_processes = [
        mp.Process(
          target=ChildWorker(
            backend_type=self.get_backend_type(),
            device=devices[i],
            backend_kwargs=backend_kwargs[i],
            top_k=top_k,
            species_thresholds=species_thresholds,
            species_blacklist=species_blacklist,
            batch_size=batch_size,
            n_slots=n_slots,
            slot_ptr=worker_slot_ptr,
            chunk_duration_samples=AcousticModelBaseV2_4.get_chunk_size_samples(),
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
            cancel_event=cancel_event,
            sem_active_workers=sem_active_workers,
          ),
          daemon=True,
        )
        for i in range(n_workers)
      ]

      for w in worker_processes:
        w.start()

      consumer = Consumer(
        n_workers=n_workers,
        worker_queue=worker_queue,
        species_tensor=result,
        max_chunk_index=max_chunk_idx_ptr,
        cancel_event=cancel_event,
      )
      consumer()

      file_analyzer_proc.join()
      logger.debug("File analyzer finished.")

      for p in producer_processes:
        p.join()
        logger.debug(f"Producer {p.pid} finished.")
      logger.debug("All producers finished.")

      for w in worker_processes:
        w.join()
        logger.debug(f"Worker {w.pid} finished.")
      logger.debug("All workers finished.")

      stop = time.perf_counter()
      end_timepoint = datetime.now()

      if track_performance:
        assert perf_tracker is not None
        perf_stop_event.set()
        perf_tracker.join()
        logger.debug("Performance tracker finished.")

    if cancel_event.is_set():
      raise RuntimeError(
        "Analysis was cancelled due to an error. Please check the logs for details."
      )

    res = PredictionResult(
      tensor=result,
      files=file_paths,
      chunk_duration_s=AcousticModelBaseV2_4.get_chunk_size_s(),
      overlap_duration_s=overlap_duration_s,
      species_list=self.species_list,
    )
    del result

    if show_stats == "minimal":
      analyzer_res: dict = analyzer_queue.get()
      file_durations_s: np.ndarray = analyzer_res["file_durations_s"]
      tot_n_chunks = analyzer_res["tot_n_chunks"]
      total_chunks_processed = tot_n_chunks
      wall_time_s = stop - start

      ringbuffer_total_MiB = (
        rf_file_indices.nbytes
        + rf_chunk_indices.nbytes
        + rf_audio_samples.nbytes
        + rf_batch_sizes.nbytes
        + rf_flags.nbytes
      ) / 1024**2

      pc_chunks_per_s = total_chunks_processed / wall_time_s
      pc_audio_min_per_s = (
        pc_chunks_per_s * AcousticModelBaseV2_4.get_chunk_size_s() / 60
      )
      tot_file_duration_h = file_durations_s.sum() / 60**2

      summary = (
        f"-------------------------------\n"
        f"----------- Summary -----------\n"
        f"-------------------------------\n"
        f"Start time: {start_timepoint.strftime('%m/%d/%Y %I:%M %p')}\n"
        f"End time:   {end_timepoint.strftime('%m/%d/%Y %I:%M %p')}\n"
        f"Wall time:  {timedelta(seconds=wall_time_s)}\n"
        f"Input: {n_files} file(s)\n"
        f"\tTotal duration: {tot_file_duration_h:.2f} h\n"
        f"\tMax duration (file): {max_audio_duration_min:.2f} min\n"
        f"Memory usage:\n"
        f"\tRingbuffer total: {ringbuffer_total_MiB:.2f} MiB\n"
        f"\tInference result: {res.memory_size_mb:.2f} MiB\n"
        f"Performance audio processing (all): {pc_audio_min_per_s:.2f} min audio/s ({60 / pc_audio_min_per_s:.2f} s/h audio; {pc_chunks_per_s:.2f} chunks/s)\n"
      )
      print(summary)
    elif show_stats == "benchmark":
      logger.info("Benchmarking is enabled. Collecting performance data...")
      analyzer_res: dict = analyzer_queue.get()
      file_durations_s: np.ndarray = analyzer_res["file_durations_s"]
      tot_n_chunks = analyzer_res["tot_n_chunks"]
      total_chunks_processed = tot_n_chunks

      bm = OrderedDict()
      # Timestamp
      bm["date"] = start_timepoint.isoformat(timespec="seconds")
      bm["start_time"] = start_timepoint.strftime("%m/%d/%Y %I:%M %p")
      bm["end_time"] = end_timepoint.strftime("%m/%d/%Y %I:%M %p")
      # Hardware
      bm["host"] = platform.node()
      bm["CPU"] = platform.processor()
      bm["cpu_cores"] = psutil.cpu_count(logical=False)
      bm["cpu_logical_cores"] = psutil.cpu_count(logical=True)
      bm["ram_GiB"] = psutil.virtual_memory().total / 1024**3
      bm["n_producers"] = n_producers
      bm["n_workers"] = n_workers
      bm["start_method"] = multiprocessing.get_start_method()
      bm["device(s)"] = ", ".join(device) if isinstance(device, list) else device
      # Software
      bm["os"] = f"{platform.system()} {platform.release()}"
      bm["python"] = platform.python_version()
      bm["package_version"] = importlib.metadata.version(PKG_NAME)
      # Model
      bm["model_type"] = AcousticModelBaseV2_4.get_model_type()
      bm["model_version"] = AcousticModelBaseV2_4.get_version()
      bm["custom_model"] = self.use_custom_model
      bm["model_path"] = str(self.model_path.absolute())
      bm["model_n_species"] = self.n_species
      # Dataset
      bm["n_files"] = len(file_paths)
      bm["tot_file_duration_h"] = file_durations_s.sum() / 60**2
      bm["avg_audio_duration_min"] = file_durations_s.mean() / 60
      bm["min_audio_duration_min"] = file_durations_s.min() / 60
      bm["max_audio_duration_min"] = file_durations_s.max() / 60
      bm["max_n_chunks"] = max_chunk_idx_ptr.value + 1
      bm["tot_n_chunks"] = tot_n_chunks
      # Parameter
      bm["chunk_s"] = AcousticModelBaseV2_4.get_chunk_size_s()
      bm["overlap_s"] = overlap_duration_s
      bm["batch_size"] = batch_size
      bm["top_k"] = top_k
      bm["n_slots_factor"] = n_slots_factor
      bm["ringsize"] = n_slots
      bm["apply_sigmoid"] = apply_sigmoid
      bm["sigmoid_sensitivity"] = sigmoid_sensitivity if apply_sigmoid else None
      bm["use_bandpass"] = use_bandpass
      bm["bandpass_fmin"] = bandpass_fmin
      bm["bandpass_fmax"] = bandpass_fmax
      bm["half_precision"] = half_precision
      bm["default_confidence_threshold"] = default_confidence_threshold
      bm["n_custom_species"] = len(custom_species_list) if custom_species_list else 0
      bm["n_custom_confidence_thresholds"] = (
        len(custom_confidence_thresholds) if custom_confidence_thresholds else 0
      )

      wall_time_s = stop - start
      pc_chunks_per_s = total_chunks_processed / wall_time_s
      # samples_per_second = (
      #   AcousticModelBaseV2_4.get_chunk_size_samples() * pc_chunks_per_s
      # )

      pid = os.getpid()
      parent = psutil.Process(pid)
      rampup_first_line_s = start_time - parent.create_time()
      bm["rampup_first_line_s"] = rampup_first_line_s
      bm["wall_time_s"] = wall_time_s
      bm["wall_time_readable"] = str(timedelta(seconds=wall_time_s))
      bm["cpu_time_s"] = None
      bm["rampup_time_s"] = None
      bm["model_pred_ms_per_chunk"] = None
      bm["pc_chunks_per_s"] = pc_chunks_per_s
      bm["pc_audio_min_per_s"] = (
        pc_chunks_per_s * AcousticModelBaseV2_4.get_chunk_size_s() / 60
      )
      bm["pc_s_per_audio_h"] = 60 / bm["pc_audio_min_per_s"]
      bm["result_memory_usage_MiB"] = res.memory_size_mb
      bm["bn_ring_file_indices_MiB"] = rf_file_indices.nbytes / 1024**2
      bm["bn_ring_chunk_indices_MiB"] = rf_chunk_indices.nbytes / 1024**2
      bm["bn_ring_audio_samples_MiB"] = rf_audio_samples.nbytes / 1024**2
      bm["bn_ring_batch_sizes_MiB"] = rf_batch_sizes.nbytes / 1024**2
      bm["bn_ring_flags_MiB"] = rf_flags.nbytes / 1024**2
      bm["bn_ring_total_MiB"] = (
        bm["bn_ring_file_indices_MiB"]
        + bm["bn_ring_chunk_indices_MiB"]
        + bm["bn_ring_audio_samples_MiB"]
        + bm["bn_ring_batch_sizes_MiB"]
        + bm["bn_ring_flags_MiB"]
      )

      assert track_performance
      assert perf_res is not None
      perf_result = perf_res.get()
      total_chunks_processed = perf_result["total_chunks_processed"]
      cpu_time_s = perf_result["summed_prediction_duration_s"]
      model_pred_ms_per_chunk = cpu_time_s / total_chunks_processed * 1000

      raw_chunks_per_s = total_chunks_processed / (
        cpu_time_s / perf_result["avg_busy_slots"]
      )

      # Metrics
      bm["cpu_time_s"] = cpu_time_s
      bm["raw_chunks_per_s"] = raw_chunks_per_s
      bm["raw_min_per_s"] = (
        bm["raw_chunks_per_s"] * AcousticModelBaseV2_4.get_chunk_size_s() / 60
      )
      bm["raw_avg_chunks_per_s_last"] = perf_result["avg_chunks_per_s_last"]
      bm["raw_avg_raw_min_per_s_last"] = (
        bm["raw_avg_chunks_per_s_last"] * AcousticModelBaseV2_4.get_chunk_size_s() / 60
      )
      bm["raw_avg_s_for_one_hour_last"] = 60 / bm["raw_avg_raw_min_per_s_last"]
      bm["raw_s_for_one_hour"] = 60 / bm["raw_min_per_s"]
      bm["raw_chunks_per_s_max"] = perf_result["max_raw_chunks_per_s"]
      bm["raw_min_per_s_max"] = (
        bm["raw_chunks_per_s_max"] * AcousticModelBaseV2_4.get_chunk_size_s() / 60
      )
      bm["raw_s_for_one_hour_max"] = 60 / bm["raw_min_per_s_max"]
      bm["rampup_time_s"] = perf_result["ramp_up_time_until_first_pred_s"]
      bm["n_chunks_processed"] = total_chunks_processed
      bm["n_batches_processed"] = perf_result["total_batches_processed"]
      bm["model_pred_ms_per_chunk"] = model_pred_ms_per_chunk
      bm["model_pred_ms_per_batch"] = (
        bm["cpu_time_s"] / bm["n_batches_processed"] * 1000
      )

      bm["n_usage_recordings"] = perf_result["n_usage_recordings"]

      bm["max_memory_usages_MiB"] = perf_result["max_memory_usages_MiB"]
      bm["avg_memory_usages_MiB"] = perf_result["avg_memory_usages_MiB"]

      bm["max_cpu_usages_pct"] = perf_result["max_cpu_usages_pct"]
      bm["avg_cpu_usages_pct"] = perf_result["avg_cpu_usages_pct"]

      bm["avg_free_slots"] = perf_result["avg_free_slots"]
      bm["avg_filled_slots"] = perf_result["avg_filled_slots"]
      bm["avg_busy_slots"] = perf_result["avg_busy_slots"]
      bm["avg_preloaded_slots"] = perf_result["avg_preloaded_slots"]

      bm["avg_free_slots_last"] = perf_result["avg_free_slots_last"]
      bm["avg_filled_slots_last"] = perf_result["avg_filled_slots_last"]
      bm["avg_busy_slots_last"] = perf_result["avg_busy_slots_last"]
      bm["avg_preloaded_slots_last"] = perf_result["avg_preloaded_slots_last"]

      benchmark_dir = get_benchmark_dir(
        model=AcousticModelBaseV2_4.get_model_type(),
        version=AcousticModelBaseV2_4.get_version(),
      )
      stats_out = (
        benchmark_dir / f"{end_timepoint.strftime('analyze_%Y%m%dT%H%M%S')}.json"
      )
      with open(stats_out, "w", encoding="utf8") as f:
        json.dump(bm, f, indent=2, ensure_ascii=False)

      meta_df_out = benchmark_dir / "analyze.csv"
      meta_df = pd.DataFrame.from_records([bm])
      meta_df.to_csv(
        meta_df_out, mode="a", header=not meta_df_out.exists(), index=False
      )

      meta_human_readable_out = stats_out.with_suffix(".txt")

      file_formats = ", ".join(sorted({x.suffix[1:].upper() for x in file_paths}))
      summary = (
        f"-------------------------------\n"
        f"------ Benchmark summary ------\n"
        f"-------------------------------\n"
        f"Start time: {bm['start_time']}\n"
        f"End time:   {bm['end_time']}\n"
        f"Wall time:  {bm['wall_time_readable']}\n"
        f"Input: {bm['n_files']} file(s) ({file_formats})\n"
        f"\tTotal duration: {bm['tot_file_duration_h']:.2f} h\n"
        f"\tMax duration (file): {bm['max_audio_duration_min']:.2f} min\n"
        f"# Processes:\n"
        f"\tProducer(s): {bm['n_producers']}\n"
        f"\tWorker(s): {bm['n_workers']}\n"
        f"Average amount of batches (ringbuffer: {n_slots} slots):\n"
        f"\tBusy: {bm['avg_busy_slots']:.1f} slots\n"
        f"\tPreloaded: {bm['avg_preloaded_slots']:.1f} slots\n"
        f"\tFree: {bm['avg_free_slots']:.1f} slots\n"
        f"Memory usage:\n"
        f"\tProgram max: {bm['max_memory_usages_MiB']:.2f} MiB\n"
        f"\tRingbuffer total: {bm['bn_ring_total_MiB']:.2f} MiB\n"
        f"\tInference result: {bm['result_memory_usage_MiB']:.2f} MiB\n"
        f"Performance:\n"
        f"\tAudio processing (all): {bm['pc_audio_min_per_s']:.2f} min audio/s ({bm['pc_s_per_audio_h']:.2f} s/h audio; {bm['pc_chunks_per_s']:.2f} chunks/s)\n"
        f"\tAudio processing (computation):\n"
        f"\t\tMean: {bm['raw_min_per_s']:.2f} min audio/s ({bm['raw_s_for_one_hour']:.2f} s/h audio; {bm['raw_chunks_per_s']:.2f} chunks/s)\n"
        f"\t\tMean (last 30s): {bm['raw_avg_raw_min_per_s_last']:.2f} min audio/s ({bm['raw_avg_s_for_one_hour_last']:.2f} s/h audio; {bm['raw_avg_chunks_per_s_last']:.2f} chunks/s)\n"
        f"\t\tBest: {bm['raw_min_per_s_max']:.2f} min audio/s ({bm['raw_s_for_one_hour_max']:.2f} s/h audio; {bm['raw_chunks_per_s_max']:.2f} chunks/s)\n"
        f"\tPrediction speed: {bm['model_pred_ms_per_chunk']:.2f} ms/chunk ({bm['model_pred_ms_per_batch']:.2f} ms/batch)\n"
        f"Benchmark results written to:\n"
        f"\t{meta_human_readable_out.absolute()}\n"
        f"\t{stats_out.absolute()}\n"
        f"\t{meta_df_out.absolute()}\n"
      )
      meta_human_readable_out.write_text(summary, encoding="utf8")
      print(summary)

    logging_queue.put_nowait(None)
    logging_listener.join()
    bn_logging.remove_queue_handler(queue_handler)

    return res
