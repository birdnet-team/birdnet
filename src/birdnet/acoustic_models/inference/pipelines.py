from __future__ import annotations

import ctypes
import json
import multiprocessing as mp
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import numpy as np
import psutil
from numpy.typing import DTypeLike
from ordered_set import OrderedSet

import birdnet.logging_utils as bn_logging
from birdnet.acoustic_models.inference.consumer import Consumer
from birdnet.acoustic_models.inference.emb.benchmarking import (
  FullBenchmarkEmbMeta,
  MinimalBenchmarkEmbMeta,
)
from birdnet.acoustic_models.inference.emb.prediction_result import (
  EmbeddingsPredictionResult,
)
from birdnet.acoustic_models.inference.emb.tensor import EmbeddingsTensor
from birdnet.acoustic_models.inference.emb.worker import EmbeddingsWorker
from birdnet.acoustic_models.inference.files_analyzer import FilesAnalyzer
from birdnet.acoustic_models.inference.perf_tracker import (
  PerformanceTracker,
  PerformanceTrackingResult,
)
from birdnet.acoustic_models.inference.producer import ChildProducer
from birdnet.acoustic_models.inference.scores.benchmarking import (
  FullBenchmarkMeta,
  MinimalBenchmarkMeta,
)
from birdnet.acoustic_models.inference.scores.prediction_result import PredictionResult
from birdnet.acoustic_models.inference.scores.tensor import ScoresTensor
from birdnet.acoustic_models.inference.scores.worker import ChildWorker
from birdnet.backends import (
  InferenceBackendLoader,
  PBInferenceBackend,
  TFInferenceBackend,
)
from birdnet.globals import (
  ACOUSTIC_MODEL_VERSIONS,
  MODEL_BACKEND_PB,
  MODEL_BACKEND_TF,
  MODEL_BACKENDS,
  MODEL_PRECISIONS,
  MODEL_TYPE_ACOUSTIC,
  PKG_NAME,
  WRITABLE_FLAG,
)
from birdnet.helper import (
  SF_FORMATS,
  RingField,
  create_shm_ring,
  get_max_n_segments,
  get_supported_audio_files,
  uint_ctype_from_dtype,
  uint_dtype_for,
)
from birdnet.local_data import get_benchmark_dir
from birdnet.logging_utils import QueueFileWriter, get_package_logging_level


def predict_embeddings_from_recordings(
  inp: Path | str | Iterable[Path | str],
  model_species_list: OrderedSet[str],
  model_path: Path,
  model_backend: MODEL_BACKENDS,
  model_backend_kwargs: dict,
  model_version: ACOUSTIC_MODEL_VERSIONS,
  model_segment_size_s: float,
  model_sample_rate: int,
  model_sig_fmin: int,
  model_sig_fmax: int,
  model_precision: MODEL_PRECISIONS,
  model_is_custom: bool,
  model_emb_dim: int,
  feeders: int = 1,
  workers: int = 4,
  batch_size: int = 1,
  prefetch_ratio: int = 1,
  overlap_duration_s: float = 0,
  use_bandpass: bool = False,
  bandpass_fmin: int | None = None,
  bandpass_fmax: int | None = None,
  half_precision: bool = True,
  max_audio_duration_min: float | None = None,
  show_stats: Literal["no", "minimal", "progress", "benchmark"] = "no",
  device: str | list[str] = "CPU",
) -> EmbeddingsPredictionResult:
  start = time.perf_counter()
  start_time = time.time()
  start_timepoint = datetime.now()

  if not batch_size >= 1:
    raise ValueError(
      "Value for 'batch_size' is invalid! It needs to be larger than or equal to 1."
    )

  if not feeders >= 1:
    raise ValueError(
      "Value for 'feeders' is invalid! It needs to be larger than or equal to 1."
    )

  if not workers >= 1:
    raise ValueError(
      "Value for 'workers' is invalid! It needs to be larger than or equal to 1."
    )

  if not prefetch_ratio >= 0:
    raise ValueError(
      "Value for 'prefetch_ratio' is invalid! It needs to be larger than or equal to 0."
    )

  if not 0 <= overlap_duration_s < 3:
    raise ValueError(
      "Value for 'overlap_duration_s' is invalid! It needs to be in interval [0.0, 3.0)."
    )

  if use_bandpass:
    if bandpass_fmin is None:
      raise ValueError("Value for 'bandpass_fmin' is required if 'use_bandpass==True'!")
    if bandpass_fmax is None:
      raise ValueError("Value for 'bandpass_fmax' is required if 'use_bandpass==True'!")

    if bandpass_fmin < 0:
      raise ValueError(
        "Value for 'bandpass_fmin' is invalid! It needs to be larger than zero."
      )

    if bandpass_fmax <= bandpass_fmin:
      raise ValueError(
        "Value for 'bandpass_fmax' is invalid! It needs to be larger than 'bandpass_fmin'."
      )

  if max_audio_duration_min is not None and not max_audio_duration_min > 0:
    raise ValueError(
      "Value for 'max_audio_duration_min' is invalid! It needs to be either None, or larger than zero."
    )

  if show_stats not in ("no", "minimal", "progress", "benchmark"):
    raise ValueError(
      f"Value for 'show_stats' is invalid! It needs to be one of: 'no', 'minimal', 'progress', or 'benchmark'."
    )

  if isinstance(device, list) and len(device) != workers:
    raise ValueError(
      f"Value for 'device' is invalid! Device should be a name, or a list with a length that should match number of workers ({workers})."
    )

  devices = device if isinstance(device, list) else [device] * workers

  if model_backend == MODEL_BACKEND_TF:
    for d in devices:
      if "GPU" in d:
        raise ValueError(
          "Value for 'device' is invalid! GPU devices are not supported for TFLite backend! Please use the 'pb' backend instead."
        )

  if show_stats == "benchmark":
    print("Starting benchmark...")
    debug_log = True

  # print("PID", os.getpid())
  track_performance = show_stats in ("progress", "benchmark")

  log_file = Path(Path(tempfile.gettempdir()) / f"{PKG_NAME}.log")

  benchmark_dir: Path | None = None
  benchmark_run_out_dir: Path | None = None
  iso_time = start_timepoint.strftime("%Y%m%dT%H%M%S")
  if show_stats == "benchmark":
    benchmark_dir = get_benchmark_dir(
      model=MODEL_TYPE_ACOUSTIC,
      version=model_version,
      method="embeddings",
    )

    benchmark_run_out_dir = benchmark_dir / f"run-{iso_time}"
    benchmark_run_out_dir.mkdir(parents=True, exist_ok=True)

    log_file = benchmark_run_out_dir / f"log-{iso_time}.log"
    print(f"Writing logs to: {log_file.absolute()}")

  cancel_event = mp.Event()
  processing_finished_event = mp.Event()

  logging_level = get_package_logging_level()
  logging_queue = mp.Queue()
  logging_stop_event = mp.Event()
  logging_listener = threading.Thread(
    target=QueueFileWriter(
      log_queue=logging_queue,
      logging_level=logging_level,
      log_file=log_file,
      cancel_event=cancel_event,
      stop_event=logging_stop_event,
      processing_finished_event=processing_finished_event,
    ),
    name="QueueFileWriter",
    daemon=True,
  )
  logging_listener.start()

  queue_handler = bn_logging.add_queue_handler(logging_queue)

  logger = bn_logging.get_logger(__name__)

  logger.info("Getting input files...")
  parsed_audio_paths = set()
  if isinstance(inp, Path | str):
    inp = (Path(inp),)

  if isinstance(inp, Iterable):
    for inp_audio in inp:
      if isinstance(inp_audio, Path | str):
        inp_path = Path(inp_audio)
        if inp_path.is_file():
          if inp_path.suffix.upper() in SF_FORMATS:
            parsed_audio_paths.add(inp_path.absolute())
          else:
            raise ValueError(
              f"Input file '{inp_path}' is not a supported audio format! Supported formats: {sorted(SF_FORMATS)}."
            )
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

  feeders = min(feeders, n_files)
  logger.debug(f"Using {feeders} producer(s) for {n_files} file(s).")
  logger.info("Starting analysis...")

  # segments_dtype for max file duration:
  # hopsize 3s & overlap 0s: n-segments ÷ 1200
  # ---
  # uint8 = 255 segments = 0 m 12 s
  # uint16 = 65 535 segments = 54 m 36 s
  # uint32 = 4 294 967 295 segments = 2 485 days = 59 652 h
  reserve_n_segments = 0
  segments_dtype = np.dtype(np.uint32)
  if max_audio_duration_min is not None:
    reserve_n_segments = get_max_n_segments(
      max_audio_duration_min * 60, model_segment_size_s, overlap_duration_s
    )
    segments_dtype = uint_dtype_for(max(0, reserve_n_segments - 1))

  segments_code_type = uint_ctype_from_dtype(segments_dtype)
  max_segment_idx_ptr = mp.RawValue(
    segments_code_type,  # type: ignore
    max(0, reserve_n_segments - 1),
  )

  prob_dtype: DTypeLike = np.float16 if half_precision else np.float32

  n_slots = workers + (workers * prefetch_ratio)

  sem_free_slots = mp.Semaphore(n_slots)
  sem_filled_slots = mp.Semaphore(0)
  sem_active_workers = mp.Semaphore(0)
  logger.debug(f"FILL: {sem_filled_slots}, FREE: {sem_free_slots}")

  rf_file_indices = RingField(
    "bn_ring_file_indices",
    dtype=uint_dtype_for(max(0, n_files - 1)),
    shape=(n_slots, batch_size),
  )

  rf_segment_indices = RingField(
    "bn_ring_segment_indices",
    dtype=segments_dtype,
    shape=(n_slots, batch_size),
  )

  model_segment_size_samples = int(model_segment_size_s * model_sample_rate)

  rf_audio_samples = RingField(
    "bn_ring_audio_samples",
    dtype=np.dtype(np.float32),
    shape=(n_slots, batch_size, model_segment_size_samples),
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
  rf_segment_indices.cleanup()
  rf_audio_samples.cleanup()
  rf_batch_sizes.cleanup()
  rf_flags.cleanup()

  result = EmbeddingsTensor(
    n_files,
    n_segments=reserve_n_segments,
    emb_dim=model_emb_dim,
    emb_dtype=prob_dtype,
    segment_indices_dtype=rf_segment_indices.dtype,
    files_dtype=rf_file_indices.dtype,
    max_segment_index=max_segment_idx_ptr,
  )

  worker_queue = mp.Queue()
  prd_ring_access_lock = mp.Lock()
  wkr_ring_access_lock = mp.Lock()
  prd_all_done_event = mp.Event()

  wkr_stats_queue = mp.Queue()
  prd_stats_queue = mp.Queue()
  analyzer_queue = mp.Queue()
  perf_res_queue: mp.Queue | None = None
  perf_result: PerformanceTrackingResult | None = None
  perf_stop_event = mp.Event()
  tot_n_segments_ptr = mp.RawValue(ctypes.c_uint64, 0)
  files_queue = mp.Queue(
    len(file_paths) + feeders
  )  # SimpleQueue would block on put() after few items, but Queue need to be filled before
  for file_idx, file_path in enumerate(file_paths):
    files_queue.put((file_idx, file_path), block=False)
  for _ in range(feeders):
    files_queue.put(None, block=False)
  prod_done_ptr = mp.Value(
    uint_ctype_from_dtype(uint_dtype_for(feeders)),  # type: ignore
    0,
    lock=True,
  )  # type: ignore

  with (
    create_shm_ring(rf_file_indices),
    create_shm_ring(rf_segment_indices),
    create_shm_ring(rf_audio_samples),
    create_shm_ring(rf_batch_sizes),
    create_shm_ring(rf_flags) as shm_ring_flags,
  ):
    logger.debug("Shared memory initialized.")

    flags = rf_flags.get_array(shm_ring_flags)
    flags[:] = WRITABLE_FLAG

    file_analyzer_proc = threading.Thread(
      target=FilesAnalyzer(
        files=file_paths,
        logging_level=logging_level,
        logging_queue=logging_queue,
        segment_duration_s=model_segment_size_s,
        overlap_duration_s=overlap_duration_s,
        max_segment_idx_ptr=max_segment_idx_ptr,
        rf_segment_indices=rf_segment_indices,
        analyzing_result=analyzer_queue,
        tot_n_segments=tot_n_segments_ptr,
        cancel_event=cancel_event,
      ),
      name="FileAnalyzer",
      daemon=True,
    )
    file_analyzer_proc.start()

    producer_processes = [
      mp.Process(
        target=ChildProducer(
          files_queue=files_queue,
          batch_size=batch_size,
          prd_all_done_event=prd_all_done_event,
          n_slots=n_slots,
          prd_ring_access_lock=prd_ring_access_lock,
          track_performance=track_performance,
          prod_stats_queue=prd_stats_queue,
          rf_file_indices=rf_file_indices,
          rf_segment_indices=rf_segment_indices,
          rf_audio_samples=rf_audio_samples,
          rf_batch_sizes=rf_batch_sizes,
          rf_flags=rf_flags,
          logging_queue=logging_queue,
          logging_level=logging_level,
          sem_free_slots=sem_free_slots,
          sem_filled_slots=sem_filled_slots,
          segment_duration_s=model_segment_size_s,
          overlap_duration_s=overlap_duration_s,
          target_sample_rate=model_sample_rate,
          use_bandpass=use_bandpass,
          bandpass_fmax=bandpass_fmax,
          bandpass_fmin=bandpass_fmin,
          fmin=model_sig_fmin,
          fmax=model_sig_fmax,
          max_segment_idx_ptr=max_segment_idx_ptr,
          prod_done_ptr=prod_done_ptr,
          n_prods=feeders,
          cancel_event=cancel_event,
        ),
        name=f"ChildProducer-{i}",
        daemon=True,
      )
      for i in range(feeders)
    ]
    for p in producer_processes:
      p.start()

    if model_backend == MODEL_BACKEND_TF:
      backend_type = TFInferenceBackend
    elif model_backend == MODEL_BACKEND_PB:
      backend_type = PBInferenceBackend
    else:
      raise AssertionError()

    backend_loader = InferenceBackendLoader(
      model_path=model_path,
      backend_type=backend_type,
      backend_kwargs=model_backend_kwargs,
    )

    try:
      backend_loader.on_before_worker_initialized()
    except Exception as exc:
      cancel_event.set()
      logger.error(f"Error during backend initialization: {exc}.")

    worker_processes = [
      mp.Process(
        target=EmbeddingsWorker(
          backend_loader=backend_loader,
          device=devices[i],
          batch_size=batch_size,
          wkr_ring_access_lock=wkr_ring_access_lock,
          n_slots=n_slots,
          segment_duration_samples=model_segment_size_samples,
          out_q=worker_queue,
          logging_queue=logging_queue,
          prd_all_done_event=prd_all_done_event,
          logging_level=logging_level,
          rf_file_indices=rf_file_indices,
          rf_segment_indices=rf_segment_indices,
          rf_audio_samples=rf_audio_samples,
          rf_batch_sizes=rf_batch_sizes,
          rf_flags=rf_flags,
          sem_fill=sem_filled_slots,
          sem_free=sem_free_slots,
          emb_dtype=prob_dtype,
          wkr_stats_queue=wkr_stats_queue,
          track_performance=track_performance,
          cancel_event=cancel_event,
          sem_active_workers=sem_active_workers,
        ),
        name=f"EmbeddingsWorker-{i}",
        daemon=True,
      )
      for i in range(workers)
    ]

    worker_start = time.perf_counter()
    for w in worker_processes:
      w.start()

    perf_tracker = None
    if track_performance:
      perf_res_queue = mp.Queue()
      perf_tracker = mp.Process(
        target=PerformanceTracker(
          pred_dur_queue=wkr_stats_queue,
          stop_event=perf_stop_event,
          processing_finished_event=processing_finished_event,
          update_interval=0.5,
          print_interval=1,
          prod_stats_queue=prd_stats_queue,
          use_stats_from_last_seconds=30,
          n_workers=workers,
          start=start,
          sem_filled_slots=sem_filled_slots,
          workers_start=worker_start,
          segment_size_s=model_segment_size_s,
          logging_queue=logging_queue,
          logging_level=logging_level,
          perf_res=perf_res_queue,
          parent_process_id=os.getpid(),
          rf_flags=rf_flags,
          tot_n_segments_ptr=tot_n_segments_ptr,
          cancel_event=cancel_event,
          sem_active_workers=sem_active_workers,
        ),
        name="PerformanceTracker",
        daemon=True,
      )
      perf_tracker.start()

    consumer = Consumer(
      n_workers=workers,
      worker_queue=worker_queue,
      tensor=result,
      cancel_event=cancel_event,
    )
    consumer()

    processing_finished_event.set()

    file_durations = np.array(cast(list[float], analyzer_queue.get()), dtype=np.float16)
    analyzer_queue.close()
    file_analyzer_proc.join()
    logger.debug("File analyzer finished.")

    for p in producer_processes:
      p.join()
      logger.debug(f"Producer '{p.name}' finished.")
    logger.debug("All producers finished.")

    for w in worker_processes:
      w.join()
      logger.debug(f"Worker '{w.name}' finished.")
    logger.debug("All workers finished.")

    stop = time.perf_counter()
    end_timepoint = datetime.now()

    if track_performance:
      assert perf_tracker is not None
      assert perf_res_queue is not None
      perf_stop_event.set()
      perf_result = cast(PerformanceTrackingResult, perf_res_queue.get())
      perf_tracker.join()
      logger.debug("Performance tracker finished.")

  if cancel_event.is_set():
    logger.error("Analysis was cancelled due to an error.")
    logging_stop_event.set()
    logging_listener.join()
    logging_queue.close()
    logging_queue.join_thread()
    raise RuntimeError(
      f"Analysis was cancelled due to an error. Please check the logs for details: {log_file.absolute()}"
    )

  res = EmbeddingsPredictionResult(
    tensor=result,
    files=file_paths,
    segment_duration_s=model_segment_size_s,
    overlap_duration_s=overlap_duration_s,
    file_durations=file_durations,
  )
  del result

  if show_stats in ("minimal", "progress"):
    bmm = MinimalBenchmarkEmbMeta(
      _start_timepoint=start_timepoint,
      _end_timepoint=end_timepoint,
      _time_wall_time_s=stop - start,
      _file_durations=file_durations,
      mem_result_total_memory_usage_MiB=res.memory_size_mb,
      mem_shm_size_file_indices_MiB=rf_file_indices.nbytes / 1024**2,
      mem_shm_size_segment_indices_MiB=rf_segment_indices.nbytes / 1024**2,
      mem_shm_size_audio_samples_MiB=rf_audio_samples.nbytes / 1024**2,
      mem_shm_size_batch_sizes_MiB=rf_batch_sizes.nbytes / 1024**2,
      mem_shm_size_flags_MiB=rf_flags.nbytes / 1024**2,
      file_segments_total=tot_n_segments_ptr.value,
      model_segment_duration_seconds=model_segment_size_s,
      file_formats=", ".join(sorted({x.suffix[1:].upper() for x in file_paths})),
    )

    summary = (
      f"-------------------------------\n"
      f"----------- Summary -----------\n"
      f"-------------------------------\n"
      f"Start time: {bmm.time_begin}\n"
      f"End time:   {bmm.time_end}\n"
      f"Wall time:  {bmm.time_wall_time}\n"
      f"Input: {bmm.file_count} file(s) ({bmm.file_formats})\n"
      f"  Total duration: {bmm.file_duration_sum}\n"
      f"  Average duration: {bmm.file_duration_average}\n"
      f"  Minimum duration (single file): {bmm.file_duration_minimum}\n"
      f"  Maximum duration (single file): {bmm.file_duration_maximum}\n"
      f"Memory usage:\n"
      f"  Buffer: {bmm.mem_shm_size_total_MiB:.2f} M (shared memory)\n"
      f"  Result: {bmm.mem_result_total_memory_usage_MiB:.2f} M (NumPy)\n"
      f"Performance:\n"
      f"  {bmm.speed_total_xrt:.0f} x real-time (RTF: {bmm.speed_total_rtf:.8f})\n"
      f"  {bmm.speed_total_seg_per_second:.0f} segments/s ({bmm.speed_total_audio_per_second} audio/s)\n"
    )
    print(summary)
  elif show_stats == "benchmark":
    assert perf_result is not None

    logger.info("Benchmarking is enabled. Collecting performance data...")

    bmm = FullBenchmarkEmbMeta(
      _start_timepoint=start_timepoint,
      _end_timepoint=end_timepoint,
      param_producers=feeders,
      param_workers=workers,
      _worker_avg_wall_time_s=perf_result.worker_avg_wall_time_s,
      param_devices=", ".join(device) if isinstance(device, list) else device,
      model_type=MODEL_TYPE_ACOUSTIC,
      model_version=model_version,
      model_is_custom=model_is_custom,
      model_path=str(model_path.absolute()),
      model_species=len(model_species_list),
      model_precision=model_precision,
      _file_durations=file_durations,
      file_segments_maximum=max_segment_idx_ptr.value + 1,
      file_segments_total=tot_n_segments_ptr.value,
      model_segment_duration_seconds=model_segment_size_s,
      param_overlap_seconds=overlap_duration_s,
      param_batch_size=batch_size,
      param_prefetch_ratio=prefetch_ratio,
      mem_shm_ringsize=n_slots,
      param_bandpass_use=use_bandpass,
      param_bandpass_fmin=bandpass_fmin,
      param_bandpass_fmax=bandpass_fmax,
      param_half_precision=half_precision,
      _time_rampup_first_line_s=start_time - psutil.Process(os.getpid()).create_time(),
      _time_wall_time_s=stop - start,
      mem_result_total_memory_usage_MiB=res.memory_size_mb,
      mem_shm_size_file_indices_MiB=rf_file_indices.nbytes / 1024**2,
      mem_shm_size_segment_indices_MiB=rf_segment_indices.nbytes / 1024**2,
      mem_shm_size_audio_samples_MiB=rf_audio_samples.nbytes / 1024**2,
      mem_shm_size_batch_sizes_MiB=rf_batch_sizes.nbytes / 1024**2,
      mem_shm_size_flags_MiB=rf_flags.nbytes / 1024**2,
      # n_usage_recordings=perf_result.n_usage_recordings,
      mem_memory_usage_maximum_MiB=perf_result.max_memory_usages_MiB,
      mem_memory_usage_average_MiB=perf_result.avg_memory_usages_MiB,
      cpu_usage_maximum_pct=perf_result.max_cpu_usages_pct,
      cpu_usage_average_pct=perf_result.avg_cpu_usages_pct,
      mem_shm_slots_average_free=perf_result.avg_free_slots,
      mem_shm_slots_average_busy=perf_result.avg_busy_slots,
      mem_shm_slots_average_buffered=perf_result.avg_preloaded_slots,
      worker_busy_average=perf_result.avg_busy_workers,
      # avg_free_slots_last=perf_result.avg_free_slots_last,
      # avg_filled_slots_last=n_slots - perf_result.avg_free_slots_last,
      # avg_busy_slots_last=perf_result.avg_busy_slots_last,
      # avg_preloaded_slots_last=perf_result.avg_preloaded_slots_last,
      # avg_busy_workers_last=perf_result.avg_busy_workers_last,
      _time_rampup_first_prediction_s=perf_result.ramp_up_time_until_first_pred_s,
      file_batches_processed=perf_result.total_batches_processed,
      speed_worker_xrt=perf_result.worker_speed_xrt,
      speed_worker_xrt_max=perf_result.worker_speed_xrt_max,
      model_backend=model_backend,
      model_sample_rate=model_sample_rate,
      model_sig_fmin=model_sig_fmin,
      model_sig_fmax=model_sig_fmax,
      worker_wait_time_average_milliseconds=perf_result.avg_wait_time_ms,
      file_formats=", ".join(sorted({x.suffix[1:].upper() for x in file_paths})),
      param_inference_library=model_backend_kwargs.get("inference_library"),
    )

    bm = asdict(bmm)
    del_keys = [k for k in bm if k.startswith("_")]
    for k in del_keys:
      del bm[k]
    bm = bmm.to_dict()

    assert benchmark_dir is not None
    assert benchmark_run_out_dir is not None

    meta_df_out = benchmark_dir / "runs.csv"
    stats_out_json = benchmark_run_out_dir / f"stats-{iso_time}.json"
    stats_human_readable_out = benchmark_run_out_dir / f"stats-{iso_time}.txt"
    result_csv = benchmark_run_out_dir / f"result-{iso_time}.csv"
    result_npz = benchmark_run_out_dir / f"result-{iso_time}.npz"

    with open(stats_out_json, "w", encoding="utf8") as f:
      json.dump(bm, f, indent=2, ensure_ascii=False)

    import pandas as pd

    meta_df = pd.DataFrame.from_records([bm])
    meta_df.to_csv(meta_df_out, mode="a", header=not meta_df_out.exists(), index=False)

    summary = (
      f"-------------------------------\n"
      f"------ Benchmark summary ------\n"
      f"-------------------------------\n"
      f"Start time: {bmm.time_begin}\n"
      f"End time:   {bmm.time_end}\n"
      f"Wall time:  {bmm.time_wall_time}\n"
      f"Input: {bmm.file_count} file(s) ({bmm.file_formats})\n"
      f"  Total duration: {bmm.file_duration_sum}\n"
      f"  Average duration: {bmm.file_duration_average}\n"
      f"  Minimum duration (single file): {bmm.file_duration_minimum}\n"
      f"  Maximum duration (single file): {bmm.file_duration_maximum}\n"
      f"Feeder(s): {bmm.param_producers}\n"
      f"Buffer: {bmm.mem_shm_slots_average_filled:.1f}/{n_slots} filled slots (mean)\n"
      f"Busy workers: {bmm.worker_busy_average:.1f}/{bmm.param_workers} (mean)\n"
      f"  Average wait time for next batch: {bmm.worker_wait_time_average_milliseconds:.3f} ms\n"
      # f"\tBusy: {bmm.avg_busy_slots:.1f} slots\n"
      # f"\tPreloaded: {bmm.avg_preloaded_slots:.1f} slots\n"
      # f"\tFree: {bmm.avg_free_slots:.1f} slots\n"
      f"Memory usage:\n"
      f"  Program: {bmm.mem_memory_usage_maximum_MiB:.2f} M (total max)\n"
      f"  Buffer: {bmm.mem_shm_size_total_MiB:.2f} M (shared memory)\n"
      f"  Result: {bmm.mem_result_total_memory_usage_MiB:.2f} M (NumPy)\n"
      f"Performance:\n"
      f"  {bmm.speed_total_xrt:.0f} x real-time (RTF: {bmm.speed_total_rtf:.8f})\n"
      f"  {bmm.speed_total_seg_per_second:.0f} segments/s ({bmm.speed_total_audio_per_second} audio/s)\n"
      f"Worker performance:\n"
      f"  {bmm.speed_worker_xrt:.0f} x real-time (RTF: {bmm.speed_worker_rtf:.8f})\n"
      f"  {bmm.speed_worker_total_seg_per_second:.0f} segments/s ({bmm.speed_worker_total_audio_per_second} audio/s)\n"
    )
    stats_human_readable_out.write_text(summary, encoding="utf8")

    print("Saving result using internal format (.npz)...")
    res.save(result_npz)
    # print("Saving result using CSV format (.csv)...")
    # res.to_csv(result_csv, encoding="utf-8", silent=False)

    summary += (
      f"-------------------------------\n"
      f"Benchmark folder:\n"
      f"  {benchmark_run_out_dir.absolute()}\n"
      f"Statistics results written to:\n"
      f"  {stats_human_readable_out.absolute()}\n"
      f"  {stats_out_json.absolute()}\n"
      f"  {meta_df_out.absolute()}\n"
      f"Prediction results written to:\n"
      f"  {result_npz.absolute()}\n"
      # f"  {result_csv.absolute()}\n"
      f"Log file written to:\n"
      f"  {log_file.absolute()}\n"
    )
    print(summary)

  logging_stop_event.set()
  logging_listener.join()
  logging_queue.close()
  logging_queue.join_thread()
  bn_logging.remove_queue_handler(queue_handler)

  global_log_file_iso = Path(Path(tempfile.gettempdir()) / f"{PKG_NAME}-{iso_time}.log")
  shutil.copyfile(log_file, global_log_file_iso)
  return res


def predict_species_from_recordings(
  inp: Path | str | Iterable[Path | str],
  model_species_list: OrderedSet[str],
  model_path: Path,
  model_backend: MODEL_BACKENDS,
  model_backend_kwargs: dict,
  model_version: ACOUSTIC_MODEL_VERSIONS,
  model_segment_size_s: float,
  model_sample_rate: int,
  model_sig_fmin: int,
  model_sig_fmax: int,
  model_precision: MODEL_PRECISIONS,
  model_is_custom: bool,
  top_k: int | None = 5,
  feeders: int = 1,
  workers: int = 4,
  batch_size: int = 1,
  prefetch_ratio: int = 1,
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
  show_stats: Literal["no", "minimal", "progress", "benchmark"] = "no",
  device: str | list[str] = "CPU",
) -> PredictionResult:
  debug_log = False

  start = time.perf_counter()
  start_time = time.time()
  start_timepoint = datetime.now()

  if not batch_size >= 1:
    raise ValueError(
      "Value for 'batch_size' is invalid! It needs to be larger than or equal to 1."
    )

  # if (
  #   default_confidence_threshold is not None
  #   and not 0 <= default_confidence_threshold < 1.0
  # ):
  #   raise ValueError(
  #     "Value for 'min_confidence' is invalid! It needs to be None or in interval [0.0, 1.0)."
  #   )

  if not feeders >= 1:
    raise ValueError(
      "Value for 'feeders' is invalid! It needs to be larger than or equal to 1."
    )

  if not workers >= 1:
    raise ValueError(
      "Value for 'workers' is invalid! It needs to be larger than or equal to 1."
    )

  if not prefetch_ratio >= 0:
    raise ValueError(
      "Value for 'prefetch_ratio' is invalid! It needs to be larger than or equal to 0."
    )

  if not 0 <= overlap_duration_s < 3:
    raise ValueError(
      "Value for 'overlap_duration_s' is invalid! It needs to be in interval [0.0, 3.0)."
    )

  if apply_sigmoid:
    if sigmoid_sensitivity is None:
      raise ValueError(
        "Value for 'sigmoid_sensitivity' is required if 'apply_sigmoid==True'!"
      )
    if not 0.5 <= sigmoid_sensitivity <= 1.5:
      raise ValueError(
        "Value for 'sigmoid_sensitivity' is invalid! It needs to be in interval [0.5, 1.5]."
      )

  if use_bandpass:
    if bandpass_fmin is None:
      raise ValueError("Value for 'bandpass_fmin' is required if 'use_bandpass==True'!")
    if bandpass_fmax is None:
      raise ValueError("Value for 'bandpass_fmax' is required if 'use_bandpass==True'!")

    if bandpass_fmin < 0:
      raise ValueError(
        "Value for 'bandpass_fmin' is invalid! It needs to be larger than zero."
      )

    if bandpass_fmax <= bandpass_fmin:
      raise ValueError(
        "Value for 'bandpass_fmax' is invalid! It needs to be larger than 'bandpass_fmin'."
      )

  if max_audio_duration_min is not None and not max_audio_duration_min > 0:
    raise ValueError(
      "Value for 'max_audio_duration_min' is invalid! It needs to be either None, or larger than zero."
    )

  if show_stats not in ("no", "minimal", "progress", "benchmark"):
    raise ValueError(
      f"Value for 'show_stats' is invalid! It needs to be one of: 'no', 'minimal', 'progress', or 'benchmark'."
    )

  if isinstance(device, list) and len(device) != workers:
    raise ValueError(
      f"Value for 'device' is invalid! Device should be a name, or a list with a length that should match number of workers ({workers})."
    )

  devices = device if isinstance(device, list) else [device] * workers

  if model_backend == MODEL_BACKEND_TF:
    for d in devices:
      if "GPU" in d:
        raise ValueError(
          "Value for 'device' is invalid! GPU devices are not supported for TFLite backend! Please use the 'pb' backend instead."
        )

  if custom_species_list is not None:
    for i, species_name in enumerate(custom_species_list):
      if species_name not in model_species_list:
        raise ValueError(
          f"Value for 'custom_species_list' is invalid! Species '{species_name}' is not in the model's species list!"
        )

  if custom_confidence_thresholds is not None and custom_confidence_thresholds:
    for species_name, threshold in custom_confidence_thresholds.items():
      if species_name not in model_species_list:
        raise ValueError(
          f"Value for 'custom_confidence_thresholds' is invalid! Species '{species_name}' is not in the model's species list!"
        )

  if top_k is not None and top_k > len(model_species_list):
    raise ValueError(
      f"top_k cannot be larger than the number of species ({len(model_species_list)})."
    )

  if top_k is None:
    top_k = len(model_species_list)

  if show_stats == "benchmark":
    print("Starting benchmark...")
    debug_log = True

  # print("PID", os.getpid())
  track_performance = show_stats in ("progress", "benchmark")

  log_file = Path(Path(tempfile.gettempdir()) / f"{PKG_NAME}.log")

  benchmark_dir: Path | None = None
  benchmark_run_out_dir: Path | None = None
  iso_time = start_timepoint.strftime("%Y%m%dT%H%M%S")
  if show_stats == "benchmark":
    benchmark_dir = get_benchmark_dir(
      model=MODEL_TYPE_ACOUSTIC,
      version=model_version,
      method="scores",
    )

    benchmark_run_out_dir = benchmark_dir / f"run-{iso_time}"
    benchmark_run_out_dir.mkdir(parents=True, exist_ok=True)

    log_file = benchmark_run_out_dir / f"log-{iso_time}.log"
    print(f"Writing logs to: {log_file.absolute()}")

  cancel_event = mp.Event()
  processing_finished_event = mp.Event()

  logging_level = get_package_logging_level()
  logging_queue = mp.Queue()
  logging_stop_event = mp.Event()
  logging_listener = threading.Thread(
    target=QueueFileWriter(
      log_queue=logging_queue,
      logging_level=logging_level,
      log_file=log_file,
      cancel_event=cancel_event,
      stop_event=logging_stop_event,
      processing_finished_event=processing_finished_event,
    ),
    name="QueueFileWriter",
    daemon=True,
  )
  logging_listener.start()

  queue_handler = bn_logging.add_queue_handler(logging_queue)

  logger = bn_logging.get_logger(__name__)

  logger.info("Getting input files...")
  parsed_audio_paths = set()
  if isinstance(inp, Path | str):
    inp = (Path(inp),)

  if isinstance(inp, Iterable):
    for inp_audio in inp:
      if isinstance(inp_audio, Path | str):
        inp_path = Path(inp_audio)
        if inp_path.is_file():
          if inp_path.suffix.upper() in SF_FORMATS:
            parsed_audio_paths.add(inp_path.absolute())
          else:
            raise ValueError(
              f"Input file '{inp_path}' is not a supported audio format! Supported formats: {sorted(SF_FORMATS)}."
            )
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

  feeders = min(feeders, n_files)
  logger.debug(f"Using {feeders} producer(s) for {n_files} file(s).")
  logger.info("Starting analysis...")

  n_species = len(model_species_list)
  species_whitelist: np.ndarray
  if custom_species_list is not None and len(custom_species_list) > 0:
    species_ids_whitelist = np.empty(len(custom_species_list), dtype=int)
    for i, species_name in enumerate(custom_species_list):
      assert species_name in model_species_list
      species_id = model_species_list.index(species_name)
      species_ids_whitelist[i] = species_id

    species_whitelist = np.full(n_species, fill_value=False, dtype=bool)
    species_whitelist[species_ids_whitelist] = True
  else:
    species_whitelist = np.full(n_species, fill_value=True, dtype=bool)
  species_whitelist.setflags(write=False)

  # Thresholds
  if default_confidence_threshold is None:
    default_confidence_threshold = -np.inf
  thresholds = np.full(n_species, default_confidence_threshold, np.float32)

  if custom_confidence_thresholds:
    for species_name, threshold in custom_confidence_thresholds.items():
      assert species_name in model_species_list
      species_id = model_species_list.index(species_name)
      thresholds[species_id] = threshold
  thresholds.setflags(write=False)

  # segments_dtype for max file duration:
  # hopsize 3s & overlap 0s: n-segments ÷ 1200
  # ---
  # uint8 = 255 segments = 0 m 12 s
  # uint16 = 65 535 segments = 54 m 36 s
  # uint32 = 4 294 967 295 segments = 2 485 days = 59 652 h
  reserve_n_segments = 0
  segments_dtype = np.dtype(np.uint32)
  if max_audio_duration_min is not None:
    reserve_n_segments = get_max_n_segments(
      max_audio_duration_min * 60, model_segment_size_s, overlap_duration_s
    )
    segments_dtype = uint_dtype_for(max(0, reserve_n_segments - 1))

  segments_code_type = uint_ctype_from_dtype(segments_dtype)
  max_segment_idx_ptr = mp.RawValue(
    segments_code_type,  # type: ignore
    max(0, reserve_n_segments - 1),
  )

  prob_dtype: DTypeLike = np.float16 if half_precision else np.float32

  n_species = n_species

  n_slots = workers + (workers * prefetch_ratio)

  sem_free_slots = mp.Semaphore(n_slots)
  sem_filled_slots = mp.Semaphore(0)
  sem_active_workers = mp.Semaphore(0)
  logger.debug(f"FILL: {sem_filled_slots}, FREE: {sem_free_slots}")

  rf_file_indices = RingField(
    "bn_ring_file_indices",
    dtype=uint_dtype_for(max(0, n_files - 1)),
    shape=(n_slots, batch_size),
  )

  rf_segment_indices = RingField(
    "bn_ring_segment_indices",
    dtype=segments_dtype,
    shape=(n_slots, batch_size),
  )

  model_segment_size_samples = int(model_segment_size_s * model_sample_rate)

  rf_audio_samples = RingField(
    "bn_ring_audio_samples",
    dtype=np.dtype(np.float32),
    shape=(n_slots, batch_size, model_segment_size_samples),
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
  rf_segment_indices.cleanup()
  rf_audio_samples.cleanup()
  rf_batch_sizes.cleanup()
  rf_flags.cleanup()

  result = ScoresTensor(
    n_files,
    n_segments=reserve_n_segments,
    top_k=top_k,
    n_species=n_species,
    prob_dtype=prob_dtype,
    segment_indices_dtype=rf_segment_indices.dtype,
    files_dtype=rf_file_indices.dtype,
    max_segment_index=max_segment_idx_ptr,
  )

  species_blacklist = ~species_whitelist[np.newaxis, :]
  species_blacklist.setflags(write=False)
  species_thresholds = thresholds[np.newaxis, :]
  species_thresholds.setflags(write=False)
  worker_queue = mp.Queue()
  prd_ring_access_lock = mp.Lock()
  wkr_ring_access_lock = mp.Lock()
  prd_all_done_event = mp.Event()

  wkr_stats_queue = mp.Queue()
  prd_stats_queue = mp.Queue()
  analyzer_queue = mp.Queue()
  perf_res_queue: mp.Queue | None = None
  perf_result: PerformanceTrackingResult | None = None
  perf_stop_event = mp.Event()
  tot_n_segments_ptr = mp.RawValue(ctypes.c_uint64, 0)
  files_queue = mp.Queue(
    len(file_paths) + feeders
  )  # SimpleQueue would block on put() after few items, but Queue need to be filled before
  for file_idx, file_path in enumerate(file_paths):
    files_queue.put((file_idx, file_path), block=False)
  for _ in range(feeders):
    files_queue.put(None, block=False)
  prod_done_ptr = mp.Value(
    uint_ctype_from_dtype(uint_dtype_for(feeders)),  # type: ignore
    0,
    lock=True,
  )  # type: ignore

  with (
    create_shm_ring(rf_file_indices),
    create_shm_ring(rf_segment_indices),
    create_shm_ring(rf_audio_samples),
    create_shm_ring(rf_batch_sizes),
    create_shm_ring(rf_flags) as shm_ring_flags,
  ):
    logger.debug("Shared memory initialized.")

    flags = rf_flags.get_array(shm_ring_flags)
    flags[:] = WRITABLE_FLAG

    file_analyzer_proc = threading.Thread(
      target=FilesAnalyzer(
        files=file_paths,
        logging_level=logging_level,
        logging_queue=logging_queue,
        segment_duration_s=model_segment_size_s,
        overlap_duration_s=overlap_duration_s,
        max_segment_idx_ptr=max_segment_idx_ptr,
        rf_segment_indices=rf_segment_indices,
        analyzing_result=analyzer_queue,
        tot_n_segments=tot_n_segments_ptr,
        cancel_event=cancel_event,
      ),
      name="FileAnalyzer",
      daemon=True,
    )
    file_analyzer_proc.start()

    producer_processes = [
      mp.Process(
        target=ChildProducer(
          files_queue=files_queue,
          batch_size=batch_size,
          prd_all_done_event=prd_all_done_event,
          n_slots=n_slots,
          prd_ring_access_lock=prd_ring_access_lock,
          track_performance=track_performance,
          prod_stats_queue=prd_stats_queue,
          rf_file_indices=rf_file_indices,
          rf_segment_indices=rf_segment_indices,
          rf_audio_samples=rf_audio_samples,
          rf_batch_sizes=rf_batch_sizes,
          rf_flags=rf_flags,
          logging_queue=logging_queue,
          logging_level=logging_level,
          sem_free_slots=sem_free_slots,
          sem_filled_slots=sem_filled_slots,
          segment_duration_s=model_segment_size_s,
          overlap_duration_s=overlap_duration_s,
          target_sample_rate=model_sample_rate,
          use_bandpass=use_bandpass,
          bandpass_fmax=bandpass_fmax,
          bandpass_fmin=bandpass_fmin,
          fmax=model_sig_fmax,
          fmin=model_sig_fmin,
          max_segment_idx_ptr=max_segment_idx_ptr,
          prod_done_ptr=prod_done_ptr,
          n_prods=feeders,
          cancel_event=cancel_event,
        ),
        name=f"ChildProducer-{i}",
        daemon=True,
      )
      for i in range(feeders)
    ]
    for p in producer_processes:
      p.start()

    if model_backend == MODEL_BACKEND_TF:
      backend_type = TFInferenceBackend
    elif model_backend == MODEL_BACKEND_PB:
      backend_type = PBInferenceBackend
    else:
      raise AssertionError()

    backend_loader = InferenceBackendLoader(
      model_path=model_path,
      backend_type=backend_type,
      backend_kwargs=model_backend_kwargs,
    )

    try:
      backend_loader.on_before_worker_initialized()
    except Exception as exc:
      cancel_event.set()
      logger.error(f"Error during backend initialization: {exc}.")

    worker_processes = [
      mp.Process(
        target=ChildWorker(
          backend_loader=backend_loader,
          device=devices[i],
          top_k=top_k,
          species_thresholds=species_thresholds,
          species_blacklist=species_blacklist,
          batch_size=batch_size,
          wkr_ring_access_lock=wkr_ring_access_lock,
          n_slots=n_slots,
          segment_duration_samples=model_segment_size_samples,
          out_q=worker_queue,
          logging_queue=logging_queue,
          prd_all_done_event=prd_all_done_event,
          logging_level=logging_level,
          rf_file_indices=rf_file_indices,
          rf_segment_indices=rf_segment_indices,
          rf_audio_samples=rf_audio_samples,
          rf_batch_sizes=rf_batch_sizes,
          rf_flags=rf_flags,
          sem_fill=sem_filled_slots,
          sem_free=sem_free_slots,
          apply_sigmoid=apply_sigmoid,
          prob_dtype=prob_dtype,
          sigmoid_sensitivity=sigmoid_sensitivity,
          wkr_stats_queue=wkr_stats_queue,
          track_performance=track_performance,
          cancel_event=cancel_event,
          sem_active_workers=sem_active_workers,
        ),
        name=f"ChildWorker-{i}",
        daemon=True,
      )
      for i in range(workers)
    ]

    worker_start = time.perf_counter()
    for w in worker_processes:
      w.start()

    perf_tracker = None
    if track_performance:
      perf_res_queue = mp.Queue()
      perf_tracker = mp.Process(
        target=PerformanceTracker(
          pred_dur_queue=wkr_stats_queue,
          stop_event=perf_stop_event,
          processing_finished_event=processing_finished_event,
          update_interval=0.5,
          print_interval=1,
          prod_stats_queue=prd_stats_queue,
          use_stats_from_last_seconds=30,
          n_workers=workers,
          start=start,
          sem_filled_slots=sem_filled_slots,
          workers_start=worker_start,
          segment_size_s=model_segment_size_s,
          logging_queue=logging_queue,
          logging_level=logging_level,
          perf_res=perf_res_queue,
          parent_process_id=os.getpid(),
          rf_flags=rf_flags,
          tot_n_segments_ptr=tot_n_segments_ptr,
          cancel_event=cancel_event,
          sem_active_workers=sem_active_workers,
        ),
        name="PerformanceTracker",
        daemon=True,
      )
      perf_tracker.start()

    consumer = Consumer(
      n_workers=workers,
      worker_queue=worker_queue,
      tensor=result,
      cancel_event=cancel_event,
    )
    consumer()

    processing_finished_event.set()

    file_durations = np.array(cast(list[float], analyzer_queue.get()), dtype=np.float16)
    analyzer_queue.close()
    file_analyzer_proc.join()
    logger.debug("File analyzer finished.")

    for p in producer_processes:
      p.join()
      logger.debug(f"Producer '{p.name}' finished.")
    logger.debug("All producers finished.")

    for w in worker_processes:
      w.join()
      logger.debug(f"Worker '{w.name}' finished.")
    logger.debug("All workers finished.")

    stop = time.perf_counter()
    end_timepoint = datetime.now()

    if track_performance:
      assert perf_tracker is not None
      assert perf_res_queue is not None
      perf_stop_event.set()
      perf_result = cast(PerformanceTrackingResult, perf_res_queue.get())
      perf_tracker.join()
      logger.debug("Performance tracker finished.")

  if cancel_event.is_set():
    logger.error("Analysis was cancelled due to an error.")
    logging_stop_event.set()
    logging_listener.join()
    logging_queue.close()
    logging_queue.join_thread()
    raise RuntimeError(
      f"Analysis was cancelled due to an error. Please check the logs for details: {log_file.absolute()}"
    )

  res = PredictionResult(
    tensor=result,
    files=file_paths,
    segment_duration_s=model_segment_size_s,
    overlap_duration_s=overlap_duration_s,
    species_list=model_species_list,
    file_durations=file_durations,
  )
  del result

  if show_stats in ("minimal", "progress"):
    bmm = MinimalBenchmarkMeta(
      _start_timepoint=start_timepoint,
      _end_timepoint=end_timepoint,
      _time_wall_time_s=stop - start,
      _file_durations=file_durations,
      mem_result_total_memory_usage_MiB=res.memory_size_mb,
      mem_shm_size_file_indices_MiB=rf_file_indices.nbytes / 1024**2,
      mem_shm_size_segment_indices_MiB=rf_segment_indices.nbytes / 1024**2,
      mem_shm_size_audio_samples_MiB=rf_audio_samples.nbytes / 1024**2,
      mem_shm_size_batch_sizes_MiB=rf_batch_sizes.nbytes / 1024**2,
      mem_shm_size_flags_MiB=rf_flags.nbytes / 1024**2,
      file_segments_total=tot_n_segments_ptr.value,
      model_segment_duration_seconds=model_segment_size_s,
      file_formats=", ".join(sorted({x.suffix[1:].upper() for x in file_paths})),
    )

    summary = (
      f"-------------------------------\n"
      f"----------- Summary -----------\n"
      f"-------------------------------\n"
      f"Start time: {bmm.time_begin}\n"
      f"End time:   {bmm.time_end}\n"
      f"Wall time:  {bmm.time_wall_time}\n"
      f"Input: {bmm.file_count} file(s) ({bmm.file_formats})\n"
      f"  Total duration: {bmm.file_duration_sum}\n"
      f"  Average duration: {bmm.file_duration_average}\n"
      f"  Minimum duration (single file): {bmm.file_duration_minimum}\n"
      f"  Maximum duration (single file): {bmm.file_duration_maximum}\n"
      f"Memory usage:\n"
      f"  Buffer: {bmm.mem_shm_size_total_MiB:.2f} M (shared memory)\n"
      f"  Result: {bmm.mem_result_total_memory_usage_MiB:.2f} M (NumPy)\n"
      f"Performance:\n"
      f"  {bmm.speed_total_xrt:.0f} x real-time (RTF: {bmm.speed_total_rtf:.8f})\n"
      f"  {bmm.speed_total_seg_per_second:.0f} segments/s ({bmm.speed_total_audio_per_second} audio/s)\n"
    )
    print(summary)
  elif show_stats == "benchmark":
    assert perf_result is not None

    logger.info("Benchmarking is enabled. Collecting performance data...")

    bmm = FullBenchmarkMeta(
      _start_timepoint=start_timepoint,
      _end_timepoint=end_timepoint,
      param_producers=feeders,
      param_workers=workers,
      _worker_avg_wall_time_s=perf_result.worker_avg_wall_time_s,
      param_devices=", ".join(device) if isinstance(device, list) else device,
      model_type=MODEL_TYPE_ACOUSTIC,
      model_version=model_version,
      model_is_custom=model_is_custom,
      model_path=str(model_path.absolute()),
      model_species=n_species,
      model_precision=model_precision,
      _file_durations=file_durations,
      file_segments_maximum=max_segment_idx_ptr.value + 1,
      file_segments_total=tot_n_segments_ptr.value,
      model_segment_duration_seconds=model_segment_size_s,
      param_overlap_seconds=overlap_duration_s,
      param_batch_size=batch_size,
      param_top_k=top_k,
      param_prefetch_ratio=prefetch_ratio,
      mem_shm_ringsize=n_slots,
      param_sigmoid_apply=apply_sigmoid,
      param_sigmoid_sensitivity=sigmoid_sensitivity if apply_sigmoid else None,
      param_bandpass_use=use_bandpass,
      param_bandpass_fmin=bandpass_fmin,
      param_bandpass_fmax=bandpass_fmax,
      param_half_precision=half_precision,
      param_confidence_threshold_default=default_confidence_threshold,
      param_custom_species=len(custom_species_list) if custom_species_list else 0,
      param_confidence_threshold_custom=(
        len(custom_confidence_thresholds) if custom_confidence_thresholds else 0
      ),
      _time_rampup_first_line_s=start_time - psutil.Process(os.getpid()).create_time(),
      _time_wall_time_s=stop - start,
      mem_result_total_memory_usage_MiB=res.memory_size_mb,
      mem_shm_size_file_indices_MiB=rf_file_indices.nbytes / 1024**2,
      mem_shm_size_segment_indices_MiB=rf_segment_indices.nbytes / 1024**2,
      mem_shm_size_audio_samples_MiB=rf_audio_samples.nbytes / 1024**2,
      mem_shm_size_batch_sizes_MiB=rf_batch_sizes.nbytes / 1024**2,
      mem_shm_size_flags_MiB=rf_flags.nbytes / 1024**2,
      # n_usage_recordings=perf_result.n_usage_recordings,
      mem_memory_usage_maximum_MiB=perf_result.max_memory_usages_MiB,
      mem_memory_usage_average_MiB=perf_result.avg_memory_usages_MiB,
      cpu_usage_maximum_pct=perf_result.max_cpu_usages_pct,
      cpu_usage_average_pct=perf_result.avg_cpu_usages_pct,
      mem_shm_slots_average_free=perf_result.avg_free_slots,
      mem_shm_slots_average_busy=perf_result.avg_busy_slots,
      mem_shm_slots_average_buffered=perf_result.avg_preloaded_slots,
      worker_busy_average=perf_result.avg_busy_workers,
      # avg_free_slots_last=perf_result.avg_free_slots_last,
      # avg_filled_slots_last=n_slots - perf_result.avg_free_slots_last,
      # avg_busy_slots_last=perf_result.avg_busy_slots_last,
      # avg_preloaded_slots_last=perf_result.avg_preloaded_slots_last,
      # avg_busy_workers_last=perf_result.avg_busy_workers_last,
      _time_rampup_first_prediction_s=perf_result.ramp_up_time_until_first_pred_s,
      file_batches_processed=perf_result.total_batches_processed,
      speed_worker_xrt=perf_result.worker_speed_xrt,
      speed_worker_xrt_max=perf_result.worker_speed_xrt_max,
      model_backend=model_backend,
      model_sample_rate=model_sample_rate,
      model_sig_fmin=model_sig_fmin,
      model_sig_fmax=model_sig_fmax,
      worker_wait_time_average_milliseconds=perf_result.avg_wait_time_ms,
      file_formats=", ".join(sorted({x.suffix[1:].upper() for x in file_paths})),
      param_inference_library=model_backend_kwargs.get("inference_library"),
    )

    bm = asdict(bmm)
    del_keys = [k for k in bm if k.startswith("_")]
    for k in del_keys:
      del bm[k]
    bm = bmm.to_dict()

    assert benchmark_dir is not None
    assert benchmark_run_out_dir is not None

    meta_df_out = benchmark_dir / "runs.csv"
    stats_out_json = benchmark_run_out_dir / f"stats-{iso_time}.json"
    stats_human_readable_out = benchmark_run_out_dir / f"stats-{iso_time}.txt"
    result_csv = benchmark_run_out_dir / f"result-{iso_time}.csv"
    result_npz = benchmark_run_out_dir / f"result-{iso_time}.npz"

    with open(stats_out_json, "w", encoding="utf8") as f:
      json.dump(bm, f, indent=2, ensure_ascii=False)

    import pandas as pd

    meta_df = pd.DataFrame.from_records([bm])
    meta_df.to_csv(meta_df_out, mode="a", header=not meta_df_out.exists(), index=False)

    summary = (
      f"-------------------------------\n"
      f"------ Benchmark summary ------\n"
      f"-------------------------------\n"
      f"Start time: {bmm.time_begin}\n"
      f"End time:   {bmm.time_end}\n"
      f"Wall time:  {bmm.time_wall_time}\n"
      f"Input: {bmm.file_count} file(s) ({bmm.file_formats})\n"
      f"  Total duration: {bmm.file_duration_sum}\n"
      f"  Average duration: {bmm.file_duration_average}\n"
      f"  Minimum duration (single file): {bmm.file_duration_minimum}\n"
      f"  Maximum duration (single file): {bmm.file_duration_maximum}\n"
      f"Feeder(s): {bmm.param_producers}\n"
      f"Buffer: {bmm.mem_shm_slots_average_filled:.1f}/{n_slots} filled slots (mean)\n"
      f"Busy workers: {bmm.worker_busy_average:.1f}/{bmm.param_workers} (mean)\n"
      f"  Average wait time for next batch: {bmm.worker_wait_time_average_milliseconds:.3f} ms\n"
      # f"\tBusy: {bmm.avg_busy_slots:.1f} slots\n"
      # f"\tPreloaded: {bmm.avg_preloaded_slots:.1f} slots\n"
      # f"\tFree: {bmm.avg_free_slots:.1f} slots\n"
      f"Memory usage:\n"
      f"  Program: {bmm.mem_memory_usage_maximum_MiB:.2f} M (total max)\n"
      f"  Buffer: {bmm.mem_shm_size_total_MiB:.2f} M (shared memory)\n"
      f"  Result: {bmm.mem_result_total_memory_usage_MiB:.2f} M (NumPy)\n"
      f"Performance:\n"
      f"  {bmm.speed_total_xrt:.0f} x real-time (RTF: {bmm.speed_total_rtf:.8f})\n"
      f"  {bmm.speed_total_seg_per_second:.0f} segments/s ({bmm.speed_total_audio_per_second} audio/s)\n"
      f"Worker performance:\n"
      f"  {bmm.speed_worker_xrt:.0f} x real-time (RTF: {bmm.speed_worker_rtf:.8f})\n"
      f"  {bmm.speed_worker_total_seg_per_second:.0f} segments/s ({bmm.speed_worker_total_audio_per_second} audio/s)\n"
    )
    stats_human_readable_out.write_text(summary, encoding="utf8")

    print("Saving result using internal format (.npz)...")
    res.save(result_npz)
    print("Saving result using CSV format (.csv)...")
    res.to_csv(result_csv, encoding="utf-8", silent=False)

    summary += (
      f"-------------------------------\n"
      f"Benchmark folder:\n"
      f"  {benchmark_run_out_dir.absolute()}\n"
      f"Statistics results written to:\n"
      f"  {stats_human_readable_out.absolute()}\n"
      f"  {stats_out_json.absolute()}\n"
      f"  {meta_df_out.absolute()}\n"
      f"Prediction results written to:\n"
      f"  {result_npz.absolute()}\n"
      f"  {result_csv.absolute()}\n"
      f"Log file written to:\n"
      f"  {log_file.absolute()}\n"
    )
    print(summary)

  logging_stop_event.set()
  logging_listener.join()
  logging_queue.close()
  logging_queue.join_thread()
  bn_logging.remove_queue_handler(queue_handler)

  global_log_file_iso = Path(Path(tempfile.gettempdir()) / f"{PKG_NAME}-{iso_time}.log")
  shutil.copyfile(log_file, global_log_file_iso)
  return res
