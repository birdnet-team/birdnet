import json
import logging
import multiprocessing
import os
import platform
import sys
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import cast

import numpy as np
import psutil

import birdnet
import birdnet.model_loader
from birdnet.acoustic.inference.core.perf_tracker import AcousticProgressStats
from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticPredictionResultBase,
)
from birdnet.acoustic.models.base import AcousticModelBase
from birdnet.core.backends import litert_installed, tf_installed
from birdnet.globals import (
  ACOUSTIC_MODEL_VERSION_V2_4,
  LIBRARY_TFLITE,
  LIBRARY_TYPES,
  MODEL_BACKEND_PB,
  MODEL_BACKEND_TF,
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
  MODEL_TYPE_ACOUSTIC,
  VALID_LIBRARY_TYPES,
  VALID_MODEL_BACKENDS,
  VALID_MODEL_PRECISIONS,
)
from birdnet.model_loader import load_perch_v2
from birdnet.utils.local_data import get_benchmark_dir, get_package_version
from birdnet.utils.logging_utils import get_package_logger
from birdnet_benchmark.argparse_helper import (
  ConvertToSetAction,
  parse_float,
  parse_non_empty_or_whitespace,
  parse_non_negative_integer,
  parse_path,
  parse_positive_float,
  parse_positive_integer,
)

APPLY_SIGMOID = True
SIGMOID_SENSITIVITY = 1.0
CUSTOM_SPECIES = None


def run_benchmark() -> None:
  args: list[str] = sys.argv[1:]
  run_benchmark_from_args(args)


def run_benchmark_from_args(args: list[str]) -> None:
  # faulthandler.enable(file=sys.stderr, all_threads=True)
  logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s (%(levelname)s): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
  )

  root = get_package_logger()
  root.setLevel(logging.DEBUG)

  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

  parser = ArgumentParser(
    # formatter_class=argparse.ArgumentDefaultsHelpFormatter(prod, max_help_position=40)
  )

  parser.add_argument(
    "inputs",
    type=parse_path,
    nargs="+",
    metavar="FILE_OR_FOLDER",
    help="input files/folders",
    action=ConvertToSetAction,
  )

  parser.add_argument(
    "-b",
    "--backend",
    type=str,
    choices=VALID_MODEL_BACKENDS,
    metavar="BACKEND",
    help=f"use this backend (default: {MODEL_BACKEND_TF})",
    default=MODEL_BACKEND_TF,
  )

  parser.add_argument(
    "--tf-library",
    type=str,
    choices=VALID_LIBRARY_TYPES,
    metavar="TF-LIBRARY",
    help=f"use this tensorflow library (default: {LIBRARY_TFLITE})",
    default=LIBRARY_TFLITE,
  )

  parser.add_argument(
    "--precision",
    type=str,
    choices=VALID_MODEL_PRECISIONS,
    metavar="PRECISION",
    help=f"model precision (default: {MODEL_PRECISION_FP32})",
    default=MODEL_PRECISION_FP32,
  )

  parser.add_argument(
    "-p",
    "--producers",
    type=parse_positive_integer,
    metavar="PRODUCERS",
    help="number of producers which will read the input files (default: 1)",
    default=1,
  )

  parser.add_argument(
    "-w",
    "--workers",
    type=parse_positive_integer,
    metavar="WORKERS",
    help=(
      "number of workers to use for processing (default: number of physical CPU cores)"
    ),
    default=psutil.cpu_count(logical=False) or 4,
  )

  parser.add_argument(
    "-k",
    "--top-k",
    type=parse_positive_integer,
    metavar="K",
    help="number of top K species to return for each audio segment (default: 5)",
    default=5,
  )

  parser.add_argument(
    "-s",
    "--batch-size",
    type=parse_positive_integer,
    metavar="BATCH-SIZE",
    help="number of top species to return for each audio segment (default: 1)",
    default=1,
  )

  parser.add_argument(
    "--half-precision",
    action="store_true",
    help=(
      "use half-precision (float16) for model inference (slower but uses less memory)"
    ),
    default=False,
  )

  parser.add_argument(
    "-d",
    "--devices",
    type=parse_non_empty_or_whitespace,
    nargs="+",
    metavar="DEVICE",
    help=(
      "device(s) to use for processing (only available for the Protobuf backend), e.g.,"
      " 'CPU', 'GPU', 'GPU:0', 'GPU:1', ...,  (default: 'CPU'); either string or list"
      " of strings, latter is useful for multi-GPU setups, the first GPU will be used "
      "for the first producer, the second for the second producer, etc."
    ),
    default=["CPU"],
  )

  parser.add_argument(
    "-o",
    "--overlap",
    type=parse_float,
    metavar="OVERLAP",
    help="overlap duration in seconds for audio segments (default: 0, no overlap)",
    default=0.0,
  )

  parser.add_argument(
    "-c",
    "--confidence",
    type=parse_float,
    metavar="THRESHOLD",
    help="default confidence threshold for species detection (default: 0.1)",
    default=0.1,
  )

  parser.add_argument(
    "--prefetch-ratio",
    type=parse_non_negative_integer,
    metavar="RATIO",
    help=(
      "amount of additional buffer capacity to keep ahead of the workers, expressed "
      "as a ratio of the amount of workers, i.e., 0 means no prefetching, 1 means one "
      "additional slot per worker, 2 means two additional slots per worker, etc. "
      "(default: 1)"
    ),
    default=1,
  )

  parser.add_argument(
    "--speed",
    type=parse_positive_float,
    metavar="SPEED",
    help="speed factor for audio processing (default: 1.0)",
    default=1.0,
  )

  parser.add_argument(
    "--show-stats",
    type=str,
    choices=["no", "minimal", "progress", "benchmark"],
    metavar="SHOW_STATS",
    help=(
      "show statistics during processing; 'no' means no statistics, 'minimal' means "
      "only minimal statistics, 'progress' means progress bar and minimal statistics, "
      "'benchmark' means progress bar and detailed statistics (default: 'benchmark')"
    ),
    default="benchmark",
  )

  parser.add_argument(
    "--use-perch",
    action="store_true",
    help="use the Perch v2 model for benchmarking instead of the Acoustic v2.4 model "
    "[only available with the Protobuf backend]",
    default=False,
  )

  ns: Namespace = parser.parse_args(args)
  run_benchmark_from_ns(ns)


@dataclass
class BenchmarkResultContainer:
  ns: Namespace
  model: AcousticModelBase
  output: AcousticPredictionResultBase | None = None
  stats: AcousticProgressStats | None = None


def run_benchmark_from_ns(ns: Namespace) -> None:
  model: AcousticModelBase
  if ns.use_perch:
    if ns.backend != MODEL_BACKEND_PB:
      raise ValueError(
        "The Perch v2 model is only available with the Protobuf backend."
      )
    if ns.precision != MODEL_PRECISION_FP32:
      raise ValueError("The Perch v2 model only supports 'fp32' precision.")

    first_device = ns.devices[0] if len(ns.devices) > 0 else "CPU"
    model = load_perch_v2(device=first_device)
  else:
    if ns.backend == MODEL_BACKEND_TF:
      model = birdnet.model_loader.load(
        MODEL_TYPE_ACOUSTIC,
        ACOUSTIC_MODEL_VERSION_V2_4,
        MODEL_BACKEND_TF,
        precision=cast(MODEL_PRECISIONS, ns.precision),
        library=cast(LIBRARY_TYPES, ns.tf_library),
      )
    elif ns.backend == MODEL_BACKEND_PB:
      model = birdnet.model_loader.load(
        MODEL_TYPE_ACOUSTIC,
        ACOUSTIC_MODEL_VERSION_V2_4,
        MODEL_BACKEND_PB,
        precision=MODEL_PRECISION_FP32,
      )
    else:
      raise AssertionError()

    result_container = BenchmarkResultContainer(ns, model)

    show_stats_fn = partial(show_progress_stats, result_container=result_container)
    result_container.output = model.predict(
      ns.inputs,
      top_k=ns.top_k,
      n_producers=ns.producers,
      n_workers=ns.workers,
      batch_size=ns.batch_size,
      overlap_duration_s=ns.overlap,
      speed=ns.speed,
      default_confidence_threshold=ns.confidence,
      custom_confidence_thresholds=None,
      apply_sigmoid=APPLY_SIGMOID,
      sigmoid_sensitivity=SIGMOID_SENSITIVITY,
      custom_species_list=CUSTOM_SPECIES,
      half_precision=ns.half_precision,
      max_audio_duration_min=None,
      show_stats=ns.show_stats,
      device=ns.devices if len(ns.devices) > 1 else ns.devices[0],
      prefetch_ratio=ns.prefetch_ratio,
      progress_callback=show_stats_fn,
      bandpass_fmin=model.get_sig_fmin(),
      bandpass_fmax=model.get_sig_fmax(),
    )

    save_statistics(result_container)


def show_progress_stats(
  info: AcousticProgressStats, result_container: BenchmarkResultContainer
) -> None:
  output_msg_fields = [
    f"MEM: {info.memory_usage_MiB:.0f} M",
    f"BUF: {info.buffer_stats.filled_slots}/{info.buffer_stats.slots}",
    f"P-SPEED: {info.producer_stats.speed_xrt:.0f} xRT "
    f"[{info.producer_stats.speed_seg_per_s:.0f} seg/s]",
    f"P-WAIT: {info.producer_stats.wait_ms:.2f} ms",
    f"P-BATCH: {info.producer_stats.batch_ms:.2f} ms",
    f"P-SEARCH: {info.producer_stats.search_ms:.2f} ms",
    f"P-FLUSH: {info.producer_stats.flush_ms:.2f} ms",
  ]
  if info.worker_stats is not None:
    output_msg_fields.extend(
      [
        f"W-SPEED: {info.worker_stats.speed_xrt:.0f} xRT "
        f"[{info.worker_stats.speed_seg_per_s:.0f} seg/s]",
        f"W-WAIT: {info.worker_stats.wait_ms:.2f} ms",
        f"W-SEARCH: {info.worker_stats.search_ms:.2f} ms",
        f"W-JOB: {info.worker_stats.job_ms:.2f} ms",
        f"W-COPY: {info.worker_stats.copy_ms:.2f} ms",
        f"W-INFER: {info.worker_stats.inference_ms:.2f} ms",
        f"W-ADD: {info.worker_stats.add_ms:.2f} ms",
        f"BUSY: {info.worker_stats.busy}/{info.worker_stats.workers}",
      ]
    )
  else:
    output_msg_fields.append("W: loading model...")

  output_msg_fields.extend(
    [
      f"PROG: {info.progress_pct:.1f} %",
      f"ETA: {info.est_remaining_time_hhmmss or '...'}",
      f"Status: {info.finished}",
    ]
  )

  output_msg = "; ".join(output_msg_fields)
  print(output_msg, file=sys.stdout)  # noqa: T201

  if info.finished:
    result_container.stats = info


def save_statistics(result_container: BenchmarkResultContainer) -> None:
  assert result_container.output is not None
  assert result_container.stats is not None
  assert result_container.stats.finished

  ns = result_container.ns
  stats = result_container.stats
  output = result_container.output
  model = result_container.model

  output_folder = get_benchmark_dir("acoustic", "predictions")

  result_json = OrderedDict()

  input_stats = OrderedDict()
  input_stats["n_files"] = output.input_durations.shape[0]
  input_stats["total_duration"] = str(
    timedelta(seconds=float(np.sum(output.input_durations, dtype=np.float64)))
  )
  input_stats["average_duration"] = str(
    timedelta(seconds=float(output.input_durations.mean()))
  )
  input_stats["min_duration"] = str(
    timedelta(seconds=float(output.input_durations.min()))
  )
  input_stats["max_duration"] = str(
    timedelta(seconds=float(output.input_durations.max()))
  )
  file_formats = ", ".join(sorted({Path(x).suffix[1:].upper() for x in output.inputs}))
  input_stats["file_formats"] = file_formats
  result_json["input"] = input_stats

  params = OrderedDict()
  params["n_producers"] = ns.producers
  params["n_workers"] = ns.workers
  params["batch_size"] = ns.batch_size
  params["overlap_duration_s"] = ns.overlap
  params["speed"] = ns.speed
  params["half_precision"] = ns.half_precision
  params["devices"] = ns.devices
  params["prefetch_ratio"] = ns.prefetch_ratio
  params["top_k"] = ns.top_k
  params["confidence_threshold"] = ns.confidence
  params["apply_sigmoid"] = APPLY_SIGMOID
  params["sigmoid_sensitivity"] = SIGMOID_SENSITIVITY
  params["custom_species"] = CUSTOM_SPECIES
  result_json["params"] = params

  # computer stats
  # hardware stats
  hw_stats = OrderedDict()
  hw_stats["host"] = platform.node()
  hw_stats["cpu"] = platform.processor()
  hw_stats["cpu_physical_cores"] = psutil.cpu_count(logical=False) or -1
  hw_stats["cpu_logical_cores"] = psutil.cpu_count(logical=True) or -1
  hw_stats["ram_GiB"] = psutil.virtual_memory().total / 1024**3
  result_json["hardware"] = hw_stats

  # software stats
  sw_stats = OrderedDict()
  sw_stats["start_method"] = multiprocessing.get_start_method()
  sw_stats["os"] = f"{platform.system()} {platform.release()}"
  sw_stats["python_version"] = platform.python_version()
  sw_stats["python_implementation"] = platform.python_implementation()
  sw_stats["package_version"] = get_package_version()
  sw_stats["tf_available"] = tf_installed()
  sw_stats["litert_available"] = litert_installed()
  result_json["software"] = sw_stats

  # model stats
  model_stats = OrderedDict()
  model_stats["path"] = str(model.model_path.absolute())
  model_stats["type"] = MODEL_TYPE_ACOUSTIC
  model_stats["version"] = model.get_version()
  model_stats["n_species"] = model.n_species
  model_stats["sig_fmin"] = model.get_sig_fmin()
  model_stats["sig_fmax"] = model.get_sig_fmax()
  model_stats["sample_rate"] = model.get_sample_rate()
  model_stats["segment_size_s"] = model.get_segment_size_s()
  model_stats["segment_size_samples"] = model.get_segment_size_samples()
  model_stats["is_custom"] = model.is_custom_model
  model_stats["sigmoid_sensitivity"] = 1.0
  result_json["model"] = model_stats

  # backend stats
  backend_stats = OrderedDict()
  backend_stats["type"] = model.backend_type.name()
  backend_stats["precision"] = model._backend_type.precision()
  backend_stats["supports_encoding"] = model._backend_type.supports_encoding()
  backend_stats["custom_kwargs"] = model.backend_kwargs
  result_json["backend"] = backend_stats

  # processing stats
  assert stats.speed_xrt is not None
  proc_stats = OrderedDict()
  proc_stats["wall_time_s"] = stats.wall_time_s
  proc_stats["wall_time_hhmmss"] = str(timedelta(seconds=stats.wall_time_s))
  proc_stats["speed_xrt"] = stats.speed_xrt
  proc_stats["speed_rtf"] = 1 / stats.speed_xrt
  proc_stats["speed_seg_per_s"] = stats.speed_seg_per_s
  proc_stats["cpu_usage_max_pct"] = stats.cpu_usage_max_pct
  proc_stats["memory_usage_max_MiB"] = stats.memory_usage_MiB
  proc_stats["processed_segments"] = stats.processed_segments
  proc_stats["processed_batches"] = stats.processed_batches

  proc_buffer = OrderedDict()
  proc_buffer["n_slots"] = stats.buffer_stats.slots
  proc_buffer["median_filled_slots"] = stats.buffer_stats.filled_slots
  proc_buffer["median_busy_slots"] = stats.buffer_stats.busy_slots
  proc_buffer["median_preloaded_slots"] = stats.buffer_stats.preloaded_slots
  proc_buffer["median_free_slots"] = stats.buffer_stats.free_slots
  proc_stats["buffer"] = proc_buffer

  proc_prod = OrderedDict()
  proc_prod["speed_xrt"] = stats.producer_stats.speed_xrt
  proc_prod["speed_rtf"] = 1 / stats.producer_stats.speed_xrt
  proc_prod["speed_seg_per_s"] = stats.producer_stats.speed_seg_per_s
  proc_prod["median_wait_ms"] = stats.producer_stats.wait_ms
  proc_prod["median_batch_ms"] = stats.producer_stats.batch_ms
  proc_prod["median_search_ms"] = stats.producer_stats.search_ms
  proc_prod["median_flush_ms"] = stats.producer_stats.flush_ms
  proc_stats["producer"] = proc_prod

  assert stats.worker_stats is not None
  proc_worker = OrderedDict()
  proc_worker["speed_xrt"] = stats.worker_stats.speed_xrt
  proc_worker["speed_rtf"] = 1 / stats.worker_stats.speed_xrt
  proc_worker["speed_seg_per_s"] = stats.worker_stats.speed_seg_per_s
  # proc_worker["speed_min_per_s"] = stats.worker_stats.speed_seg_per_s
  proc_worker["median_wait_ms"] = stats.worker_stats.wait_ms
  proc_worker["median_search_ms"] = stats.worker_stats.search_ms
  proc_worker["median_job_ms"] = stats.worker_stats.job_ms
  proc_worker["median_copy_ms"] = stats.worker_stats.copy_ms
  proc_worker["median_inference_ms"] = stats.worker_stats.inference_ms
  proc_worker["median_add_ms"] = stats.worker_stats.add_ms
  proc_worker["median_busy"] = stats.worker_stats.busy
  proc_stats["worker"] = proc_worker
  result_json["processing"] = proc_stats

  output_stats = OrderedDict()
  output_stats["file_size_MiB"] = output.memory_size_MiB
  output_stats["n_unprocessable_inputs"] = len(output.unprocessable_inputs)

  stats_json_path = output_folder / "benchmark_stats.json"
  with stats_json_path.open("w", encoding="utf8") as f:
    json.dump(result_json, f, indent=2)
  print(f"Saved benchmark statistics to: {stats_json_path.absolute()}")  # noqa: T201

  summary = (
    f"-------------------------------\n"
    f"------ Benchmark summary ------\n"
    f"-------------------------------\n"
    # f"Start time: {bmm.time_begin}\n"
    # f"End time:   {bmm.time_end}\n"
    f"Wall time:  {proc_stats['wall_time_hhmmss']}\n"
    f"Input: {input_stats['n_files']} file(s) ({input_stats['file_formats']})\n"
    f"  Total duration: {input_stats['total_duration']}\n"
    f"  Average duration: {input_stats['average_duration']}\n"
    f"  Minimum duration (single file): {input_stats['min_duration']}\n"
    f"  Maximum duration (single file): {input_stats['max_duration']}\n"
    f"Producer(s): {params['n_producers']}\n"
    f"Buffer: {proc_buffer['median_filled_slots']:.1f}/{proc_buffer['n_slots']} "
    "filled slots (median)\n"
    f"Busy workers: {proc_worker['median_busy']:.1f}/{params['n_workers']} (median)\n"
    f"  Median wait time for next batch: {proc_worker['median_wait_ms']:.3f} ms\n"
    f"Memory usage:\n"
    f"  Program: {proc_stats['memory_usage_max_MiB']:.2f} M (total max)\n"
    # f"  Buffer: {bmm.mem_shm_size_total_MiB:.2f} M (shared memory)\n"
    f"  Result: {output_stats['file_size_MiB']:.2f} M (NumPy)\n"
    f"Performance:\n"
    f"  {proc_stats['speed_xrt']:.0f} x real-time (RTF: {proc_stats['speed_rtf']:.8f})\n"  # noqa: E501
    f"  {proc_stats['speed_seg_per_s']:.0f} segments/s "  # ({bmm.speed_total_audio_per_second} audio/s)\n"  # noqa: E501
    f"Worker performance:\n"
    f"  {proc_worker['speed_xrt']:.0f} x real-time (RTF: {proc_worker['speed_rtf']:.8f})\n"  # noqa: E501
    f"  {proc_worker['speed_seg_per_s']:.0f} segments/s"  #  ({bmm.speed_worker_total_audio_per_second} audio/s)\n"  # noqa: E501
  )

  stats_human_readable_out = output_folder / "benchmark_stats.txt"
  stats_human_readable_out.write_text(summary, encoding="utf8")
  print(f"Saved benchmark statistics to: {stats_human_readable_out.absolute()}")  # noqa: T201

  return
  # print("Saving result using internal format (.npz)...")
  # result.save(result_npz)
  # saved_files = [result_npz]
  # saved_files += strategy.save_results_extra(result, benchmark_run_dir, prepend)

  # summary += (
  #   f"-------------------------------\n"
  #   f"Benchmark folder:\n"
  #   f"  {benchmark_run_dir.absolute()}\n"
  #   f"Statistics results written to:\n"
  #   f"  {stats_human_readable_out.absolute()}\n"
  #   f"  {stats_out_json.absolute()}\n"
  #   f"  {all_sessions_meta_df_out.absolute()}\n"
  #   f"Prediction results written to:\n"
  # )
  # for saved_file in saved_files:
  #   summary += f"  {saved_file.absolute()}\n"
  # summary += (
  #   f"Session log file:\n  {resources.logging_resources.session_log_file.absolute()}\n"  # noqa: E501
  # )
