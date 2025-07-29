import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from typing import cast

import psutil

import birdnet
import birdnet.model_loader
from birdnet.acoustic_models.v2_4.base import AcousticModelBaseV2_4
from birdnet.argparse_helper import (
  ConvertToSetAction,
  parse_float,
  parse_non_empty_or_whitespace,
  parse_non_negative_integer,
  parse_path,
  parse_positive_integer,
)
from birdnet.globals import (
  ACOUSTIC_MODEL_VERSION_V2_4,
  LIBRARY_TF,
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
from birdnet.logging_utils import get_package_logger


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
    help=f"use this tensorflow library (default: {LIBRARY_TF})",
    default=LIBRARY_TF,
  )

  parser.add_argument(
    "-p",
    "--precision",
    type=str,
    choices=VALID_MODEL_PRECISIONS,
    metavar="PRECISION",
    help=f"model precision (default: {MODEL_PRECISION_FP32})",
    default=MODEL_PRECISION_FP32,
  )

  parser.add_argument(
    "-f",
    "--feeders",
    type=parse_positive_integer,
    metavar="FEEDERS",
    help="number of feeders which will read the input files (default: 1)",
    default=1,
  )

  parser.add_argument(
    "-w",
    "--workers",
    type=parse_positive_integer,
    metavar="WORKERS",
    help="number of workers to use for processing (default: number of physical CPU cores)",
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
    "-d",
    "--devices",
    type=parse_non_empty_or_whitespace,
    nargs="+",
    metavar="DEVICE",
    help="device(s) to use for processing (only available for the Protobuf backend), e.g., 'CPU', 'GPU', 'GPU:0', 'GPU:1', ...,  (default: 'CPU'); either string or list of strings, latter is useful for multi-GPU setups, the first GPU will be used for the first feeder, the second for the second feeder, etc.",
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
    help="amount of additional buffer capacity to keep ahead of the workers, expressed as a ratio of the amount of workers, i.e., 0 means no prefetching, 1 means one additional slot per worker, 2 means two additional slots per worker, etc. (default: 1)",
    default=1,
  )

  parser.add_argument(
    "--show-stats",
    type=str,
    choices=["no", "minimal", "progress", "benchmark"],
    metavar="SHOW_STATS",
    help="show statistics during processing; 'no' means no statistics, 'minimal' means only minimal statistics, 'progress' means progress bar and minimal statistics, 'benchmark' means progress bar and detailed statistics (default: 'benchmark')",
    default="benchmark",
  )

  ns: Namespace = parser.parse_args(args)
  run_benchmark_from_ns(ns)


def run_benchmark_from_ns(ns: Namespace) -> None:
  if ns.backend == MODEL_BACKEND_TF:
    model = birdnet.model_loader.load(
      MODEL_TYPE_ACOUSTIC,
      ACOUSTIC_MODEL_VERSION_V2_4,
      MODEL_BACKEND_TF,
      precision=cast(MODEL_PRECISIONS, ns.precision),
      library=cast(LIBRARY_TYPES, ns.tf_library),
    )

    model.predict(
      ns.inputs,
      top_k=ns.top_k,
      feeders=ns.feeders,
      workers=ns.workers,
      batch_size=ns.batch_size,
      overlap_duration_s=ns.overlap,
      default_confidence_threshold=ns.confidence,
      custom_confidence_thresholds=None,
      apply_sigmoid=True,
      sigmoid_sensitivity=1.0,
      custom_species_list=None,
      half_precision=True,
      max_audio_duration_min=None,
      show_stats=ns.show_stats,
      prefetch_ratio=ns.prefetch_ratio,
      bandpass_fmin=AcousticModelBaseV2_4.get_sig_fmin(),
      bandpass_fmax=AcousticModelBaseV2_4.get_sig_fmax(),
    )
  elif ns.backend == MODEL_BACKEND_PB:
    model = birdnet.model_loader.load(
      MODEL_TYPE_ACOUSTIC,
      ACOUSTIC_MODEL_VERSION_V2_4,
      MODEL_BACKEND_PB,
      precision=MODEL_PRECISION_FP32,
    )

    model.predict(
      ns.inputs,
      top_k=ns.top_k,
      feeders=ns.feeders,
      workers=ns.workers,
      batch_size=ns.batch_size,
      overlap_duration_s=ns.overlap,
      default_confidence_threshold=ns.confidence,
      custom_confidence_thresholds=None,
      apply_sigmoid=True,
      sigmoid_sensitivity=1.0,
      custom_species_list=None,
      half_precision=True,
      max_audio_duration_min=None,
      show_stats=ns.show_stats,
      device=ns.devices if len(ns.devices) > 1 else ns.devices[0],
      prefetch_ratio=ns.prefetch_ratio,
      bandpass_fmin=AcousticModelBaseV2_4.get_sig_fmin(),
      bandpass_fmax=AcousticModelBaseV2_4.get_sig_fmax(),
    )
  else:
    raise AssertionError()
