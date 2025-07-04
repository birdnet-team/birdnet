import argparse
import os
import sys
from argparse import ArgumentParser, Namespace

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


def run_benchmark() -> None:
  args: list[str] = sys.argv[1:]
  run_benchmark_from_args(args)


def run_benchmark_from_args(args: list[str]) -> None:
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
    "output",
    type=parse_path,
    metavar="OUTPUT_CSV",
    help="output CSV file for the prediction results",
  )

  parser.add_argument(
    "-b",
    "--backend",
    type=str,
    choices=["tf", "pb"],
    metavar="BACKEND",
    help="use this backend",
    default="tf",
  )

  parser.add_argument(
    "-p",
    "--producers",
    type=parse_positive_integer,
    metavar="PRODUCERS",
    help="number of producers to use for processing",
    default=1,
  )

  parser.add_argument(
    "-w",
    "--workers",
    type=parse_positive_integer,
    metavar="WORKERS",
    help="number of workers to use for processing",
    default=os.cpu_count() or 4,
  )

  parser.add_argument(
    "-k",
    "--top-k",
    type=parse_positive_integer,
    metavar="K",
    help="number of top K species to return for each audio segment",
    default=5,
  )

  parser.add_argument(
    "-s",
    "--batch-size",
    type=parse_positive_integer,
    metavar="BATCH-SIZE",
    help="number of top species to return for each audio segment",
    default=1,
  )

  parser.add_argument(
    "-d",
    "--devices",
    type=parse_non_empty_or_whitespace,
    nargs="+",
    metavar="DEVICE",
    help="device to use for processing (e.g., 'CPU', 'GPU', 'GPU:0', 'GPU:1', ...); either string or list of strings, latter is useful for multi-GPU setups, the first GPU will be used for the first producer, the second GPU for the second producer, etc.; GPU is only available for the Protobuf backend",
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
    "-f",
    "--prefetch_ratio",
    type=parse_non_negative_integer,
    metavar="RATIO",
    help="amount of additional ring-buffer capacity to keep ahead of the workers, expressed as a ratio of the default size, which is the amount of workers.",
    default=0,
  )

  ns: Namespace = parser.parse_args(args)
  run_benchmark_from_ns(ns)


def run_benchmark_from_ns(ns: Namespace) -> None:
  model: AcousticModelBaseV2_4 = birdnet.model_loader.load(
    model_type="acoustic", version="2.4", backend=ns.backend
  )
  result = model.analyze(
    ns.inputs,
    top_k=ns.top_k,
    n_producers=ns.producers,
    n_workers=ns.workers,
    batch_size=ns.batch_size,
    overlap_duration_s=ns.overlap,
    default_confidence_threshold=ns.confidence,
    custom_confidence_thresholds=None,
    apply_sigmoid=False,
    sigmoid_sensitivity=None,
    custom_species_list=None,
    half_precision=True,
    max_audio_duration_min=None,
    show_stats="benchmark",
    device=ns.devices if len(ns.devices) > 1 else ns.devices[0],
    n_slots_factor=ns.prefetch_ratio + 1,
    use_bandpass=False,
    bandpass_fmax=None,
    bandpass_fmin=None,
  )

  result.to_csv(ns.output, encoding="utf-8", silent=False)
  print(f"Prediction results saved to: {ns.output.absolute()}.")
