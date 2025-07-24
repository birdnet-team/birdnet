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


class AcousticPipeline:
  def __init__(
    self,
    model_species_list: OrderedSet[str],
    model_path: Path,
    model_backend: MODEL_BACKENDS,
    model_version: ACOUSTIC_MODEL_VERSIONS,
    model_segment_size_s: float,
    model_sample_rate: int,
    model_sig_fmin: int,
    model_sig_fmax: int,
    model_precision: MODEL_PRECISIONS,
    model_is_custom: bool,
  ) -> None:
    self._model_species_list = model_species_list
    self._model_path = model_path
    self._model_backend = model_backend
    self._model_version = model_version
    self._model_segment_size_s = model_segment_size_s
    self._model_sample_rate = model_sample_rate
    self._model_sig_fmin = model_sig_fmin
    self._model_sig_fmax = model_sig_fmax
    self._model_precision = model_precision
    self._model_is_custom = model_is_custom

  def predict_species_from_recordings(
    inp: Path | str | Iterable[Path | str],
    model_backend_kwargs: dict,
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