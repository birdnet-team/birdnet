# birdnet_batch_inference.py – raw‑audio version
from __future__ import annotations

import ctypes

# You'll need these imports in your own code
import logging
import logging.handlers
import math
import multiprocessing
import multiprocessing as mp
import os
import queue
import shutil
import sys
import tempfile
import time
import zipfile
from collections import deque
from collections.abc import Generator
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
from birdnet_v2.inference.producer import (
  Producer,
  get_chunks_with_overlap,  # type: ignore
  load_audio_in_chunks_with_overlap,
  shm_ring_from_name,
)
from birdnet_v2.inference.species_tensor import SpeciesTensor
from birdnet_v2.inference.worker import ChildWorker
from birdnet_v2.logging_utils import (
  QueueFileWriter,
  get_package_logging_level,
)
from birdnet_v2.model_downloader import ModelDownloader


class PerformanceTracker(bn_logging.LogableProcessBase):
  def __init__(
    self,
    pred_dur_queue: mp.SimpleQueue,
    stop_event: mp.Event,
    update_interval: float,
    print_last_n: int,
    start: float,
    stop_time: mp.RawValue,
    logging_queue: mp.Queue,
    logging_level: int,
    perf_res: mp.SimpleQueue,
    chunk_size_s: float = 3.0,
  ):
    super().__init__(__name__, logging_queue, logging_level)

    self._perf_res = perf_res
    self._pred_dur_queue = pred_dur_queue
    self._pred_dur_deque = deque(maxlen=print_last_n)
    self._batch_sizes_deque = deque(maxlen=print_last_n)
    self._update_every = update_interval
    self._stop_event = stop_event
    self._next_print = time.time()
    self._total_chunks_processed = 0
    self._summed_pred_duration = 0.0
    self._start = start
    self._chunk_size_s = chunk_size_s
    self._stop_time = stop_time

  def __call__(self):
    self._init_logging()

    stop = None
    perf_duration = 0
    ramp_up_time_until_first_pred = None

    while True:
      processing_finished = self._stop_event.is_set()
      queue_is_empty = self._pred_dur_queue.empty()
      if processing_finished:
        stop = self._stop_time.value
        if queue_is_empty:
          break

      while not self._pred_dur_queue.empty():
        dur, batch_size = self._pred_dur_queue.get()
        self._pred_dur_deque.append(dur)
        self._batch_sizes_deque.append(batch_size)
        self._total_chunks_processed += batch_size
        self._summed_pred_duration += dur
        if ramp_up_time_until_first_pred is None:
          ramp_up_time_until_first_pred = time.perf_counter() - self._start - dur
          self._logger.info(
            f"Rampup time until first prediction: {ramp_up_time_until_first_pred:.2f}s"
          )

      now = time.time()
      perf_duration = time.perf_counter() - self._start
      if now >= self._next_print and len(self._pred_dur_deque) > 0:
        avg = sum(self._pred_dur_deque) / sum(self._batch_sizes_deque)
        chunks_per_s = self._total_chunks_processed / perf_duration
        output_msg = f"Ø Inference speed: {self._summed_pred_duration / self._total_chunks_processed * 1000:.0f} ms/chunk; last {len(self._pred_dur_deque)} predictions: {avg * 1000:.0f} ms/chunk; {chunks_per_s:.0f} chunks/s = {chunks_per_s * self._chunk_size_s / 60:.2f} min/s"
        self._logger.info(output_msg)
        print(output_msg, file=sys.stdout)

        self._next_print = now + self._update_every
    assert stop is not None
    total_duration = stop - self._start
    self._logger.info(f"Total processing time: {total_duration:.2f} s")
    stats = {}
    stats["total_chunks_processed"] = self._total_chunks_processed
    stats["summed_prediction_duration_s"] = self._summed_pred_duration
    stats["total_duration_s"] = total_duration
    stats["ramp_up_time_until_first_pred_s"] = ramp_up_time_until_first_pred
    stats["model_pred_ms_per_chunk"] = (
      stats["summed_prediction_duration_s"] / stats["total_chunks_processed"] * 1000
    )
    stats["pc_chunks_per_s"] = (
      stats["total_chunks_processed"] / stats["total_duration_s"]
    )
    stats["pc_audio_min_per_s"] = stats["pc_chunks_per_s"] * self._chunk_size_s / 60
    self._logger.info(stats)
    print(stats, file=sys.stdout)
    self._perf_res.put(stats)
    self._uninit_logging()
