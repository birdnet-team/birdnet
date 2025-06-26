import logging
import multiprocessing
import multiprocessing as mp
import os
from collections.abc import Generator, Iterable
from itertools import count, islice
from logging import getLogger
from logging.handlers import QueueHandler
from multiprocessing import Queue, shared_memory
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event, Semaphore
from pathlib import Path
from time import sleep
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import requests
import soundfile as sf
from numpy.typing import DTypeLike
from ordered_set import OrderedSet
from scipy.signal import butter, lfilter, resample
from tqdm import tqdm

from birdnet_v2.inference.producer import get_audio_duration_s
import birdnet_v2.logging_utils as bn_logging
from birdnet.types import Species, TimeInterval
from birdnet.utils import (
  bandpass_signal,
  fillup_with_silence,
  get_chunks_with_overlap,
  itertools_batched,
  resample_array,
)
from birdnet_v2.globals import DONE_FLAG, READABLE_FLAG, WRITABLE_FLAG
from birdnet_v2.helper import (
  RingField,
  get_max_n_chunks,
  max_value_for_uint_dtype,
  uint_dtype_for,
)


class FilesAnalyzer(bn_logging.LogableProcessBase):
  def __init__(
    self,
    files: List[Path],
    logging_queue: mp.Queue,
    logging_level: int,
    chunk_duration_s: float,
    overlap_duration_s: float,
    rf_chunk_indices: RingField,
    max_chunk_idx_ptr: mp.RawValue,
    analyzing_result: mp.SimpleQueue,
    tot_n_chunks: mp.RawValue,
  ):
    super().__init__(__name__, logging_queue, logging_level)
    self._files = files
    self.chunk_duration_s = chunk_duration_s
    self.overlap_duration_s = overlap_duration_s
    self._rf_chunk_indices = rf_chunk_indices
    self._max_chunk_idx_ptr = max_chunk_idx_ptr
    self._tot_n_chunks = tot_n_chunks
    self._max_supported_chunk_index = (
      max_value_for_uint_dtype(rf_chunk_indices.dtype) - 1
    )
    self._analyzing_result = analyzing_result

  def __call__(self) -> None:
    self._init_logging()
    durations = []
    current_max_chunk_index = 0
    n_chunks = 0
    for path in self._files:
      audio_duration_s = get_audio_duration_s(path)
      durations.append(audio_duration_s)

      file_n_chunks = get_max_n_chunks(
        audio_duration_s, self.chunk_duration_s, self.overlap_duration_s
      )
      file_max_chunk_index = file_n_chunks - 1
      n_chunks += file_n_chunks

      if file_max_chunk_index > current_max_chunk_index:
        if file_max_chunk_index > self._max_supported_chunk_index:
          self._logger.error(
            f"File {path} has a duration of {audio_duration_s / 60:.2f} min and contains {file_n_chunks} chunks, which exceeds the maximum supported amount of chunks {self._max_supported_chunk_index + 1}. Please set maximum audio duration."
          )
          continue
        current_max_chunk_index = file_max_chunk_index
        self._max_chunk_idx_ptr.value = current_max_chunk_index
    self._tot_n_chunks.value = n_chunks
    res = {}
    res["file_durations_s"] = durations
    res["max_chunk_index"] = current_max_chunk_index
    res["tot_n_chunks"] = n_chunks
    self._analyzing_result.put(res)
    self._logger.info(f"Total duration of all files: {sum(durations) / 60**2:.2f} h.")
    self._uninit_logging()
