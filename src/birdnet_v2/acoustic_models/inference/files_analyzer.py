import ctypes
import multiprocessing
import multiprocessing as mp
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import List

import numpy as np
from ordered_set import OrderedSet

import birdnet_v2.logging_utils as bn_logging
from birdnet_v2.acoustic_models.inference.producer import get_audio_duration_s
from birdnet_v2.helper import RingField, get_max_n_chunks, max_value_for_uint_dtype


class FilesAnalyzer(bn_logging.LogableProcessBase):
  def __init__(
    self,
    files: OrderedSet[Path],
    logging_queue: mp.Queue,
    logging_level: int,
    chunk_duration_s: float,
    overlap_duration_s: float,
    rf_chunk_indices: RingField,
    max_chunk_idx_ptr: mp.RawValue,
    analyzing_result: mp.SimpleQueue,
    tot_n_chunks: ctypes.c_uint64,
    cancel_event: Event,
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
    self._cancel_event = cancel_event

  def __call__(self) -> None:
    self._init_logging()
    durations = []
    current_max_chunk_index = 0
    n_chunks = 0
    for path in self._files:
      if self._cancel_event.is_set():
        self._logger.info("FilesAnalyzer canceled because of cancel event.")
        self._uninit_logging()
        return

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
    res["file_durations_s"] = np.array(durations)
    res["tot_n_chunks"] = n_chunks
    self._analyzing_result.put(res)
    self._logger.info(f"Total duration of all files: {sum(durations) / 60**2:.2f} h.")
    self._uninit_logging()
