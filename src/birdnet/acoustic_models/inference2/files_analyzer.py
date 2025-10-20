import ctypes
import multiprocessing as mp
from multiprocessing.synchronize import Event
from pathlib import Path
from queue import Empty

from ordered_set import OrderedSet

import birdnet.logging_utils as bn_logging
from birdnet.acoustic_models.inference2.producer import get_audio_duration_s
from birdnet.helper import RingField, get_max_n_segments, max_value_for_uint_dtype


class FilesAnalyzer(bn_logging.LogableProcessBase):
  def __init__(
    self,
    logging_queue: mp.Queue,
    logging_level: int,
    segment_duration_s: float,
    overlap_duration_s: float,
    rf_segment_indices: RingField,
    max_segment_idx_ptr: mp.RawValue,
    input_files_queue: mp.Queue,
    analyzing_result: mp.Queue,
    tot_n_segments: ctypes.c_uint64,
    cancel_event: Event,
    end_event: Event,
  ):
    super().__init__(__name__, logging_queue, logging_level)
    # self._files = files
    self._input_files_queue = input_files_queue
    self.segment_duration_s = segment_duration_s
    self.overlap_duration_s = overlap_duration_s
    self._rf_segment_indices = rf_segment_indices
    self._max_segment_idx_ptr = max_segment_idx_ptr
    self._tot_n_segments = tot_n_segments
    self._max_supported_segment_index = (
      max_value_for_uint_dtype(rf_segment_indices.dtype) - 1
    )
    self._analyzing_result = analyzing_result
    self._cancel_event = cancel_event
    self._end_event = end_event

  def __call__(self) -> None:
    self._init_logging()
    durations = []
    current_max_segment_index = 0
    n_segments = 0

    while True:
      self._logger.info("FilesAnalyzer waiting for input files...")
      while waiting_for_input_files := True:
        try:
          files = self._input_files_queue.get(timeout=1.0)
          self._logger.info(f"FilesAnalyzer received {len(files)} files to analyze.")
          break
        except Empty:
          if self._cancel_event.is_set():
            self._logger.info("FilesAnalyzer canceled because of cancel event.")
            self._uninit_logging()
            return

          if self._end_event.is_set():
            self._logger.info("FilesAnalyzer ended because of end event.")
            self._uninit_logging()
            return
      
      # check that it was resetted
      assert self._tot_n_segments.value == 0

      for path in files:
        if self._cancel_event.is_set():
          self._logger.info("FilesAnalyzer canceled because of cancel event.")
          self._uninit_logging()
          return

        audio_duration_s = get_audio_duration_s(path)
        durations.append(audio_duration_s)

        file_n_segments = get_max_n_segments(
          audio_duration_s, self.segment_duration_s, self.overlap_duration_s
        )
        file_max_segment_index = file_n_segments - 1
        n_segments += file_n_segments

        if file_max_segment_index > current_max_segment_index:
          if file_max_segment_index > self._max_supported_segment_index:
            self._logger.error(
              f"File {path} has a duration of {audio_duration_s / 60:.2f} min and contains {file_n_segments} segments, which exceeds the maximum supported amount of segments {self._max_supported_segment_index + 1}. Please set maximum audio duration."
            )
            self._cancel_event.set()
            self._uninit_logging()
            return

          current_max_segment_index = file_max_segment_index
          self._max_segment_idx_ptr.value = current_max_segment_index
      self._tot_n_segments.value = n_segments
      self._logger.debug("Putting analyzing result into queue.")
      self._analyzing_result.put(durations, block=True)
      self._logger.debug("Done putting analyzing result into queue.")
      self._logger.info(f"Total duration of all files: {sum(durations) / 60**2:.2f} h.")

      if self._end_event.is_set():
        self._logger.info("FilesAnalyzer ended because of end event.")
        self._uninit_logging()
        return
