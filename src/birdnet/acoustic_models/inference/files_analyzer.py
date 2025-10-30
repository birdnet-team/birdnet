import ctypes
import multiprocessing as mp
import os
from multiprocessing.synchronize import Event
from queue import Empty

import birdnet.acoustic_models.inference_pipeline.logging as bn_logging
from birdnet.acoustic_models.inference.producer import get_audio_duration_s
from birdnet.helper import RingField, get_max_n_segments, max_value_for_uint_dtype


class FilesAnalyzer():
  def __init__(
    self,
    session_id: str,
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
    finished: Event,
    state: mp.RawValue,
    start_signal: Event,
  ) -> None:
    # super().__init__(session_id, __name__, logging_queue, logging_level)
    self._logger = bn_logging.get_logger_from_session(session_id, __name__)
    # self._files = files
    self._state = state
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
    self._finished = finished
    self._start_signal = start_signal

  def _check_cancel_event(self) -> bool:
    if self._cancel_event.is_set():
      self._logger.debug(f"FilesAnalyzer({os.getpid()}) - Received cancel event.")
      return True
    return False

  def _check_end_event(self) -> bool:
    if self._end_event.is_set():
      self._logger.debug(f"FilesAnalyzer({os.getpid()}) - Received end event.")
      return True
    return False

  def __call__(self) -> None:
    #self._init_logging()
    self.run_main_loop()
    # self._uninit_logging()

  def run_main_loop(self) -> None:
    while True:
      self._logger.info("FilesAnalyzer waiting for input files batch...")
      while not self._start_signal.wait(timeout=1.0):
        if self._check_cancel_event():
          # self._uninit_logging()
          return
        if self._check_end_event():
          return

      self._start_signal.clear()
      self._logger.debug(
        f"FilesAnalyzer({os.getpid()}) - Received start signal. Starting processing."
      )
      # check that it was resetted
      assert self._tot_n_segments.value == 0
      self.run_main()

  def run_main(self) -> None:
    durations = []
    current_max_segment_index = 0
    n_segments = 0

    while True:
      try:
        files = self._input_files_queue.get(block=True, timeout=1.0)
        break
      except Empty:
        # it has started, so ending is not possible, only canceling
        if self._check_cancel_event():
          return

    self._logger.info(f"FilesAnalyzer received {len(files)} files to analyze.")

    for path in files:
      if self._check_cancel_event():
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
            f"File {path} has a duration of {audio_duration_s / 60:.2f} min and "
            f"contains {file_n_segments} segments, which exceeds the maximum supported "
            f"amount of segments {self._max_supported_segment_index + 1}. "
            f"Please set maximum audio duration."
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

    self._finished.set()
