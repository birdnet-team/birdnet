import ctypes
import multiprocessing as mp
import os
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from pathlib import Path
from queue import Empty

import numpy as np

import birdnet.acoustic_models.inference_pipeline.logging as bn_logging
from birdnet.acoustic_models.inference.producer import get_audio_duration_s
from birdnet.helper import (
  RingField,
  get_n_segments_speed,
  max_value_for_uint_dtype,
)


class FilesAnalyzer:
  def __init__(
    self,
    session_id: str,
    segment_duration_s: float,
    overlap_duration_s: float,
    speed: float,
    rf_segment_indices: RingField,
    max_segment_idx_ptr: mp.RawValue,  # type: ignore
    input_queue: Queue[list[Path] | list[tuple[np.ndarray, int]]],
    analyzing_result: Queue[list[float]],
    tot_n_segments: ctypes.c_uint64,
    cancel_event: Event,
    end_event: Event,
    finished: Event,
    start_signal: Event,
  ) -> None:
    self._logger = bn_logging.get_logger_from_session(session_id, __name__)
    self._input_queue = input_queue
    self._segment_duration_s = segment_duration_s
    self._overlap_duration_s = overlap_duration_s
    self._speed = speed
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
      self._log("Received cancel event.")
      return True
    return False

  def _check_end_event(self) -> bool:
    if self._end_event.is_set():
      self._log("Received end event.")
      return True
    return False

  def _log(self, message: str) -> None:
    self._logger.debug(f"FILES_ANALYZER({os.getpid()}) - {message}")

  def __call__(self) -> None:
    self.run_main_loop()

  def run_main_loop(self) -> None:
    while True:
      self._log("FilesAnalyzer waiting for input files batch...")
      while not self._start_signal.wait(timeout=1.0):
        if self._check_cancel_event():
          # self._uninit_logging()
          return
        if self._check_end_event():
          return

      self._start_signal.clear()
      self._log("Received start signal. Starting processing.")
      # check that it was resetted
      assert self._tot_n_segments.value == 0
      self.run_main()

  def run_main(self) -> None:
    durations: list[float] = []
    current_max_segment_index = 0
    n_segments = 0

    while True:
      try:
        input_data = self._input_queue.get(block=True, timeout=1.0)
        break
      except Empty:
        # it has started, so ending is not possible, only canceling
        if self._check_cancel_event():
          return

    self._log(f"Received {len(input_data)} inputs to analyze.")

    for i, inp_data in enumerate(input_data):
      if self._check_cancel_event():
        return

      if isinstance(inp_data, Path):
        audio_duration_s = get_audio_duration_s(inp_data)
      else:
        assert isinstance(inp_data, tuple)
        assert len(inp_data) == 2
        assert isinstance(inp_data[0], np.ndarray)
        assert isinstance(inp_data[1], int)
        audio_array, sample_rate = inp_data
        audio_duration_s = audio_array.shape[0] / sample_rate

      durations.append(audio_duration_s)

      file_n_segments = get_n_segments_speed(
        audio_duration_s,
        self._segment_duration_s,
        self._overlap_duration_s,
        self._speed,
      )
      file_max_segment_index = file_n_segments - 1
      n_segments += file_n_segments

      if file_max_segment_index > current_max_segment_index:
        if file_max_segment_index > self._max_supported_segment_index:
          inp_path = (
            f"'{inp_data.absolute()}'"
            if isinstance(inp_data, Path)
            else f"<in-memory audio array> at index {i}"
          )
          self._logger.error(
            f"Input {inp_path} has a duration of {audio_duration_s / 60:.2f} min and "
            f"contains {file_n_segments} segments, which exceeds the maximum supported "
            f"amount of segments {self._max_supported_segment_index + 1}. "
            f"Please set maximum audio duration."
          )
          self._cancel_event.set()
          return

        current_max_segment_index = file_max_segment_index
        self._max_segment_idx_ptr.value = current_max_segment_index
    self._tot_n_segments.value = n_segments
    self._log("Putting analyzing result into queue.")
    self._analyzing_result.put(durations, block=True)
    self._log("Done putting analyzing result into queue.")
    self._log(f"Total duration of all files: {sum(durations) / 60**2:.2f} h.")

    self._finished.set()
