from __future__ import annotations

import os
import queue
import threading
from collections.abc import Callable
from multiprocessing.synchronize import Event

import numpy as np

import birdnet.acoustic.inference.core.logs as bn_logging

# (file_path, species_ids, species_probs, species_masked, is_invalid, duration_s)
DispatchItem = tuple[object, np.ndarray, np.ndarray, np.ndarray, bool, float]

# (file_path, species_ids, species_probs, species_masked, is_invalid, duration_s)
BuildResultFn = Callable[
  [object, np.ndarray, np.ndarray, np.ndarray, bool, float], object
]


class FileCompletionDispatcher:
  """Turns per-file completion items into results and invokes the user callback.

  Runs on a dedicated thread in the main process, draining the in-process
  dispatch queue the consumer feeds. Keeping the (potentially heavy) result
  construction and user I/O here — off the consumer's hot path — is what lets
  ``on_file_complete`` stream results without throttling inference.

  Mirrors :class:`ProgressDispatcher`: it loops per run, driven by
  ``start_signal``/``finish_signal``, and exits on ``end_event``.
  """

  def __init__(
    self,
    session_id: str,
    dispatch_queue: queue.Queue,
    callback_fn: Callable[[object], None],
    build_result_fn: BuildResultFn,
    cancel_event: Event,
    end_event: Event,
    start_signal: threading.Event,
    finish_signal: threading.Event,
  ) -> None:
    self._session_id = session_id
    self._dispatch_queue = dispatch_queue
    self._callback_fn = callback_fn
    self._build_result_fn = build_result_fn
    self._cancel_event = cancel_event
    self._end_event = end_event
    self._start_signal = start_signal
    self._finish_signal = finish_signal
    self._logger = bn_logging.get_logger_from_session(session_id, __name__)

  def _log(self, message: str) -> None:
    self._logger.debug(f"FCD_{os.getpid()}: {message}")

  def __call__(self) -> None:
    try:
      self.run_main_loop()
    except Exception as e:
      self._logger.exception(
        "FileCompletionDispatcher encountered an exception.",
        exc_info=e,
        stack_info=True,
      )
      self._cancel_event.set()

  def run_main_loop(self) -> None:
    while True:
      while not self._start_signal.wait(timeout=1.0):
        if self._cancel_event.is_set():
          return
        if self._end_event.is_set():
          return

      self._start_signal.clear()
      self._log("Received start signal. Dispatching file completions.")
      self.run_main()

      self._log("Set finish signal.")
      self._finish_signal.set()

  def run_main(self) -> None:
    while True:
      try:
        item = self._dispatch_queue.get(timeout=1.0)
      except queue.Empty:
        if self._cancel_event.is_set() or self._end_event.is_set():
          return
        continue

      if item is None:
        # Sentinel: the consumer finished this run.
        return

      file_path, ids, probs, masked, is_invalid, duration = item
      try:
        result = self._build_result_fn(
          file_path, ids, probs, masked, is_invalid, duration
        )
        self._callback_fn(result)
      except Exception as e:
        self._logger.exception(
          "on_file_complete callback raised; cancelling analysis.",
          exc_info=e,
          stack_info=True,
        )
        self._cancel_event.set()
