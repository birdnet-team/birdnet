from __future__ import annotations

from multiprocessing import Queue
from multiprocessing.synchronize import Event
from queue import Empty

from birdnet.acoustic_models.inference.tensor import AcousticTensorBase
from birdnet.acoustic_models.inference_pipeline.logs import get_logger_from_session


class Consumer:
  def __init__(
    self,
    session_id: str,
    n_workers: int,
    worker_queue: Queue,
    tensor: AcousticTensorBase,
    cancel_event: Event,
  ) -> None:
    self._n_workers = n_workers
    self._queue = worker_queue
    self._tensor = tensor
    self._cancel_event = cancel_event
    self._logger = get_logger_from_session(session_id, __name__)

  def _log(self, message: str) -> None:
    self._logger.debug(f"C: {message}")

  def __call__(self) -> None:
    try:
      self._run_main_loop()
    except Exception as e:
      self._logger.exception(
        "Consumer encountered an exception.", exc_info=e, stack_info=True
      )
      self._cancel_event.set()

  def _run_main_loop(self) -> None:
    finished_workers = 0
    n_received_predictions = 0
    while finished_workers < self._n_workers:
      if self._cancel_event.is_set():
        self._log("Cancel event set. Exiting.")
        return

      received_block = None
      while True:
        try:
          received_block = self._queue.get(timeout=1.0)
          break
        except Empty:
          if self._cancel_event.is_set():
            self._log("Cancel event set. Exiting.")
            return

      if self._cancel_event.is_set():
        self._log("Cancel event set. Exiting.")
        return

      got_stop_signal_from_worker = received_block is None
      if got_stop_signal_from_worker:
        self._log(
          f"Received stop signal from worker. Finished workers: {finished_workers + 1}."
        )
        finished_workers += 1
        continue

      assert received_block is not None

      block = received_block
      n_received_predictions += 1
      self._log(
        f"Received block with {len(block)} values from worker. "
        f"Total received: {n_received_predictions}"
      )
      self._tensor.write_block(*block)
