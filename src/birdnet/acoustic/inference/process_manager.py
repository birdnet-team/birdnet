from __future__ import annotations

import contextvars
import functools
import multiprocessing as mp
import os
import threading
import time
from contextlib import suppress
from logging import Logger
from multiprocessing.process import BaseProcess
from pathlib import Path
from queue import Empty
from typing import TYPE_CHECKING

import numpy as np

from birdnet.acoustic.inference.configs import (
  ConfigType,
  InferenceConfig,
  ResultType,
  TensorType,
)
from birdnet.acoustic.inference.core.consumer import Consumer
from birdnet.acoustic.inference.core.file_completion import FileCompletionDispatcher
from birdnet.acoustic.inference.core.input_analyzer import InputAnalyzer
from birdnet.acoustic.inference.core.logs import (
  get_logger_from_session,
)
from birdnet.acoustic.inference.core.perf_tracker import (
  PerformanceTracker,
  ProgressDispatcher,
)
from birdnet.acoustic.inference.core.producer import Producer
from birdnet.acoustic.inference.core.tensor import AcousticTensorBase
from birdnet.acoustic.inference.file_writer import QueueFileWriter
from birdnet.acoustic.inference.resources import PipelineResources
from birdnet.acoustic.inference.strategy import InferenceStrategyBase
from birdnet.core.base import get_session_id_hash

if TYPE_CHECKING:
  from multiprocessing import Queue

# After a cancelled process is asked to stop (SIGTERM), how long to wait for it
# to actually exit before escalating to SIGKILL. Terminated processes normally
# die within a moment; this only bounds a child that ignores SIGTERM so teardown
# can never hang on it.
_TERMINATE_JOIN_TIMEOUT_S = 5.0


class ProcessManager:
  def __init__(
    self,
    session_id: str,
    config: InferenceConfig,
    strategy: InferenceStrategyBase[ResultType, ConfigType, TensorType],
    specific_config: ConfigType,
    resources: PipelineResources,
  ) -> None:
    self._session_id = session_id
    self._session_hash = get_session_id_hash(session_id)
    self._logger = get_logger_from_session(session_id, __name__)
    self._cfg = config
    # All pipeline processes are created from the session's resolved start
    # method; the resources (queues, events, semaphores) were created from the
    # same context, and mixing contexts is not reliable.
    self._ctx = mp.get_context(config.start_method)
    self._strategy = strategy
    self._specific_cfg = specific_config
    self._res = resources
    self._logging_thread: threading.Thread | None = None
    self._analyzer_thread: threading.Thread | None = None
    self._progress_dispatcher_thread: threading.Thread | None = None
    self._file_completion_thread: threading.Thread | None = None
    self._perf_tracker_process: BaseProcess | None = None
    self._producer_processes: list[BaseProcess] | None = None
    self._worker_processes: list[BaseProcess] | None = None

  def start_file_logging_thread(self) -> threading.Thread:
    logging_listener = threading.Thread(
      target=QueueFileWriter(
        session_id=self._session_id,
        log_queue=self._res.logging_resources.logging_queue,
        logging_level=self._res.logging_resources.logging_level,
        log_file=self._res.logging_resources.session_log_file,
        cancel_event=self._res.processing_resources.cancel_event,
        stop_event=self._res.logging_resources.stop_logging_event,
        processing_finished_event=self._res.processing_resources.processing_finished_event,
      ),
      name=f"{self._session_hash}-QueueFileWriter",
      daemon=True,
    )
    logging_listener.start()
    assert self._logging_thread is None
    self._logging_thread = logging_listener
    return logging_listener

  def start_progress_dispatcher_thread(self) -> threading.Thread:
    assert self._res.stats_resources.callback_queue is not None
    assert self._res.stats_resources.callback_start_signal is not None
    assert self._res.stats_resources.callback_finish_signal is not None
    assert self._res.stats_resources.callback_fn is not None

    dispatcher = ProgressDispatcher(
      session_id=self._session_id,
      callback_fn=self._res.stats_resources.callback_fn,
      check_interval=self._res.processing_resources.update_interval,
      start_signal=self._res.stats_resources.callback_start_signal,
      finish_signal=self._res.stats_resources.callback_finish_signal,
      end_event=self._res.processing_resources.end_event,
      callback_queue=self._res.stats_resources.callback_queue,
      cancel_event=self._res.processing_resources.cancel_event,
      processing_finished_event=self._res.processing_resources.processing_finished_event,
    )

    # Capture the caller's context so the progress callback runs with the same
    # contextvars (e.g. request-scoped state) as the thread that started the
    # session. The callback is invoked from this background worker thread, and
    # without this it would otherwise run with an empty/default context.
    # copy_context() is a one-time shallow copy; ctx.run enters the context once
    # for the whole thread, so there is no per-callback overhead.
    ctx = contextvars.copy_context()
    progress_dispatcher = threading.Thread(
      target=ctx.run,
      args=(dispatcher,),
      name=f"{self._session_hash}-ProgressDispatcher",
      daemon=True,
    )
    progress_dispatcher.start()

    assert self._progress_dispatcher_thread is None
    self._progress_dispatcher_thread = progress_dispatcher
    return progress_dispatcher

  def start_file_completion_dispatcher_thread(self) -> threading.Thread:
    fc = self._res.file_completion_resources
    assert fc.enabled
    assert fc.dispatch_queue is not None
    assert fc.callback_fn is not None
    assert fc.start_signal is not None
    assert fc.finish_signal is not None

    dispatcher = FileCompletionDispatcher(
      session_id=self._session_id,
      dispatch_queue=fc.dispatch_queue,
      callback_fn=fc.callback_fn,
      build_result_fn=functools.partial(
        self._strategy.build_single_file_result, self._cfg
      ),
      cancel_event=self._res.processing_resources.cancel_event,
      end_event=self._res.processing_resources.end_event,
      start_signal=fc.start_signal,
      finish_signal=fc.finish_signal,
    )

    # Run the callback with a copy of the caller's context (contextvars), the
    # same guarantee the progress callback gives.
    ctx = contextvars.copy_context()
    thread = threading.Thread(
      target=ctx.run,
      args=(dispatcher,),
      name=f"{self._session_hash}-FileCompletionDispatcher",
      daemon=True,
    )
    thread.start()

    assert self._file_completion_thread is None
    self._file_completion_thread = thread
    return thread

  def start_performance_tracker_process(self) -> BaseProcess:
    assert self._res.stats_resources.track_performance
    assert self._res.stats_resources.sem_active_workers is not None
    assert self._res.stats_resources.perf_res_queue is not None
    assert self._res.stats_resources.perf_res_start_signal is not None
    assert self._res.stats_resources.perf_res_finish_signal is not None
    assert self._res.stats_resources.wkr_stats_queue is not None
    assert self._res.stats_resources.prd_stats_queue is not None

    perf_tracker_proc = self._ctx.Process(
      target=PerformanceTracker(
        session_id=self._session_id,
        pred_dur_queue=self._res.stats_resources.wkr_stats_queue,
        processing_finished_event=self._res.processing_resources.processing_finished_event,
        update_interval=self._res.processing_resources.update_interval,
        prod_stats_queue=self._res.stats_resources.prd_stats_queue,
        n_workers=self._cfg.processing_conf.workers,
        start=self._res.stats_resources.start,
        sem_filled_slots=self._res.ring_buffer_resources.sem_filled_slots,
        segment_size_s=self._cfg.model_conf.segment_size_s,
        logging_queue=self._res.logging_resources.logging_queue,
        logging_level=self._res.logging_resources.logging_level,
        perf_res=self._res.stats_resources.perf_res_queue,
        parent_process_id=os.getpid(),
        rf_flags=self._res.ring_buffer_resources.rf_flags,
        tot_n_segments_ptr=self._res.analyzer_resources.tot_n_segments_ptr,
        cancel_event=self._res.processing_resources.cancel_event,
        sem_active_workers=self._res.stats_resources.sem_active_workers,
        end_event=self._res.processing_resources.end_event,
        start_signal=self._res.stats_resources.perf_res_start_signal,
        finish_signal=self._res.stats_resources.perf_res_finish_signal,
        callback_queue=self._res.stats_resources.callback_queue,
        start_method=self._cfg.start_method,
      ),
      name=f"{self._session_hash}-PerformanceTracker",
      daemon=True,
    )
    perf_tracker_proc.start()

    assert self._perf_tracker_process is None
    self._perf_tracker_process = perf_tracker_proc
    return perf_tracker_proc

  def start_file_analyzer_thread(self) -> threading.Thread:
    file_analyzer_proc = threading.Thread(
      target=InputAnalyzer(
        session_id=self._session_id,
        segment_duration_s=self._cfg.model_conf.segment_size_s,
        overlap_duration_s=self._cfg.processing_conf.overlap_duration_s,
        speed=self._cfg.processing_conf.speed,
        max_segment_idx_ptr=self._res.analyzer_resources.max_segment_idx_ptr,
        rf_segment_indices=self._res.ring_buffer_resources.rf_segment_indices,
        analyzing_result=self._res.analyzer_resources.analyzer_queue,
        tot_n_segments=self._res.analyzer_resources.tot_n_segments_ptr,
        cancel_event=self._res.processing_resources.cancel_event,
        end_event=self._res.processing_resources.end_event,
        input_queue=self._res.analyzer_resources.input_queue,
        finished=self._res.analyzer_resources.finished,
        start_signal=self._res.analyzer_resources.start_signal,
        finish_signal=self._res.analyzer_resources.finish_signal,
      ),
      name=f"{self._session_hash}-FileAnalyzer",
      daemon=True,
    )
    file_analyzer_proc.start()

    assert self._analyzer_thread is None
    self._analyzer_thread = file_analyzer_proc
    return file_analyzer_proc

  def start_producer_processes(self) -> list[BaseProcess]:
    use_bandpass = not (
      self._cfg.model_conf.sig_fmin == self._cfg.filtering_conf.bandpass_fmin
      and self._cfg.model_conf.sig_fmax == self._cfg.filtering_conf.bandpass_fmax
    )

    producer_processes = [
      self._ctx.Process(
        target=Producer(
          session_id=self._session_id,
          input_queue=self._res.producer_resources.input_queue,
          batch_size=self._cfg.processing_conf.batch_size,
          all_finished=self._res.producer_resources.all_finished,
          n_slots=self._cfg.processing_conf.n_slots,
          prd_ring_access_lock=self._res.producer_resources.ring_access_lock,
          prod_stats_queue=self._res.stats_resources.prd_stats_queue,
          rf_file_indices=self._res.ring_buffer_resources.rf_file_indices,
          rf_segment_indices=self._res.ring_buffer_resources.rf_segment_indices,
          rf_audio_samples=self._res.ring_buffer_resources.rf_audio_samples,
          rf_batch_sizes=self._res.ring_buffer_resources.rf_batch_sizes,
          rf_flags=self._res.ring_buffer_resources.rf_flags,
          logging_queue=self._res.logging_resources.logging_queue,
          logging_level=self._res.logging_resources.logging_level,
          sem_free_slots=self._res.ring_buffer_resources.sem_free_slots,
          sem_filled_slots=self._res.ring_buffer_resources.sem_filled_slots,
          segment_duration_s=self._cfg.model_conf.segment_size_s,
          overlap_duration_s=self._cfg.processing_conf.overlap_duration_s,
          speed=self._cfg.processing_conf.speed,
          target_sample_rate=self._cfg.model_conf.sample_rate,
          use_bandpass=use_bandpass,
          bandpass_fmax=self._cfg.filtering_conf.bandpass_fmax,
          bandpass_fmin=self._cfg.filtering_conf.bandpass_fmin,
          fmin=self._cfg.model_conf.sig_fmin,
          fmax=self._cfg.model_conf.sig_fmax,
          max_segment_idx_ptr=self._res.analyzer_resources.max_segment_idx_ptr,
          prod_done_ptr=self._res.producer_resources.n_finished_pointer,
          n_producers=self._res.producer_resources.n_producers,
          cancel_event=self._res.processing_resources.cancel_event,
          end_event=self._res.processing_resources.end_event,
          start_signal=self._res.producer_resources.start_signals[i],
          finish_signal=self._res.producer_resources.finish_signals[i],
          unprocessed_inputs_queue=self._res.producer_resources.unprocessed_inputs_queue,
          start_method=self._cfg.start_method,
          completion_queue=self._res.file_completion_resources.marker_queue,
        ),
        name=f"{self._session_hash}-Producer-{i}",
        daemon=True,
      )
      for i in range(self._res.producer_resources.n_producers)
    ]

    for p in producer_processes:
      p.start()

    assert self._producer_processes is None
    self._producer_processes = producer_processes
    return producer_processes

  def start_worker_processes(self) -> list[BaseProcess]:
    try:
      self._res.worker_resources.backend_loader.load_backend_in_main_process_if_possible(
        self._res.worker_resources.devices,
        self._cfg.processing_conf.half_precision,
        self._cfg.start_method,
      )
    except Exception as exc:
      raise RuntimeError(f"Error during backend initialization: {exc}") from exc

    worker_processes = [
      self._ctx.Process(
        target=w,
        name=f"{self._session_hash}-Worker-{i}",
        daemon=True,
      )
      for i, w in enumerate(
        self._strategy.create_workers(
          self._session_id, self._cfg, self._specific_cfg, self._res
        )
      )
    ]

    for w in worker_processes:
      w.start()

    assert self._worker_processes is None
    self._worker_processes = worker_processes
    return worker_processes

  def start_processing(
    self, input_data: list[Path] | list[tuple[np.ndarray, int]]
  ) -> None:
    res = self._res
    # start file analyzer
    self._logger.debug("[START_SIG] Starting file analyzer...")
    res.analyzer_resources.start_signal.set()
    res.analyzer_resources.input_queue.put(input_data, block=True)

    # start producers
    self._logger.debug("[START_SIG] Starting producers...")
    for i in range(res.producer_resources.n_producers):
      res.producer_resources.start_signals[i].set()

    # set input data for producers
    self._logger.debug("Feeding input data to producers...")
    for input_idx, inp_data in enumerate(input_data):
      res.producer_resources.input_queue.put((input_idx, inp_data), block=True)
    for _ in range(res.producer_resources.n_producers):
      res.producer_resources.input_queue.put(None, block=True)

    # start workers
    self._logger.debug("[START_SIG] Starting workers...")
    for i in range(self._cfg.processing_conf.workers):
      res.worker_resources.start_signals[i].set()

    # start performance tracker
    self._logger.debug("[START_SIG] Starting performance tracker...")
    if res.stats_resources.track_performance:
      assert res.stats_resources.perf_res_start_signal is not None
      res.stats_resources.perf_res_start_signal.set()

    # start progress dispatcher
    self._logger.debug("[START_SIG] Starting progress dispatcher...")
    if res.stats_resources.use_callback:
      assert res.stats_resources.callback_start_signal is not None
      res.stats_resources.callback_start_signal.set()

  def wait_until_all_finished(self) -> None:
    res = self._res

    # wait for file analyzer to finish
    res.analyzer_resources.finish_signal.wait(timeout=None)

    # wait for producers to finish
    for i in range(res.producer_resources.n_producers):
      res.producer_resources.finish_signals[i].wait(timeout=None)

    # wait for workers to finish
    for i in range(self._cfg.processing_conf.workers):
      res.worker_resources.finish_signals[i].wait(timeout=None)

    # wait for performance tracker to finish
    if res.stats_resources.track_performance:
      assert res.stats_resources.perf_res_finish_signal is not None
      res.stats_resources.perf_res_finish_signal.wait(timeout=None)

    # wait for progress dispatcher to finish
    if res.stats_resources.use_callback:
      assert res.stats_resources.callback_finish_signal is not None
      res.stats_resources.callback_finish_signal.wait(timeout=None)

  def run_consumer(
    self,
    result_tensor: AcousticTensorBase,
    inputs: list[Path] | None = None,
    *,
    completion_active: bool = False,
  ) -> None:
    fc = self._res.file_completion_resources
    marker_queue = fc.marker_queue if completion_active else None
    dispatch_queue = fc.dispatch_queue if completion_active else None
    consumer = Consumer(
      session_id=self._session_id,
      n_workers=self._cfg.processing_conf.workers,
      worker_queue=self._res.worker_resources.results_queue,
      tensor=result_tensor,
      cancel_event=self._res.processing_resources.cancel_event,
      n_inputs=len(inputs) if inputs is not None else 0,
      inputs=inputs,
      completion_marker_queue=marker_queue,
      completion_dispatch_queue=dispatch_queue,
    )
    consumer()

  def start(self) -> None:
    self.start_file_analyzer_thread()
    self.start_producer_processes()
    self.start_worker_processes()

    if self._res.stats_resources.track_performance:
      self.start_performance_tracker_process()

    if self._res.stats_resources.use_callback:
      self.start_progress_dispatcher_thread()

    if self._res.file_completion_resources.enabled:
      self.start_file_completion_dispatcher_thread()

  def join(self) -> None:
    logger = get_logger_from_session(self._session_id, __name__)

    logger.debug("Joining file analyzer thread...")
    assert self._analyzer_thread is not None
    self._analyzer_thread.join()
    self._analyzer_thread = None
    logger.debug("File analyzer finished.")

    if self._res.processing_resources.cancel_event.is_set():
      self._join_processes_after_cancel(logger)
    else:
      logger.debug("Joining producer processes...")
      assert self._producer_processes is not None
      for p in self._producer_processes:
        p.join()
        logger.debug(f"Producer '{p.name}' finished.")
      self._producer_processes = None
      logger.debug("All producers finished.")

      logger.debug("Joining worker processes...")
      assert self._worker_processes is not None
      for w in self._worker_processes:
        w.join()
        logger.debug(f"Worker '{w.name}' finished.")
      self._worker_processes = None
      logger.debug("All workers finished.")

      if self._res.stats_resources.track_performance:
        logger.debug("Joining performance tracker process...")
        assert self._perf_tracker_process is not None
        self._perf_tracker_process.join()
        self._perf_tracker_process = None
        logger.debug("Performance tracker finished.")

    if self._res.stats_resources.use_callback:
      logger.debug("Joining dispatcher thread...")
      assert self._progress_dispatcher_thread is not None
      self._progress_dispatcher_thread.join()
      self._progress_dispatcher_thread = None
      logger.debug("Dispatcher thread finished.")

    if self._res.file_completion_resources.enabled:
      logger.debug("Joining file completion dispatcher thread...")
      assert self._file_completion_thread is not None
      self._file_completion_thread.join()
      self._file_completion_thread = None
      logger.debug("File completion dispatcher thread finished.")

  def _join_processes_after_cancel(
    self, logger: Logger, grace_period_s: float = 30.0
  ) -> None:
    """Join child processes after a cancelled run without deadlocking.

    A cancelled run leaves undelivered items in the child-to-parent queues
    (results, stats, completion markers, unprocessed inputs): the consumer and
    the performance tracker stop reading when the cancel event is set. A child
    process cannot exit while its queue feeder threads still hold buffered
    data — its exit handler joins the feeders, which block on the full pipe —
    so a plain ``join()`` waits forever. Drain every parent-side queue while
    joining so the children can flush and exit, and terminate any process
    that still lingers after the grace period. The drained data is discarded,
    which is fine: the run was cancelled.
    """
    logger.debug("Joining processes after cancellation (draining queues)...")

    processes: list[BaseProcess] = [
      *(self._producer_processes or []),
      *(self._worker_processes or []),
    ]

    if self._perf_tracker_process is not None:
      processes.append(self._perf_tracker_process)

    stats_res = self._res.stats_resources
    queues: list[Queue] = [
      self._res.producer_resources.unprocessed_inputs_queue,
      self._res.worker_resources.results_queue,
      self._res.file_completion_resources.marker_queue,
      stats_res.wkr_stats_queue,
      stats_res.prd_stats_queue,
      stats_res.perf_res_queue,
      stats_res.callback_queue,
    ]
    queues = [q for q in queues if q is not None]

    deadline = time.monotonic() + grace_period_s
    terminated = False

    while True:
      alive = [p for p in processes if p.is_alive()]

      if not alive:
        break

      for q in queues:
        with suppress(Empty):
          while True:
            q.get_nowait()

      if time.monotonic() >= deadline:
        for p in alive:
          logger.warning(
            f"Process '{p.name}' did not exit after cancellation; terminating."
          )
          p.terminate()
        terminated = True
        break

      time.sleep(0.05)

    # Reap every process. Ones that exited on their own join instantly; ones we
    # just terminated get a bounded wait and are then SIGKILLed if they ignored
    # SIGTERM, so a single unresponsive child can never hang teardown.
    for p in processes:
      p.join(timeout=_TERMINATE_JOIN_TIMEOUT_S if terminated else None)
      if p.is_alive():
        logger.warning(f"Process '{p.name}' ignored termination; killing.")
        p.kill()
        p.join()
      logger.debug(f"Process '{p.name}' finished.")

    self._producer_processes = None
    self._worker_processes = None
    self._perf_tracker_process = None
    logger.debug("All processes finished after cancellation.")

  def join_logging(self) -> None:
    assert self._logging_thread is not None
    self._logging_thread.join()
    self._logging_thread = None

  def close_queues(self) -> None:
    """Release the parent's handles on the multiprocessing queues.

    Call once during teardown, after all child processes and the logging thread
    have been joined. Closing is non-blocking (``cancel_join_thread`` first, so
    ``close`` never waits on a feeder) and lets the OS reclaim the pipes and
    semaphores promptly instead of leaving it to garbage collection -- which the
    caller may skip entirely (e.g. ``os._exit``).
    """
    res = self._res
    stats = res.stats_resources
    queues: list[Queue | None] = [
      res.producer_resources.input_queue,
      res.producer_resources.unprocessed_inputs_queue,
      res.worker_resources.results_queue,
      res.file_completion_resources.marker_queue,
      res.logging_resources.logging_queue,
      stats.wkr_stats_queue,
      stats.prd_stats_queue,
      stats.perf_res_queue,
      stats.callback_queue,
    ]
    for q in queues:
      if q is not None:
        q.cancel_join_thread()
        q.close()
