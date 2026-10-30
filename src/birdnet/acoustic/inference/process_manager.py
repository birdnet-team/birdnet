from __future__ import annotations

import contextvars
import functools
import multiprocessing as mp
import os
import threading
import time
from collections.abc import Iterator
from contextlib import suppress
from logging import Logger
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Event as EventType
from pathlib import Path
from queue import Empty
from typing import Literal

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
from birdnet.acoustic.inference.core.sync import (
  PROMISED_MESSAGE_DEADLINE_S as _PROMISED_MESSAGE_DEADLINE_S,
)
from birdnet.acoustic.inference.core.sync import (
  READER_CLOSE_TIMEOUT_S,
  ChildToParentQueue,
  Deadline,
  FailureSlot,
  ThreadedQueueReader,
  read_failure,
  readable,
)
from birdnet.acoustic.inference.core.tensor import AcousticTensorBase
from birdnet.acoustic.inference.file_writer import QueueFileWriter
from birdnet.acoustic.inference.resources import PipelineResources
from birdnet.acoustic.inference.strategy import InferenceStrategyBase
from birdnet.core.base import get_session_id_hash

# One budget for the whole teardown: every wait in it draws what is left, so a
# child that cannot exit costs the caller at most this much. A healthy teardown
# uses a small fraction of a second of it.
TEARDOWN_BUDGET_S = 30.0
# Kept back from the children's grace, so that terminating what is still there
# and then killing what ignores that still fit inside the budget.
_KILL_RESERVE_S = 5.0
# Kept back from the terminate wait for the thread joins after the children: a
# healthy drainer, dispatcher, reader or log writer needs milliseconds.
_TAIL_RESERVE_S = 2.0
# How long teardown waits for the log writer after the stop event. Every child
# is joined by then, so the writer has only what is left in the pipe to write;
# one that has not stopped in this time is blocked on a half-written record
# from a killed child and never will.
_LOG_WRITER_STOP_TIMEOUT_S = 5.0
# How often the parent looks up from an otherwise unbounded wait to check that
# its children are still alive. Short enough that a dead child is reported
# promptly, long enough to be free next to the work being waited on.
_LIVENESS_POLL_INTERVAL_S = 1.0
# How long teardown waits for the queue drainers to notice they are done. They
# only ever have one ``get_nowait`` left to finish, so anything still running
# after this is stuck for good and is left to the daemon-thread machinery.
_DRAIN_STOP_TIMEOUT_S = 1.0


class PipelineThreadError(RuntimeError):
  """The manager's own diagnosis of a pipeline thread of this process."""


def _drain_queue(q: ChildToParentQueue, stop: threading.Event) -> None:
  """Discard everything a queue holds until told to stop.

  Runs on its own thread, one per queue, and never on the teardown path itself.
  ``get_nowait`` is not actually non-blocking: it checks that *some* bytes are
  available and then waits for the rest of the frame, so a child killed
  mid-``put`` leaves a truncated message that stops the call for good. Nothing
  is raised -- the other children still hold the pipe's write end, so there is
  no EOF either -- which means no handler can recover from it and only keeping
  it off the teardown path can (issue #77). One thread per queue so a queue that
  does wedge cannot stop the others from being drained.
  """
  reads = readable(q)
  while True:
    # Best-effort by design: this drain only exists to unblock the children so
    # they can exit, and the run has already failed. A payload that arrived
    # intact but cannot be deserialized surfaces as any number of exceptions,
    # and letting one escape would leave the drain dead while teardown still
    # expects it -- so none of them are enumerated.
    with suppress(Exception):
      while True:
        reads.get_nowait()
        # A child that keeps writing would otherwise hold this loop past the
        # stop, and teardown would then record a perfectly healthy drainer as
        # wedged and never close its queue.
        if stop.is_set():
          return
    # Checked after a pass, never before one: children that exit on the first
    # liveness poll stop this thread within milliseconds of starting it, and a
    # pre-check would then let it finish without having drained anything.
    if stop.is_set():
      return
    stop.wait(0.05)


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
    self._child_death_error: ChildProcessError | None = None
    self._run_error: BaseException | None = None
    self._run_error_lock = threading.Lock()
    self._undrainable_queues: list[ChildToParentQueue] = []
    # Readers whose pumps were asked to stop but not yet confirmed gone.
    # Confirming joins a thread that is usually just asleep in its poll, which
    # costs tens of milliseconds -- so it is deferred to the moments that need
    # the answer: before the same queues are read again (start of the next
    # run) and before they are closed (session teardown).
    self._stopped_readers: list[tuple[ThreadedQueueReader, ChildToParentQueue]] = []
    # Set when teardown starts; every wait after that draws from it.
    self._teardown: Deadline | None = None

  @property
  def child_death_error(self) -> ChildProcessError | None:
    """Set when a child was found dead; the session surfaces it to the caller."""
    return self._child_death_error

  @property
  def run_error(self) -> BaseException | None:
    """The first failure caught by a pipeline thread of this process."""
    return self._run_error

  def record_run_error(self, error: BaseException) -> None:
    """Keep the first failure; later ones are usually its consequences."""
    with self._run_error_lock:
      if self._run_error is None:
        self._run_error = error

  def start_file_logging_thread(self) -> threading.Thread:
    logging_listener = threading.Thread(
      target=QueueFileWriter(
        session_id=self._session_id,
        log_queue=self._res.logging_resources.logging_queue,
        logging_level=self._res.logging_resources.logging_level,
        log_file=self._res.logging_resources.session_log_file,
        cancel_event=self._res.processing_resources.cancel_event,
        stop_event=self._res.logging_resources.stop_logging_event,
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
    assert self._res.stats_resources.callback_last_update_signal is not None
    assert self._res.stats_resources.callback_busy_signal is not None
    assert self._res.stats_resources.callback_fn is not None

    dispatcher = ProgressDispatcher(
      session_id=self._session_id,
      callback_fn=self._res.stats_resources.callback_fn,
      check_interval=self._res.processing_resources.update_interval,
      start_signal=self._res.stats_resources.callback_start_signal,
      finish_signal=self._res.stats_resources.callback_finish_signal,
      last_update_signal=self._res.stats_resources.callback_last_update_signal,
      busy_signal=self._res.stats_resources.callback_busy_signal,
      end_event=self._res.processing_resources.end_event,
      callback_queue=self._res.stats_resources.callback_queue,
      cancel_event=self._res.processing_resources.cancel_event,
      processing_finished_event=self._res.processing_resources.processing_finished_event,
      report_error=self.record_run_error,
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
      report_error=self.record_run_error,
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
        processing_finished=self._res.processing_resources.processing_finished_sem,
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
        failure_slot=self._res.stats_resources.perf_failure_slot,
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
        start_signal=self._res.analyzer_resources.start_signal,
        finish_signal=self._res.analyzer_resources.finish_signal,
        report_error=self.record_run_error,
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
          failure_slot=self._res.producer_resources.failure_slots[i],
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
    # A reused session must not surface the previous run's diagnosis -- unless
    # that run is still unresolved. `increment_run_nr` is the last statement of
    # `_run`, so a failed run leaves `is_first_run` True and the next call
    # skips `resources.reset()`, keeping the cancel event set. Clearing here
    # would then replace a precise "worker X died" with a bare "cancelled" on
    # exactly the retry where the user most needs the reason.
    if not res.processing_resources.cancel_event.is_set():
      self._child_death_error = None
      self._run_error = None
    self._logger.debug("[START_SIG] Starting file analyzer...")
    res.analyzer_resources.start_signal.set()
    res.analyzer_resources.input_queue.put(input_data, block=True)

    self._logger.debug("[START_SIG] Starting producers...")
    for i in range(res.producer_resources.n_producers):
      res.producer_resources.start_signals[i].release()

    self._logger.debug("Feeding input data to producers...")
    for input_idx, inp_data in enumerate(input_data):
      res.producer_resources.input_queue.put((input_idx, inp_data), block=True)
    for _ in range(res.producer_resources.n_producers):
      res.producer_resources.input_queue.put(None, block=True)

    self._logger.debug("[START_SIG] Starting workers...")
    for i in range(self._cfg.processing_conf.workers):
      res.worker_resources.start_signals[i].release()

    self._logger.debug("[START_SIG] Starting performance tracker...")
    if res.stats_resources.track_performance:
      assert res.stats_resources.perf_res_start_signal is not None
      res.stats_resources.perf_res_start_signal.release()

    self._logger.debug("[START_SIG] Starting progress dispatcher...")
    if res.stats_resources.use_callback:
      assert res.stats_resources.callback_start_signal is not None
      res.stats_resources.callback_start_signal.set()

  def wake_idle_children(self) -> None:
    """Release every idle wait on a start signal so the end event is seen now.

    Idle children and threads poll their start signal with a 1 s timeout and
    look at the end event only when it expires. The cross-process signals are
    semaphores because a release never blocks: ``Event.set()`` waits for every
    registered waiter to wake, forever if that waiter was killed inside its
    wait. The loops check the end event on every wake, so a wake is not a run
    start.
    """
    res = self._res
    # A released start signal is a run start unless the end event is already set.
    assert res.processing_resources.end_event.is_set()
    res.analyzer_resources.start_signal.set()
    for start_signal in res.producer_resources.start_signals:
      start_signal.release()
    for start_signal in res.worker_resources.start_signals:
      start_signal.release()
    if res.stats_resources.track_performance:
      assert res.stats_resources.perf_res_start_signal is not None
      res.stats_resources.perf_res_start_signal.release()
    if res.stats_resources.use_callback:
      assert res.stats_resources.callback_start_signal is not None
      res.stats_resources.callback_start_signal.set()
    if res.file_completion_resources.enabled:
      assert res.file_completion_resources.start_signal is not None
      res.file_completion_resources.start_signal.set()

  def _iter_children_with_finish_signals(
    self,
  ) -> Iterator[tuple[BaseProcess, EventType]]:
    """Every child process paired with the signal it sets when it is done."""
    res = self._res
    # strict=True on purpose: the lists are created 1:1 from the same counts, so
    # a length mismatch is a broken invariant. Silently zipping to the shorter
    # one would drop a child from the liveness check -- exactly the hang this is
    # here to prevent -- so fail loudly instead.
    if self._producer_processes is not None:
      yield from zip(
        self._producer_processes, res.producer_resources.finish_signals, strict=True
      )
    if self._worker_processes is not None:
      yield from zip(
        self._worker_processes, res.worker_resources.finish_signals, strict=True
      )
    perf_finish_signal = res.stats_resources.perf_res_finish_signal
    if self._perf_tracker_process is not None and perf_finish_signal is not None:
      yield self._perf_tracker_process, perf_finish_signal

  def _children_with_failure_slots(self) -> list[tuple[BaseProcess, FailureSlot]]:
    res = self._res
    slots: list[tuple[BaseProcess, FailureSlot]] = []
    if self._producer_processes is not None:
      slots += zip(
        self._producer_processes, res.producer_resources.failure_slots, strict=True
      )
    if self._worker_processes is not None:
      slots += zip(
        self._worker_processes, res.worker_resources.failure_slots, strict=True
      )
    perf_slot = res.stats_resources.perf_failure_slot
    if self._perf_tracker_process is not None and perf_slot is not None:
      slots.append((self._perf_tracker_process, perf_slot))
    return slots

  def child_failure(self) -> tuple[str, str] | None:
    """A child that reported its own exception, in process order: name and text."""
    for process, slot in self._children_with_failure_slots():
      text = read_failure(slot)
      if text is not None:
        return process.name, text
    return None

  def raise_if_child_died(self) -> None:
    """Fail fast when a child process is gone without having signalled.

    Every parent-side wait in a healthy run is unbounded on purpose: the run
    takes as long as the audio requires. That only holds while the children are
    actually working. A child killed by the OOM killer, or dying in a native
    crash during its TensorFlow import, never sets its finish signal and never
    puts its sentinel on the results queue, so an unbounded wait would block
    forever with no output at all. Checking liveness turns that silent hang
    into an error naming the process.

    A process that exited *after* setting its finish signal is not an error:
    that is the normal end-of-session shutdown path.

    Marks the run cancelled before raising. The cancel event is what routes
    teardown through ``_join_processes_after_cancel``, which drains the
    child-to-parent queues while joining; the plain path assumes every child
    still delivers its buffered data, which a dead one never will. Setting it
    here keeps both callers consistent -- the consumer would set it via its own
    exception handler, ``wait_until_all_finished`` has no such handler.

    The error is also stored so the session can surface it to the caller: the
    consumer's broad ``except Exception`` swallows it into a generic
    "cancelled", and the process name and exit code are the only actionable
    part.
    """
    # A child that wrote its failure and left (exit code 0, cancel event set)
    # has explained itself; the slot is the diagnosis, not a mystery shutdown.
    explained = {
      id(p) for p, slot in self._children_with_failure_slots() if read_failure(slot)
    }
    for process, finish_signal in self._iter_children_with_finish_signals():
      if process.is_alive() or finish_signal.is_set():
        continue
      if process.exitcode == 0 and id(process) in explained:
        continue
      if process.exitcode == 0:
        # Children exit cleanly on their own when they see the cancel or end
        # event while parked (see WorkerBase.run_main_loop). Reaching this
        # means the session was already torn down and is being reused, which
        # has nothing to do with memory -- say so rather than blaming the OOM
        # killer.
        message = (
          f"Pipeline process '{process.name}' has already shut down, so this "
          f"session can no longer run. Usually the session was cancelled or "
          f"closed and is being reused, which is not supported -- create a "
          f"new session with 'predict_session(..)'/'encode_session(..)'. If "
          f"this is the first run, the process failed during start-up "
          f"instead; the log holds the reason."
        )
      else:
        message = (
          f"Pipeline process '{process.name}' exited unexpectedly with exit "
          f"code {process.exitcode} before it finished its work. This usually "
          f"means the process was killed from the outside (e.g. by the OOM "
          f"killer when memory ran out) or crashed in native code. Reducing "
          f"'n_workers' or 'batch_size' lowers the memory needed per run."
        )
      self._logger.error(message)
      error = ChildProcessError(message)
      self._child_death_error = error
      self._res.processing_resources.cancel_event.set()
      raise error

  def _wait_for_finish_signal(self, finish_signal: EventType | threading.Event) -> None:
    """Wait for one finish signal, giving up if the run cannot finish.

    Two ways it cannot: a child process died, or the run was cancelled.

    The cancel case matters for the threads this wait covers, which the
    liveness check cannot see because they are not processes: one that fails
    sets the cancel event and may return *without* setting its finish signal,
    and waiting on that signal would then block forever.
    (``ProgressDispatcher`` has its own wait below; ``FileCompletionDispatcher``
    catches its callback's exception internally and does signal, so it is safe
    either way.)

    Both conditions return rather than raise, so the caller's
    ``_raise_if_cancelled`` reports every failure the same way. Letting the
    ``ChildProcessError`` escape from here instead would surface an ``OSError``
    to a caller that gets a ``RuntimeError`` from every other path.
    """
    while not finish_signal.wait(timeout=_LIVENESS_POLL_INTERVAL_S):
      if self._res.processing_resources.cancel_event.is_set():
        self._logger.debug("Run cancelled while waiting for a finish signal.")
        return
      try:
        self.raise_if_child_died()
      except ChildProcessError:
        # Already logged, stored and marked cancelled by the check itself.
        return

  def wait_for_completion_dispatcher(
    self, finish_signal: EventType | threading.Event
  ) -> None:
    """Wait for the per-file completion dispatcher, cancel- and liveness-aware.

    Same guard as the finish-signal waits: this dispatcher is a thread, so a
    callback that raises sets the cancel event and returns without signalling.
    """
    self._wait_for_finish_signal(finish_signal)

  def wait_until_all_finished(self) -> None:
    res = self._res

    self._wait_for_finish_signal(res.analyzer_resources.finish_signal)

    for i in range(res.producer_resources.n_producers):
      self._wait_for_finish_signal(res.producer_resources.finish_signals[i])

    for i in range(self._cfg.processing_conf.workers):
      self._wait_for_finish_signal(res.worker_resources.finish_signals[i])

    if res.stats_resources.track_performance:
      assert res.stats_resources.perf_res_finish_signal is not None
      self._wait_for_finish_signal(res.stats_resources.perf_res_finish_signal)

    if res.stats_resources.use_callback:
      assert res.stats_resources.callback_last_update_signal is not None
      assert res.stats_resources.callback_busy_signal is not None
      assert res.stats_resources.callback_finish_signal is not None
      self._wait_for_progress_dispatcher(
        res.stats_resources.callback_last_update_signal,
        res.stats_resources.callback_busy_signal,
        res.stats_resources.callback_finish_signal,
      )

  def _wait_for_progress_dispatcher(
    self,
    last_update_signal: threading.Event,
    busy_signal: threading.Event,
    finish_signal: threading.Event,
  ) -> None:
    """Wait for the dispatcher to read the tracker's last update, then to finish.

    Phase 1 waits for the dispatcher to read the tracker's closing update, the
    one flagged finished. The tracker hands that update to its queue's feeder
    thread and then sets its finish signal, so a tracker killed after the
    signal and before the feeder wrote the update leaves the signal set and
    the update unsent. The liveness check skips a signalled child on purpose
    (a finished child may exit), so this is ``read_promised`` in another
    shape: the same deadline, counted from the tracker's finish signal, which
    the caller has just seen. The deadline is for the message, not for the
    caller's code: while the dispatcher is inside the progress callback
    (``busy_signal``) it cannot read the update, and every poll that finds it
    there starts the deadline afresh. The update counts as lost only once the
    dispatcher was free to read it for a whole deadline, less at most one poll.

    Phase 2 waits for the dispatcher to finish, that is for the progress
    callback on the closing update to return. It has no deadline: the update
    has arrived, and the callback is the caller's own code.

    The dispatcher reports whatever ends it and cancels the run, which ends
    either phase. The liveness check cannot see a thread, so both phases also
    watch the dispatcher themselves, for one that ends any other way. Returns
    rather than raises, for the reason ``_wait_for_finish_signal`` gives.
    """
    thread = self._progress_dispatcher_thread
    assert thread is not None
    outcome = self._poll_progress_dispatcher(
      thread,
      last_update_signal,
      Deadline(_PROMISED_MESSAGE_DEADLINE_S),
      busy_signal,
    )
    if outcome == "expired":
      self._record_lost_last_update()
    elif outcome == "dead":
      self._record_dispatcher_death(thread, "without delivering")
    elif (
      outcome == "set"
      and self._poll_progress_dispatcher(thread, finish_signal, None) == "dead"
    ):
      self._record_dispatcher_death(thread, "without finishing after")

  def _record_lost_last_update(self) -> None:
    tracker = self._perf_tracker_process
    assert tracker is not None
    if tracker.is_alive():
      message = (
        f"Pipeline process '{tracker.name}' finished its work, but its last "
        f"progress update did not arrive within "
        f"{_PROMISED_MESSAGE_DEADLINE_S:.0f} s, although the process is still "
        f"alive. The update was lost on its way to this process."
      )
    else:
      message = (
        f"Pipeline process '{tracker.name}' exited unexpectedly with exit code "
        f"{tracker.exitcode} after it finished its work, with its last progress "
        f"update unsent: the update did not arrive within "
        f"{_PROMISED_MESSAGE_DEADLINE_S:.0f} s. This usually means the process "
        f"was killed from the outside (e.g. by the OOM killer when memory ran "
        f"out) or crashed in native code."
      )
    self.record_child_death(message)

  def _poll_progress_dispatcher(
    self,
    thread: threading.Thread,
    signal: threading.Event,
    deadline: Deadline | None,
    busy_signal: threading.Event | None = None,
  ) -> Literal["set", "cancelled", "dead", "expired"]:
    """Wait for one of the dispatcher's signals, watching the thread that sets it.

    The deadline starts afresh at every poll that finds ``busy_signal`` set.
    """
    cancel_event = self._res.processing_resources.cancel_event
    while True:
      timeout = _LIVENESS_POLL_INTERVAL_S
      if deadline is not None:
        timeout = min(timeout, deadline.remaining())
      if signal.wait(timeout):
        return "set"
      # Read before the signal and the cancel: a thread that set either on its
      # way out did so before it ended, so it is not taken for a dead one.
      alive = thread.is_alive()
      if signal.is_set():
        return "set"
      if cancel_event.is_set():
        self._logger.debug("Run cancelled while waiting for the progress dispatcher.")
        return "cancelled"
      if not alive:
        return "dead"
      if deadline is not None:
        if busy_signal is not None and busy_signal.is_set():
          deadline = Deadline(deadline.budget_s)
        if deadline.remaining() <= 0:
          return "expired"

  def _record_dispatcher_death(self, thread: threading.Thread, what: str) -> None:
    """Fail the run for a dispatcher thread that ended without signalling.

    The dispatcher reports its own failures, its callback's included; this is
    for one that ended without doing so. Not a child process, so the diagnosis
    goes to the run-error slot.
    """
    message = (
      f"Progress dispatcher thread '{thread.name}' ended {what} the "
      f"performance tracker's last progress update and reported no error."
    )
    self.record_run_error(PipelineThreadError(message))
    self._logger.error(message)
    self._res.processing_resources.cancel_event.set()

  def run_consumer(
    self,
    result_tensor: AcousticTensorBase,
    inputs: list[Path] | None = None,
    *,
    completion_active: bool = False,
  ) -> None:
    fc = self._res.file_completion_resources
    results_queue = self._res.worker_resources.results_queue
    marker_queue = fc.marker_queue if completion_active else None
    dispatch_queue = fc.dispatch_queue if completion_active else None

    # The consumer never touches the queues directly: a plain get can block
    # forever on a message a killed worker left half-written, and this loop is
    # the run itself (issue #83). Each reader is one sacrificial daemon thread.
    results_reader = ThreadedQueueReader(
      results_queue, f"{self._session_hash}-ResultsReader"
    )
    marker_reader = (
      ThreadedQueueReader(marker_queue, f"{self._session_hash}-MarkerReader")
      if marker_queue is not None
      else None
    )
    try:
      consumer = Consumer(
        session_id=self._session_id,
        n_workers=self._cfg.processing_conf.workers,
        results=results_reader,
        tensor=result_tensor,
        cancel_event=self._res.processing_resources.cancel_event,
        n_inputs=len(inputs) if inputs is not None else 0,
        inputs=inputs,
        markers=marker_reader,
        completion_dispatch_queue=dispatch_queue,
        check_children_alive=self.raise_if_child_died,
        all_workers_finished=lambda: all(
          sig.is_set() for sig in self._res.worker_resources.finish_signals
        ),
        report_child_death=self.record_child_death,
        report_error=self.record_run_error,
      )
      consumer()
    finally:
      self._stop_reader(results_reader, results_queue)
      if marker_reader is not None:
        assert marker_queue is not None
        self._stop_reader(marker_reader, marker_queue)

  def _stop_reader(self, reader: ThreadedQueueReader, q: ChildToParentQueue) -> None:
    """Ask a queue reader to stop, without waiting for it.

    Waiting here would put a fixed per-call cost on every ``session.run``;
    ``reap_stopped_readers`` does the bounded wait at the two points that need
    the answer.
    """
    reader.stop()
    self._stopped_readers.append((reader, q))

  def reap_stopped_readers(self) -> None:
    """Confirm stopped readers are gone; record the wedged ones.

    Called before the next run reads these queues -- a not-yet-exited pump
    could otherwise steal that run's first message -- and before the queues
    are closed. A pump that does not confirm is stuck inside ``_recv_bytes``
    on a message a killed child never finished; the run that produced it has
    already failed, so the cost is a parked daemon thread, and its queue must
    not be closed underneath the blocked read (same reasoning as
    ``_collect_wedged_drainers``). A side effect worth knowing during
    teardown: the wedged pump holds the queue's read lock, so the cancel-path
    drainer on that queue drains nothing and children with buffered data are
    terminated at the grace period instead of flushing.
    """
    for reader, q in self._stopped_readers:
      timeout = READER_CLOSE_TIMEOUT_S
      if self._teardown is not None:
        timeout = min(timeout, self._teardown.remaining())
      if reader.confirm_stopped(timeout=timeout):
        continue
      self._logger.warning(
        "A queue reader is still blocked on a message a killed child never "
        "finished writing; leaving that queue open."
      )
      if not any(q is seen for seen in self._undrainable_queues):
        self._undrainable_queues.append(q)
    self._stopped_readers.clear()

  def record_child_death(self, message: str) -> None:
    """Store a diagnosis and fail the run, for a loss the liveness check cannot see.

    A child dead *after* its finish signal, its last message unflushed, is
    found by whoever waited for that message: the consumer, which has no
    handle on this manager's error slot, or the first phase of the dispatcher
    wait above, for the tracker's closing update. Without this the caller
    would see a bare "Analysis was cancelled" for exactly the failure this
    machinery exists to name.
    """
    self._logger.error(message)
    self._child_death_error = ChildProcessError(message)
    self._res.processing_resources.cancel_event.set()

  def read_promised(self, q: ChildToParentQueue, n: int, what: str) -> list[object]:
    """Read ``n`` messages that finished children have already promised.

    Called only after every sender set its finish signal, so each message is
    either in flight or lost with a child that died on its way out: ``put``
    hands off to a feeder thread, so a child killed *after* signalling can die
    with the message unsent -- or half-written, which no ``Queue.get`` timeout
    bounds. The liveness check cannot flag that child (its signal is set), so
    a deadline is the only honest wait.
    """
    reader = ThreadedQueueReader(q, f"{self._session_hash}-Promised")
    try:
      out: list[object] = []
      deadline = time.monotonic() + _PROMISED_MESSAGE_DEADLINE_S
      while len(out) < n:
        if self._res.processing_resources.cancel_event.is_set():
          raise RuntimeError(f"The run was cancelled while collecting {what}.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
          message = (
            f"Only {len(out)} of {n} {what} arrived within "
            f"{_PROMISED_MESSAGE_DEADLINE_S:.0f} s of the senders finishing. "
            f"Either a pipeline process died while delivering its last "
            f"message (e.g. killed by the OOM killer on its way out), or an "
            f"earlier failed run of this session left the queue unreadable. "
            f"Please check the logs: "
            f"{self._res.logging_resources.session_log_file.absolute()}"
          )
          self._logger.error(message)
          # Stored like any other child death, so the caller gets the same
          # error shape as every other failure of this kind.
          self._child_death_error = ChildProcessError(message)
          self._res.processing_resources.cancel_event.set()
          raise RuntimeError(message)
        with suppress(Empty):
          out.append(reader.get(timeout=min(1.0, remaining)))
      return out
    finally:
      self._stop_reader(reader, q)

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
    """Bring every child and dispatcher down inside ``TEARDOWN_BUDGET_S``.

    Each wait takes what is left of the one deadline, which ``join_logging``
    and ``close_queues`` draw from afterwards, so whatever cannot finish, a
    worker stuck in a model load, a child with buffered data and no reader, a
    wedged log writer, costs the budget once and not per step. Only a killed
    child that the kernel cannot reap yet (uninterruptible sleep) holds the
    call longer.
    """
    logger = get_logger_from_session(self._session_id, __name__)
    deadline = Deadline(TEARDOWN_BUDGET_S)
    self._teardown = deadline

    logger.debug("Joining file analyzer thread...")
    assert self._analyzer_thread is not None
    self._analyzer_thread.join(timeout=deadline.remaining())
    if self._analyzer_thread.is_alive():
      logger.warning("The file analyzer thread did not stop; continuing without it.")
    self._analyzer_thread = None
    logger.debug("File analyzer finished.")

    if self._res.processing_resources.cancel_event.is_set():
      self._join_processes_after_cancel(logger, deadline)
    else:
      self._join_processes(logger, deadline)

    if self._res.stats_resources.use_callback:
      logger.debug("Joining dispatcher thread...")
      assert self._progress_dispatcher_thread is not None
      self._join_reader_thread(
        logger,
        self._progress_dispatcher_thread,
        self._res.stats_resources.callback_queue,
        deadline,
      )
      self._progress_dispatcher_thread = None
      logger.debug("Dispatcher thread finished.")

    if self._res.file_completion_resources.enabled:
      logger.debug("Joining file completion dispatcher thread...")
      assert self._file_completion_thread is not None
      # Its queue is an in-process queue.Queue, which cannot carry a half-
      # written message, so there is nothing to leave open.
      self._join_reader_thread(logger, self._file_completion_thread, None, deadline)
      self._file_completion_thread = None
      logger.debug("File completion dispatcher thread finished.")

  def _children(self) -> list[BaseProcess]:
    processes: list[BaseProcess] = [
      *(self._producer_processes or []),
      *(self._worker_processes or []),
    ]
    if self._perf_tracker_process is not None:
      processes.append(self._perf_tracker_process)
    return processes

  def _join_processes(self, logger: Logger, deadline: Deadline) -> None:
    """Join the children of a run that ended cleanly.

    ``wake_idle_children`` has released their start-signal waits, so they see
    the end event at once; the budget covers a child that never gets there,
    say one still inside a model load.
    """
    logger.debug("Joining child processes...")
    self._reap_children(logger, self._children(), deadline)
    self._producer_processes = None
    self._worker_processes = None
    self._perf_tracker_process = None
    logger.debug("All child processes finished.")

  def _reap_children(
    self, logger: Logger, processes: list[BaseProcess], deadline: Deadline
  ) -> None:
    """Join, then terminate, then kill: the budget less the reserves for each."""
    grace_end = time.monotonic() + max(0.0, deadline.remaining() - _KILL_RESERVE_S)
    for p in processes:
      # A blocking join, not a poll: a child that exits is reaped the moment
      # it does, and the grace is shared, so the joins together never exceed it.
      p.join(timeout=max(0.0, grace_end - time.monotonic()))
    terminated = [p for p in processes if p.is_alive()]
    for p in terminated:
      logger.warning(
        f"Process '{p.name}' did not exit within the teardown budget; terminating."
      )
      p.terminate()
    for p in terminated:
      # SIGKILL after that cannot be ignored; the tail reserve stays for the
      # thread joins behind the children.
      p.join(timeout=max(0.0, deadline.remaining() - _TAIL_RESERVE_S))
      if p.is_alive():
        logger.warning(f"Process '{p.name}' ignored termination; killing.")
        p.kill()
        p.join()
    for p in processes:
      logger.debug(f"Process '{p.name}' finished.")

  def _join_reader_thread(
    self,
    logger: Logger,
    thread: threading.Thread,
    q: ChildToParentQueue | None,
    deadline: Deadline,
  ) -> None:
    """Join a thread that reads a child-to-parent queue, without waiting forever.

    ``ProgressDispatcher`` reads its queue with ``get(block=True, timeout=..)``,
    and that timeout bounds acquiring the read lock and the poll but *not*
    ``_recv_bytes``. So a performance tracker killed mid-``put`` leaves a
    message this thread can never finish reading, with nothing raised -- the
    same trap as the cancel-path drain (issue #77). These are daemon threads,
    so parking one costs the closing progress callback; joining it costs
    teardown.
    """
    thread.join(timeout=deadline.remaining())
    if not thread.is_alive():
      return
    logger.warning(
      f"'{thread.name}' is still blocked reading its queue; continuing "
      f"teardown without it."
    )
    if q is not None:
      # Left open for the same reason a wedged drainer's queue is; see
      # _collect_wedged_drainers.
      self._undrainable_queues.append(q)

  def _join_processes_after_cancel(self, logger: Logger, deadline: Deadline) -> None:
    """Join child processes after a cancelled run without deadlocking.

    A cancelled run leaves undelivered items in the child-to-parent queues
    (results, stats, completion markers, unprocessed inputs): the consumer and
    the performance tracker stop reading when the cancel event is set. A child
    process cannot exit while its queue feeder threads still hold buffered
    data — its exit handler joins the feeders, which block on the full pipe —
    so a plain ``join()`` waits forever. Drain every parent-side queue while
    joining so the children can flush and exit, and terminate any process
    that still lingers when the budget runs out. The drained data is
    discarded, which is fine: the run was cancelled.

    The drain runs on background threads rather than here, because a single
    ``get_nowait`` can block for good on a message truncated by a killed child
    (see ``_drain_queue``). Teardown must stay bounded whatever the drain does,
    so it only ever *starts* and *stops* the drainers and never waits on their
    progress.
    """
    logger.debug("Joining processes after cancellation (draining queues)...")

    processes = self._children()

    stats_res = self._res.stats_resources
    optional_queues: list[ChildToParentQueue | None] = [
      self._res.producer_resources.unprocessed_inputs_queue,
      self._res.worker_resources.results_queue,
      self._res.file_completion_resources.marker_queue,
      stats_res.wkr_stats_queue,
      stats_res.prd_stats_queue,
      stats_res.perf_res_queue,
      stats_res.callback_queue,
    ]
    queues: list[ChildToParentQueue] = [q for q in optional_queues if q is not None]

    stop_draining = threading.Event()
    drainers = [
      threading.Thread(
        target=_drain_queue,
        args=(q, stop_draining),
        name=f"{self._session_hash}-CancelDrain-{i}",
        daemon=True,
      )
      for i, q in enumerate(queues)
    ]
    for drainer in drainers:
      drainer.start()

    try:
      self._reap_children(logger, processes, deadline)
    finally:
      stop_draining.set()

    self._collect_wedged_drainers(logger, queues, drainers, deadline)

    self._producer_processes = None
    self._worker_processes = None
    self._perf_tracker_process = None
    logger.debug("All processes finished after cancellation.")

  def _collect_wedged_drainers(
    self,
    logger: Logger,
    queues: list[ChildToParentQueue],
    drainers: list[threading.Thread],
    deadline: Deadline,
  ) -> None:
    """Note which queues a drainer is still stuck inside, and why it matters.

    A drainer blocked on a truncated message holds the queue's reader. Closing
    that reader would not wake it -- a blocked ``read`` is not interrupted by
    the descriptor being closed -- but it would free the descriptor number for
    reuse, and the stuck thread would then be reading from whatever the process
    opens next. Leaving those queues open costs two descriptors and a parked
    daemon thread per teardown that hits this; closing them risks silent data
    corruption elsewhere in the process, so the queues stay open.
    """
    for q, drainer in zip(queues, drainers, strict=True):
      # At most a second each, so one wedged drainer cannot spend what the
      # others and the log writer still need; the tail reserve is what keeps
      # that from being zero after the children.
      drainer.join(timeout=min(_DRAIN_STOP_TIMEOUT_S, deadline.remaining()))
      if drainer.is_alive():
        logger.warning(
          f"Queue drain '{drainer.name}' is still blocked on a message a "
          f"killed child never finished writing; leaving that queue open."
        )
        self._undrainable_queues.append(q)

  def join_logging(self) -> None:
    """Wait for the log writer, but never on something that cannot finish.

    The writer reads the shared logging queue, and a child killed mid-``put``
    leaves a message whose remainder never arrives: the read then blocks for
    good, with no exception to catch, exactly as on the cancel-path drain
    (issue #77). It is a daemon thread, so leaving it parked costs the tail of
    the session log and nothing else -- whereas waiting on it costs the whole
    teardown, so the wait is the shorter of the budget's remainder and
    ``_LOG_WRITER_STOP_TIMEOUT_S``.
    """
    assert self._logging_thread is not None
    if self._teardown is None:
      self._teardown = Deadline(TEARDOWN_BUDGET_S)
    self._logging_thread.join(
      timeout=min(self._teardown.remaining(), _LOG_WRITER_STOP_TIMEOUT_S)
    )
    if self._logging_thread.is_alive():
      self._logger.warning(
        "The log writer is still blocked reading the logging queue; the "
        "session log may be missing its last lines."
      )
      # Its queue must stay open for the same reason a wedged drainer's does;
      # see _collect_wedged_drainers.
      self._undrainable_queues.append(self._res.logging_resources.logging_queue)
    self._logging_thread = None

  def close_queues(self) -> None:
    """Release the parent's handles on the multiprocessing queues.

    Call once during teardown, after all child processes and the logging thread
    have been joined. Closing is non-blocking (``cancel_join_thread`` first, so
    ``close`` never waits on a feeder) and lets the OS reclaim the pipes and
    semaphores promptly instead of leaving it to garbage collection -- which the
    caller may skip entirely (e.g. ``os._exit``).

    A queue whose drainer or reader is still wedged is skipped; see
    ``_collect_wedged_drainers`` for why closing it would be worse than leaking
    it.
    """
    self.reap_stopped_readers()
    res = self._res
    stats = res.stats_resources
    # Closing needs the write side only, so the parent-to-child input queue
    # and the logging queue fit the same type here.
    queues: list[ChildToParentQueue | None] = [
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
      if q is None:
        continue
      # Always cancelled, even on a queue left open below: it only drops the
      # writer-side join, and leaving that registered would make the interpreter
      # join this queue's feeder thread at exit -- which blocks if the feeder is
      # itself stuck on a pipe nobody drains.
      q.cancel_join_thread()
      if any(q is wedged for wedged in self._undrainable_queues):
        continue
      q.close()
