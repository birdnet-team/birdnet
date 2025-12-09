from __future__ import annotations

import os
import threading
from multiprocessing import Process
from pathlib import Path

import numpy as np

import birdnet.acoustic_models.inference_pipeline.logs
from birdnet.acoustic_models.inference.consumer import Consumer
from birdnet.acoustic_models.inference.files_analyzer import FilesAnalyzer
from birdnet.acoustic_models.inference.perf_tracker import (
  PerformanceTracker,
  ProgressDispatcher,
)
from birdnet.acoustic_models.inference.producer import Producer
from birdnet.acoustic_models.inference.tensor import AcousticTensorBase
from birdnet.acoustic_models.inference_pipeline.configs import (
  ConfigType,
  InferenceConfig,
  ResultType,
  TensorType,
)
from birdnet.acoustic_models.inference_pipeline.resources import PipelineResources
from birdnet.acoustic_models.inference_pipeline.strategy import InferenceStrategyBase
from birdnet.base import get_session_id_hash


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
    self._logger = (
      birdnet.acoustic_models.inference_pipeline.logs.get_logger_from_session(
        session_id, __name__
      )
    )
    self._cfg = config
    self._strategy = strategy
    self._specific_cfg = specific_config
    self._res = resources
    self._logging_thread: threading.Thread | None = None
    self._analyzer_thread: threading.Thread | None = None
    self._perf_tracker_process: Process | None = None
    self._producer_processes: list[Process] | None = None
    self._worker_processes: list[Process] | None = None

  def start_logging_thread(self) -> threading.Thread:
    logging_listener = threading.Thread(
      target=birdnet.acoustic_models.inference_pipeline.logs.QueueFileWriter(
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
    assert self._res.stats_resources.callback_fn is not None

    progress_dispatcher = threading.Thread(
      target=ProgressDispatcher(
        session_id=self._session_id,
        callback_fn=self._res.stats_resources.callback_fn,
        check_interval=0.5,
        start_signal=self._res.stats_resources.callback_start_signal,
        end_event=self._res.processing_resources.end_event,
        callback_queue=self._res.stats_resources.callback_queue,
        cancel_event=self._res.processing_resources.cancel_event,
        processing_finished_event=self._res.processing_resources.processing_finished_event,
      ),
      name=f"{self._session_hash}-ProgressDispatcher",
      daemon=True,
    )
    progress_dispatcher.start()
    return progress_dispatcher

  def start_performance_tracker_process(self) -> Process:
    assert self._res.stats_resources.track_performance
    assert self._res.stats_resources.sem_active_workers is not None
    assert self._res.stats_resources.perf_res_queue is not None
    assert self._res.stats_resources.perf_res_start_signal is not None
    assert self._res.stats_resources.wkr_stats_queue is not None
    assert self._res.stats_resources.prd_stats_queue is not None

    perf_tracker_proc = Process(
      target=PerformanceTracker(
        session_id=self._session_id,
        pred_dur_queue=self._res.stats_resources.wkr_stats_queue,
        processing_finished_event=self._res.processing_resources.processing_finished_event,
        update_interval=0.5,
        print_interval=1,
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
        callback_queue=self._res.stats_resources.callback_queue,
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
      target=FilesAnalyzer(
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
      ),
      name=f"{self._session_hash}-FileAnalyzer",
      daemon=True,
    )
    file_analyzer_proc.start()

    assert self._analyzer_thread is None
    self._analyzer_thread = file_analyzer_proc
    return file_analyzer_proc

  def start_producer_processes(self) -> list[Process]:
    use_bandpass = not (
      self._cfg.model_conf.sig_fmin == self._cfg.filtering_conf.bandpass_fmin
      and self._cfg.model_conf.sig_fmax == self._cfg.filtering_conf.bandpass_fmax
    )

    producer_processes = [
      Process(
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
          n_feeders=self._res.producer_resources.n_producers,
          cancel_event=self._res.processing_resources.cancel_event,
          end_event=self._res.processing_resources.end_event,
          start_signal=self._res.producer_resources.start_signals[i],
          unprocessed_inputs_queue=self._res.producer_resources.unprocessed_inputs_queue,
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

  def start_worker_processes(self) -> list[Process]:
    try:
      self._res.worker_resources.backend_loader.load_backend_in_main_process_if_possible(
        self._res.worker_resources.devices, self._cfg.processing_conf.half_precision
      )
    except Exception as exc:
      raise RuntimeError(f"Error during backend initialization: {exc}") from exc

    worker_processes = [
      Process(
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

  def run_consumer(self, result_tensor: AcousticTensorBase) -> None:
    consumer = Consumer(
      session_id=self._session_id,
      n_workers=self._cfg.processing_conf.workers,
      worker_queue=self._res.worker_resources.results_queue,
      tensor=result_tensor,
      cancel_event=self._res.processing_resources.cancel_event,
    )
    consumer()

  def start_main_processes(self) -> None:
    self.start_file_analyzer_thread()
    self.start_producer_processes()
    self.start_worker_processes()

    if self._res.stats_resources.track_performance:
      self.start_performance_tracker_process()

    if self._res.stats_resources.use_callback:
      self.start_progress_dispatcher_thread()

  def join_main_processes(self) -> None:
    logger = birdnet.acoustic_models.inference_pipeline.logs.get_logger_from_session(
      self._session_id, __name__
    )

    logger.debug("Joining file analyzer thread...")
    assert self._analyzer_thread is not None
    self._analyzer_thread.join()
    self._analyzer_thread = None
    logger.debug("File analyzer finished.")

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

  def join_logging(self) -> None:
    assert self._logging_thread is not None
    self._logging_thread.join()
    self._logging_thread = None
