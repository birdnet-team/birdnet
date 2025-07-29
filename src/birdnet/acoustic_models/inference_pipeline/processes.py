from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time

import birdnet.logging_utils as bn_logging
from birdnet.acoustic_models.inference.consumer import Consumer
from birdnet.acoustic_models.inference.files_analyzer import FilesAnalyzer
from birdnet.acoustic_models.inference.perf_tracker import (
  PerformanceTracker,
)
from birdnet.acoustic_models.inference.producer import Producer
from birdnet.acoustic_models.inference.tensor import TensorBase
from birdnet.acoustic_models.inference_pipeline.configs import (
  ConfigType,
  PredictionConfig,
  ResultType,
  TensorType,
)
from birdnet.acoustic_models.inference_pipeline.resources import (
  PipelineResources,
)
from birdnet.acoustic_models.inference_pipeline.strategy import (
  PredictionStrategy,
)


class ProcessManager:
  def __init__(
    self,
    config: PredictionConfig,
    strategy: PredictionStrategy[ResultType, ConfigType, TensorType],
    specific_config: ConfigType,
    resources: PipelineResources,
  ) -> None:
    self._cfg = config
    self._strategy = strategy
    self._specific_cfg = specific_config
    self._res = resources
    self._logging_thread: threading.Thread | None = None
    self._analyzer_thread: threading.Thread | None = None
    self._perf_tracker_process: mp.Process | None = None
    self._producer_processes: list[mp.Process] | None = None
    self._worker_processes: list[mp.Process] | None = None

  def start_logging(self) -> threading.Thread:
    logging_listener = threading.Thread(
      target=bn_logging.QueueFileWriter(
        log_queue=self._res.logging_resources.logging_queue,
        logging_level=self._res.logging_resources.logging_level,
        log_file=self._res.logging_resources.log_file,
        cancel_event=self._res.processing_state.cancel_event,
        stop_event=self._res.logging_resources.stop_logging_event,
        processing_finished_event=self._res.processing_state.processing_finished_event,
      ),
      name="QueueFileWriter",
      daemon=True,
    )
    logging_listener.start()
    assert self._logging_thread is None
    self._logging_thread = logging_listener
    return logging_listener

  def start_performance_tracker(self) -> mp.Process:
    assert self._res.stats_resources.track_performance
    assert self._res.stats_resources.sem_active_workers is not None
    assert self._res.stats_resources.perf_res_queue is not None
    assert self._res.stats_resources.wkr_stats_queue is not None
    assert self._res.stats_resources.prd_stats_queue is not None

    perf_tracker_proc = mp.Process(
      target=PerformanceTracker(
        pred_dur_queue=self._res.stats_resources.wkr_stats_queue,
        processing_finished_event=self._res.processing_state.processing_finished_event,
        update_interval=0.5,
        print_interval=1,
        prod_stats_queue=self._res.stats_resources.prd_stats_queue,
        n_workers=self._cfg.processing_conf.workers,
        start=self._res.stats_resources.start,
        sem_filled_slots=self._res.ring_buffer_resources.sem_filled_slots,
        workers_start=time.perf_counter(),
        segment_size_s=self._cfg.model_conf.segment_size_s,
        logging_queue=self._res.logging_resources.logging_queue,
        logging_level=self._res.logging_resources.logging_level,
        perf_res=self._res.stats_resources.perf_res_queue,
        parent_process_id=os.getpid(),
        rf_flags=self._res.ring_buffer_resources.rf_flags,
        tot_n_segments_ptr=self._res.analyzer_resources.tot_n_segments_ptr,
        cancel_event=self._res.processing_state.cancel_event,
        sem_active_workers=self._res.stats_resources.sem_active_workers,
      ),
      name="PerformanceTracker",
      daemon=True,
    )
    perf_tracker_proc.start()

    assert self._perf_tracker_process is None
    self._perf_tracker_process = perf_tracker_proc
    return perf_tracker_proc

  def start_file_analyzer(self) -> threading.Thread:
    file_analyzer_proc = threading.Thread(
      target=FilesAnalyzer(
        files=self._res.analyzer_resources.file_paths,
        logging_level=self._res.logging_resources.logging_level,
        logging_queue=self._res.logging_resources.logging_queue,
        segment_duration_s=self._cfg.model_conf.segment_size_s,
        overlap_duration_s=self._cfg.processing_conf.overlap_duration_s,
        max_segment_idx_ptr=self._res.analyzer_resources.max_segment_idx_ptr,
        rf_segment_indices=self._res.ring_buffer_resources.rf_segment_indices,
        analyzing_result=self._res.analyzer_resources.analyzer_queue,
        tot_n_segments=self._res.analyzer_resources.tot_n_segments_ptr,
        cancel_event=self._res.processing_state.cancel_event,
      ),
      name="FileAnalyzer",
      daemon=True,
    )
    file_analyzer_proc.start()

    assert self._analyzer_thread is None
    self._analyzer_thread = file_analyzer_proc
    return file_analyzer_proc

  def start_producers(self) -> list[mp.Process]:
    use_bandpass = not (
      self._cfg.model_conf.sig_fmin == self._cfg.filtering_conf.bandpass_fmin
      and self._cfg.model_conf.sig_fmax == self._cfg.filtering_conf.bandpass_fmax
    )

    producer_processes = [
      mp.Process(
        target=Producer(
          files_queue=self._res.producer_resources.files_queue,
          batch_size=self._cfg.processing_conf.batch_size,
          prd_all_done_event=self._res.producer_resources.prd_all_done_event,
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
          target_sample_rate=self._cfg.model_conf.sample_rate,
          use_bandpass=use_bandpass,
          bandpass_fmax=self._cfg.filtering_conf.bandpass_fmax,
          bandpass_fmin=self._cfg.filtering_conf.bandpass_fmin,
          fmin=self._cfg.model_conf.sig_fmin,
          fmax=self._cfg.model_conf.sig_fmax,
          max_segment_idx_ptr=self._res.analyzer_resources.max_segment_idx_ptr,
          prod_done_ptr=self._res.producer_resources.n_finished_pointer,
          n_feeders=self._res.producer_resources.n_producers,
          cancel_event=self._res.processing_state.cancel_event,
        ),
        name=f"Producer-{i}",
        daemon=True,
      )
      for i in range(self._res.producer_resources.n_producers)
    ]

    for p in producer_processes:
      p.start()

    assert self._producer_processes is None
    self._producer_processes = producer_processes
    return producer_processes

  def start_workers(self) -> list[mp.Process]:
    try:
      self._res.worker_resources.backend_loader.on_before_worker_initialized()
    except Exception as exc:
      raise RuntimeError(f"Error during backend initialization: {exc}")

    worker_processes = [
      mp.Process(
        target=w,
        name=f"Worker-{i}",
        daemon=True,
      )
      for i, w in enumerate(
        self._strategy.create_workers(self._cfg, self._specific_cfg, self._res)
      )
    ]

    for w in worker_processes:
      w.start()

    assert self._worker_processes is None
    self._worker_processes = worker_processes
    return worker_processes

  def run_consumer(self, result_tensor: TensorBase) -> None:
    consumer = Consumer(
      n_workers=self._cfg.processing_conf.workers,
      worker_queue=self._res.worker_resources.results_queue,
      tensor=result_tensor,
      cancel_event=self._res.processing_state.cancel_event,
    )
    consumer()

  def start_main_processes(self) -> None:
    self.start_file_analyzer()
    self.start_producers()
    self.start_workers()

    if self._res.stats_resources.track_performance:
      self.start_performance_tracker()

  def join_main_processes(self):
    logger = bn_logging.get_logger(__name__)

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

  def join_logging(self):
    assert self._logging_thread is not None
    self._logging_thread.join()
    self._logging_thread = None
