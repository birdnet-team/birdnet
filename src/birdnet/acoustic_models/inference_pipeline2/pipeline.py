from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import ContextManager, Generic, Iterable

from ordered_set import OrderedSet

import birdnet.logging_utils as bn_logging
from birdnet.acoustic_models.inference_pipeline2.configs import (
  ConfigType,
  PredictionConfig,
  ResultType,
  TensorType,
)
from birdnet.acoustic_models.inference_pipeline2.processes import ProcessManager
from birdnet.acoustic_models.inference_pipeline2.resources import (
  PipelineResources,
  ResourceManager,
)
from birdnet.acoustic_models.inference_pipeline2.strategy import (
  PredictionStrategy,
)
from birdnet.globals import BATCH_END_SENTINEL, BATCH_START_SENTINEL, WRITABLE_FLAG
from birdnet.helper import create_shm_ring


class PredictionSession(Generic[ResultType, ConfigType, TensorType]):
  def __init__(
    self,
    conf: PredictionConfig,
    strategy: PredictionStrategy[ResultType, ConfigType, TensorType],
    specific_config: ConfigType,
  ) -> None:
    self._conf = conf
    self._strategy = strategy
    self._specific_config = specific_config
    self._resources: PipelineResources | None = None
    self._process_manager: ProcessManager | None = None
    self._shm_context: ContextManager | None = None
    self._is_initialized = False

  def __enter__(self):
    assert not self._is_initialized
    resource_manager = ResourceManager(
      self._conf, self._strategy.get_benchmark_dir_name()
    )
    self._resources = resource_manager.create_all_resources()

    self._process_manager = ProcessManager(
      self._conf, self._strategy, self._specific_config, self._resources
    )
    self._process_manager.start_logging()

    self._shm_context = shared_memory_context(self._resources)
    self._shm_context.__enter__()

    self._process_manager.start_main_processes()

    self._is_initialized = True
    return self

  def run(self, paths: Path | str | Iterable[Path | str]) -> ResultType:
    assert self._is_initialized
    assert self._resources is not None
    assert self._process_manager is not None

    # todo: only once in model class?
    paths = PredictionConfig.validate_input_files(paths)

    if len(paths) > self._conf.processing_conf.max_n_files:
      raise RuntimeError(
        f"Number of input files ({len(paths)}) exceeds the maximum allowed ({self._conf.processing_conf.max_n_files})."
      )

    logger = bn_logging.get_logger(__name__)
    logger.info(f"Got {len(paths)} audio files for analysis.")

    file_paths = OrderedSet(sorted(paths))

    result_tensor = self._strategy.create_tensor(
      self._conf, self._specific_config, self._resources, len(file_paths)
    )

    self._resources.reset()

    # start file analyzer
    self._resources.analyzer_resources.start_signal.set()
    self._resources.analyzer_resources.input_files_queue.put(file_paths)

    # start producers
    for i in range(self._resources.producer_resources.n_producers):
      self._resources.producer_resources.start_signals[i].set()
    for file_idx, file_path in enumerate(file_paths):
      self._resources.producer_resources.files_queue.put(
        (file_idx, file_path), block=False
      )
    for _ in range(self._resources.producer_resources.n_producers):
      self._resources.producer_resources.files_queue.put(None, block=False)

    # start workers
    for i in range(self._conf.processing_conf.workers):
      self._resources.worker_resources.start_signals[i].set()

    # start performance tracker
    if self._resources.stats_resources.track_performance:
      assert self._resources.stats_resources.perf_res_start_signal is not None
      self._resources.stats_resources.perf_res_start_signal.set()

    self._process_manager.run_consumer(result_tensor)
    self._resources.processing_resources.processing_finished_event.set()
    self._resources.stats_resources.save_end_time()
    self._resources.stats_resources.collect_performance_results()
    self._resources.analyzer_resources.collect_file_durations()

    if self._resources.processing_resources.cancel_event.is_set():
      raise RuntimeError(
        f"Analysis was cancelled due to an error. Please check the logs: {self._resources.logging_resources.log_file.absolute()}"
      )

    result = self._strategy.create_result(
      result_tensor, self._conf, self._resources, file_paths
    )

    _handle_statistics(
      self._conf, self._strategy, self._specific_config, result, self._resources
    )

    return result

  def cancel(self) -> None:
    if not self._is_initialized:
      raise RuntimeError("Pipeline is not initialized.")
    assert self._resources is not None

    self._resources.processing_resources.cancel_event.set()

  def end(self) -> None:
    if not self._is_initialized:
      raise RuntimeError("Pipeline is not initialized.")

    assert self._resources is not None
    self._resources.processing_resources.end_event.set()

  def __exit__(self, *args):
    assert self._is_initialized

    assert self._resources is not None
    assert self._process_manager is not None
    assert self._shm_context is not None

    self._resources.processing_resources.end_event.set()
    self._process_manager.join_main_processes()

    self._shm_context.__exit__(*args)
    self._shm_context = None

    self._resources.logging_resources.stop_logging_event.set()

    self._process_manager.join_logging()
    self._process_manager = None

    shutil.copyfile(
      self._resources.logging_resources.log_file,
      self._resources.logging_resources.global_log_file,
    )

    self._resources = None
    self._is_initialized = False


def predict_from_recordings_generic(
  conf: PredictionConfig,
  strategy: PredictionStrategy[ResultType, ConfigType, TensorType],
  specific_config: ConfigType,
):
  resource_manager = ResourceManager(conf, strategy.get_benchmark_dir_name())
  resources = resource_manager.create_all_resources()

  process_manager = ProcessManager(conf, strategy, specific_config, resources)
  process_manager.start_logging()

  try:
    result_tensor = strategy.create_tensor(conf, specific_config, resources)

    with shared_memory_context(resources):
      process_manager.start_main_processes()

      file_paths = OrderedSet(sorted(conf.input_files))

      result = None

      for i in range(3):
        resources.reset()

        # start file analyzer
        resources.analyzer_resources.start_signal.set()
        resources.analyzer_resources.input_files_queue.put(file_paths)

        # start producers
        for i in range(resources.producer_resources.n_producers):
          resources.producer_resources.start_signals[i].set()
        for file_idx, file_path in enumerate(file_paths):
          resources.producer_resources.files_queue.put(
            (file_idx, file_path), block=False
          )
        for _ in range(resources.producer_resources.n_producers):
          resources.producer_resources.files_queue.put(None, block=False)

        # start workers
        for i in range(conf.processing_conf.workers):
          resources.worker_resources.start_signals[i].set()

        # start performance tracker
        if resources.stats_resources.track_performance:
          assert resources.stats_resources.perf_res_start_signal is not None
          resources.stats_resources.perf_res_start_signal.set()

        process_manager.run_consumer(result_tensor)
        resources.processing_resources.processing_finished_event.set()
        resources.stats_resources.save_end_time()
        resources.stats_resources.collect_performance_results()
        resources.analyzer_resources.collect_file_durations()

        if resources.processing_resources.cancel_event.is_set():
          raise RuntimeError(
            f"Analysis was cancelled due to an error. Please check the logs: {resources.logging_resources.log_file.absolute()}"
          )

        result = strategy.create_result(result_tensor, conf, resources)

        _handle_statistics(conf, strategy, specific_config, result, resources)

      # end
      resources.processing_resources.end_event.set()
      process_manager.join_main_processes()

    return result

  finally:
    resources.logging_resources.stop_logging_event.set()
    process_manager.join_logging()

    shutil.copyfile(
      resources.logging_resources.log_file, resources.logging_resources.global_log_file
    )


def predict_from_recordings_generic_legacy(
  conf: PredictionConfig,
  strategy: PredictionStrategy[ResultType, ConfigType, TensorType],
  specific_config: ConfigType,
) -> ResultType:
  resource_manager = ResourceManager(conf, strategy.get_benchmark_dir_name())
  resources = resource_manager.create_all_resources()

  process_manager = ProcessManager(conf, strategy, specific_config, resources)
  process_manager.start_logging()

  try:
    result_tensor = strategy.create_tensor(conf, specific_config, resources)

    with shared_memory_context(resources):
      process_manager.start_main_processes()
      process_manager.run_consumer(result_tensor)
      resources.processing_resources.processing_finished_event.set()
      resources.stats_resources.save_end_time()
      resources.stats_resources.collect_performance_results()
      resources.analyzer_resources.collect_file_durations()
      process_manager.join_main_processes()

    if resources.processing_resources.cancel_event.is_set():
      raise RuntimeError(
        f"Analysis was cancelled due to an error. Please check the logs: {resources.logging_resources.log_file.absolute()}"
      )

    result = strategy.create_result(result_tensor, conf, resources)

    _handle_statistics(conf, strategy, specific_config, result, resources)

    return result

  finally:
    resources.logging_resources.stop_logging_event.set()
    process_manager.join_logging()

    shutil.copyfile(
      resources.logging_resources.log_file, resources.logging_resources.global_log_file
    )


@contextmanager
def shared_memory_context(resources: PipelineResources):
  with (
    create_shm_ring(resources.ring_buffer_resources.rf_file_indices),
    create_shm_ring(resources.ring_buffer_resources.rf_segment_indices),
    create_shm_ring(resources.ring_buffer_resources.rf_audio_samples),
    create_shm_ring(resources.ring_buffer_resources.rf_batch_sizes),
    create_shm_ring(resources.ring_buffer_resources.rf_flags) as shm_ring_flags,
  ):
    flags = resources.ring_buffer_resources.rf_flags.get_array(shm_ring_flags)
    flags[:] = WRITABLE_FLAG
    yield


def _handle_statistics(
  config: PredictionConfig,
  strategy: PredictionStrategy[ResultType, ConfigType, TensorType],
  specific_config: ConfigType,
  result: ResultType,
  resources: PipelineResources,
) -> None:
  if config.output_conf.show_stats in ("minimal", "progress"):
    _show_minimal_statistics(
      config,
      strategy,
      resources,
      specific_config,
      result,
    )
  elif config.output_conf.show_stats == "benchmark":
    _create_benchmark_statistics(
      config,
      strategy,
      resources,
      specific_config,
      result,
    )


def _show_minimal_statistics(
  config: PredictionConfig,
  strategy: PredictionStrategy[ResultType, ConfigType, TensorType],
  resources: PipelineResources,
  specific_config: ConfigType,
  result: ResultType,
) -> None:
  bmm = strategy.create_minimal_benchmark_meta(
    config, specific_config, resources, result
  )

  summary = (
    f"-------------------------------\n"
    f"----------- Summary -----------\n"
    f"-------------------------------\n"
    f"Start time: {bmm.time_begin}\n"
    f"End time:   {bmm.time_end}\n"
    f"Wall time:  {bmm.time_wall_time}\n"
    f"Input: {bmm.file_count} file(s) ({bmm.file_formats})\n"
    f"  Total duration: {bmm.file_duration_sum}\n"
    f"  Average duration: {bmm.file_duration_average}\n"
    f"  Minimum duration (single file): {bmm.file_duration_minimum}\n"
    f"  Maximum duration (single file): {bmm.file_duration_maximum}\n"
    f"Memory usage:\n"
    f"  Buffer: {bmm.mem_shm_size_total_MiB:.2f} M (shared memory)\n"
    f"  Result: {bmm.mem_result_total_memory_usage_MiB:.2f} M (NumPy)\n"
    f"Performance:\n"
    f"  {bmm.speed_total_xrt:.0f} x real-time (RTF: {bmm.speed_total_rtf:.8f})\n"
    f"  {bmm.speed_total_seg_per_second:.0f} segments/s ({bmm.speed_total_audio_per_second} audio/s)\n"
  )
  print(summary)


def _create_benchmark_statistics(
  config: PredictionConfig,
  strategy: PredictionStrategy[ResultType, ConfigType, TensorType],
  resources: PipelineResources,
  specific_config: ConfigType,
  result: ResultType,
) -> None:
  assert resources.stats_resources.tracking_result is not None

  bmm = strategy.create_full_benchmark_meta(config, specific_config, resources, result)

  benchmark_dir = resources.stats_resources.benchmark_dir
  benchmark_run_out_dir = resources.stats_resources.benchmark_run_dir
  iso_time = resources.stats_resources.start_iso_time

  assert benchmark_dir is not None
  assert benchmark_run_out_dir is not None

  meta_df_out = benchmark_dir / "runs.csv"
  stats_out_json = benchmark_run_out_dir / f"stats-{iso_time}.json"
  stats_human_readable_out = benchmark_run_out_dir / f"stats-{iso_time}.txt"
  result_npz = benchmark_run_out_dir / f"result-{iso_time}.npz"

  bm = asdict(bmm)
  del_keys = [k for k in bm if k.startswith("_")]
  for k in del_keys:
    del bm[k]
  bm = bmm.to_dict()

  with open(stats_out_json, "w", encoding="utf8") as f:
    json.dump(bm, f, indent=2, ensure_ascii=False)

  import pandas as pd

  meta_df = pd.DataFrame.from_records([bm])
  meta_df.to_csv(meta_df_out, mode="a", header=not meta_df_out.exists(), index=False)

  summary = (
    f"-------------------------------\n"
    f"------ Benchmark summary ------\n"
    f"-------------------------------\n"
    f"Start time: {bmm.time_begin}\n"
    f"End time:   {bmm.time_end}\n"
    f"Wall time:  {bmm.time_wall_time}\n"
    f"Input: {bmm.file_count} file(s) ({bmm.file_formats})\n"
    f"  Total duration: {bmm.file_duration_sum}\n"
    f"  Average duration: {bmm.file_duration_average}\n"
    f"  Minimum duration (single file): {bmm.file_duration_minimum}\n"
    f"  Maximum duration (single file): {bmm.file_duration_maximum}\n"
    f"Feeder(s): {bmm.param_producers}\n"
    f"Buffer: {bmm.mem_shm_slots_average_filled:.1f}/{config.processing_conf.n_slots} filled slots (mean)\n"
    f"Busy workers: {bmm.worker_busy_average:.1f}/{bmm.param_workers} (mean)\n"
    f"  Average wait time for next batch: {bmm.worker_wait_time_average_milliseconds:.3f} ms\n"
    f"Memory usage:\n"
    f"  Program: {bmm.mem_memory_usage_maximum_MiB:.2f} M (total max)\n"
    f"  Buffer: {bmm.mem_shm_size_total_MiB:.2f} M (shared memory)\n"
    f"  Result: {bmm.mem_result_total_memory_usage_MiB:.2f} M (NumPy)\n"
    f"Performance:\n"
    f"  {bmm.speed_total_xrt:.0f} x real-time (RTF: {bmm.speed_total_rtf:.8f})\n"
    f"  {bmm.speed_total_seg_per_second:.0f} segments/s ({bmm.speed_total_audio_per_second} audio/s)\n"
    f"Worker performance:\n"
    f"  {bmm.speed_worker_xrt:.0f} x real-time (RTF: {bmm.speed_worker_rtf:.8f})\n"
    f"  {bmm.speed_worker_total_seg_per_second:.0f} segments/s ({bmm.speed_worker_total_audio_per_second} audio/s)\n"
  )

  stats_human_readable_out.write_text(summary, encoding="utf8")

  print("Saving result using internal format (.npz)...")
  result.save(result_npz)
  saved_files = [result_npz]
  saved_files += strategy.save_results_extra(result, benchmark_run_out_dir, iso_time)

  summary += (
    f"-------------------------------\n"
    f"Benchmark folder:\n"
    f"  {benchmark_run_out_dir.absolute()}\n"
    f"Statistics results written to:\n"
    f"  {stats_human_readable_out.absolute()}\n"
    f"  {stats_out_json.absolute()}\n"
    f"  {meta_df_out.absolute()}\n"
    f"Prediction results written to:\n"
  )
  for saved_file in saved_files:
    summary += f"  {saved_file.absolute()}\n"
  summary += (
    f"Log file written to:\n  {resources.logging_resources.log_file.absolute()}\n"
  )

  print(summary)
