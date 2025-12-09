from __future__ import annotations

import json
import shutil
from abc import ABC
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import ContextManager, Generic, Self, cast

import numpy as np

from birdnet.acoustic_models.inference_pipeline.configs import (
  ConfigType,
  InferenceConfig,
  ResultType,
  TensorType,
)
from birdnet.acoustic_models.inference_pipeline.logs import get_logger_from_session
from birdnet.acoustic_models.inference_pipeline.processes import ProcessManager
from birdnet.acoustic_models.inference_pipeline.resources import (
  PipelineResources,
  ResourceManager,
)
from birdnet.acoustic_models.inference_pipeline.strategy import InferenceStrategyBase
from birdnet.base import SessionBase, get_session_id_hash
from birdnet.globals import WRITABLE_FLAG
from birdnet.shm import create_shm_ring


class AcousticSessionBase(
  Generic[ResultType, ConfigType, TensorType], SessionBase, ABC
):
  def __init__(
    self,
    conf: InferenceConfig,
    strategy: InferenceStrategyBase[ResultType, ConfigType, TensorType],
    specific_config: ConfigType,
  ) -> None:
    self._conf = conf
    self._strategy = strategy
    self._specific_config = specific_config
    self._resource_manager: ResourceManager | None = None
    self._process_manager: ProcessManager | None = None
    self._shm_context: ContextManager | None = None
    self._is_initialized = False
    super().__init__()

  def __enter__(self) -> Self:
    assert not self._is_initialized
    self._resource_manager = ResourceManager(self._conf)
    res = self._resource_manager.create_resources(
      self._session_id, self._strategy.get_benchmark_dir_name()
    )

    self._process_manager = ProcessManager(
      self._session_id, self._conf, self._strategy, self._specific_config, res
    )
    self._process_manager.start_logging_thread()

    self._shm_context = shared_memory_context(self._session_id, res)
    self._shm_context.__enter__()

    self._process_manager.start_main_processes()

    self._is_initialized = True
    self._logger = get_logger_from_session(self._session_id, __name__)
    return self

  @property
  def _resources(self) -> PipelineResources:
    assert self._is_initialized
    assert self._resource_manager is not None
    assert self._resource_manager.resources is not None
    return self._resource_manager.resources

  def _run(self, inputs: list[Path] | list[tuple[np.ndarray, int]]) -> ResultType:
    assert self._is_initialized
    assert self._process_manager is not None
    assert self._logger is not None

    if not self._resources.processing_resources.is_first_run:
      self._resources.reset()

    self._logger.info(f"Got {len(inputs)} inputs for analysis.")
    self._process_manager.start_processing(inputs)

    result_tensor = self._strategy.create_tensor(
      self._session_id,
      self._conf,
      self._specific_config,
      self._resources,
      len(inputs),
    )

    self._process_manager.run_consumer(result_tensor)

    if self._resources.processing_resources.cancel_event.is_set():
      raise RuntimeError(
        f"Analysis was cancelled. "
        f"Please check the logs: "
        f"{self._resources.logging_resources.session_log_file.absolute()}"
      )

    self._resources.processing_resources.processing_finished_event.set()
    self._resources.stats_resources.save_end_time()

    # Collect only if no cancellation occurred, otherwise result queues might be empty
    self._resources.analyzer_resources.collect_input_durations()
    self._resources.producer_resources.collect_unprocessed_inputs()
    self._resources.stats_resources.collect_performance_results()

    result_tensor.set_unprocessable_inputs(
      self._resources.producer_resources.unprocessed_inputs
    )

    if is_file_input := any(isinstance(inp, Path) for inp in inputs):
      assert all(isinstance(inp, Path) for inp in inputs)
      inputs = cast(list[Path], inputs)
      result = self._strategy.create_files_result(
        result_tensor, self._conf, self._resources, inputs
      )
    else:
      result = self._strategy.create_array_result(
        result_tensor, self._conf, self._resources
      )

    _handle_statistics(
      self._session_id,
      self._conf,
      self._strategy,
      self._specific_config,
      result,
      self._resources,
    )

    self._resources.processing_resources.increment_run_nr()

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

  def __exit__(self, *args) -> None:
    assert self._is_initialized

    assert self._resources is not None
    assert self._process_manager is not None
    assert self._shm_context is not None

    self.end()
    self._process_manager.join_main_processes()

    self._shm_context.__exit__(*args)
    self._shm_context = None

    self._resources.logging_resources.stop_logging_event.set()

    self._process_manager.join_logging()
    self._process_manager = None

    shutil.copyfile(
      self._resources.logging_resources.session_log_file,
      self._resources.logging_resources.global_log_file,
    )

    self._resource_manager = None
    self._is_initialized = False
    self._logger = None


@contextmanager
def shared_memory_context(session_id: str, resources: PipelineResources):
  with (
    create_shm_ring(session_id, resources.ring_buffer_resources.rf_file_indices),
    create_shm_ring(session_id, resources.ring_buffer_resources.rf_segment_indices),
    create_shm_ring(session_id, resources.ring_buffer_resources.rf_audio_samples),
    create_shm_ring(session_id, resources.ring_buffer_resources.rf_batch_sizes),
    create_shm_ring(
      session_id, resources.ring_buffer_resources.rf_flags
    ) as shm_ring_flags,
  ):
    flags = resources.ring_buffer_resources.rf_flags.get_array(shm_ring_flags)
    flags[:] = WRITABLE_FLAG
    yield


def _handle_statistics(
  session_id: str,
  config: InferenceConfig,
  strategy: InferenceStrategyBase[ResultType, ConfigType, TensorType],
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
      session_id,
      config,
      strategy,
      resources,
      specific_config,
      result,
    )


def _show_minimal_statistics(
  config: InferenceConfig,
  strategy: InferenceStrategyBase[ResultType, ConfigType, TensorType],
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
  session_id: str,
  config: InferenceConfig,
  strategy: InferenceStrategyBase[ResultType, ConfigType, TensorType],
  resources: PipelineResources,
  specific_config: ConfigType,
  result: ResultType,
) -> None:
  assert resources.stats_resources.tracking_result is not None
  assert resources.stats_resources.benchmarking is True
  assert resources.stats_resources.benchmark_dir is not None
  assert resources.stats_resources.benchmark_session_dir is not None

  bmm = strategy.create_full_benchmark_meta(config, specific_config, resources, result)

  benchmark_dir = resources.stats_resources.benchmark_dir
  benchmark_session_dir = resources.stats_resources.benchmark_session_dir
  # iso_time = resources.stats_resources.start_iso_time
  run_name = f"run-{resources.processing_resources.current_run_nr}"
  session_id_hash = get_session_id_hash(session_id)
  benchmark_run_dir = benchmark_session_dir / run_name
  benchmark_run_dir.mkdir(parents=True, exist_ok=True)
  prepend = f"{session_id_hash}-{run_name}"

  sessions_meta_df_out = resources.logging_resources.session_log_file.with_stem(
    resources.logging_resources.session_log_file.stem + "-runs"
  ).with_suffix(".csv")
  all_sessions_meta_df_out = benchmark_dir / "runs.csv"
  stats_out_json = benchmark_run_dir / f"{prepend}-stats.json"
  stats_human_readable_out = benchmark_run_dir / f"{prepend}-stats.txt"
  result_npz = benchmark_run_dir / f"{prepend}-result.npz"

  bm = asdict(bmm)
  del_keys = [k for k in bm if k.startswith("_")]
  for k in del_keys:
    del bm[k]
  bm = bmm.to_dict()

  with open(stats_out_json, "w", encoding="utf8") as f:
    json.dump(bm, f, indent=2, ensure_ascii=False)

  import pandas as pd

  meta_df = pd.DataFrame.from_records([bm])

  meta_df.to_csv(
    sessions_meta_df_out,
    mode="a",
    header=not sessions_meta_df_out.exists(),
    index=False,
  )
  meta_df.to_csv(
    all_sessions_meta_df_out,
    mode="a",
    header=not all_sessions_meta_df_out.exists(),
    index=False,
  )

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
  saved_files += strategy.save_results_extra(result, benchmark_run_dir, prepend)

  summary += (
    f"-------------------------------\n"
    f"Benchmark folder:\n"
    f"  {benchmark_run_dir.absolute()}\n"
    f"Statistics results written to:\n"
    f"  {stats_human_readable_out.absolute()}\n"
    f"  {stats_out_json.absolute()}\n"
    f"  {all_sessions_meta_df_out.absolute()}\n"
    f"Prediction results written to:\n"
  )
  for saved_file in saved_files:
    summary += f"  {saved_file.absolute()}\n"
  summary += (
    f"Session log file:\n  {resources.logging_resources.session_log_file.absolute()}\n"
  )

  print(summary)
