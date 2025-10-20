from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from dataclasses import asdict

from birdnet.acoustic_models.inference_pipeline.configs import (
  ConfigType,
  PredictionConfig,
  ResultType,
  TensorType,
)
from birdnet.acoustic_models.inference_pipeline.processes import ProcessManager
from birdnet.acoustic_models.inference_pipeline.resources import (
  PipelineResources,
  ResourceManager,
)
from birdnet.acoustic_models.inference_pipeline.strategy import (
  PredictionStrategy,
)
from birdnet.globals import WRITABLE_FLAG
from birdnet.helper import create_shm_ring


def predict_from_recordings_generic(
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
      resources.processing_state.processing_finished_event.set()
      resources.stats_resources.mark_stop()
      resources.stats_resources.collect_performance_results()
      resources.analyzer_resources.collect_file_durations()
      process_manager.join_main_processes()

    if resources.processing_state.cancel_event.is_set():
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
