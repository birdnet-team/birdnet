from __future__ import annotations

import json
from dataclasses import asdict

from birdnet.acoustic.inference.configs import (
  ConfigType,
  InferenceConfig,
  ResultType,
  TensorType,
)
from birdnet.acoustic.inference.resources import (
  PipelineResources,
)
from birdnet.acoustic.inference.strategy import InferenceStrategyBase
from birdnet.core.base import get_session_id_hash


def handle_statistics(
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
    f"Producer(s): {bmm.param_producers}\n"
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
