from __future__ import annotations

import shutil
from abc import ABC
from contextlib import contextmanager, suppress
from multiprocessing import shared_memory
from pathlib import Path
from typing import ContextManager, Generic, Self, cast

import numpy as np

from birdnet.acoustic.inference.benchmarking import handle_statistics
from birdnet.acoustic.inference.configs import (
  ConfigType,
  InferenceConfig,
  ResultType,
  TensorType,
)
from birdnet.acoustic.inference.core.logs import get_logger_from_session
from birdnet.acoustic.inference.core.shm import RingField, create_shm_ring
from birdnet.acoustic.inference.processes import ProcessManager
from birdnet.acoustic.inference.resources import (
  PipelineResources,
  ResourceManager,
)
from birdnet.acoustic.inference.strategy import InferenceStrategyBase
from birdnet.core.base import SessionBase
from birdnet.globals import WRITABLE_FLAG


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
    self._process_manager.start_file_logging_thread()

    self._shm_context = res.ring_buffer_resources.shared_memory_context(
      self._session_id
    )
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

    self._resources.ring_buffer_resources.set_all_flags_writeable()
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

    handle_statistics(
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

    self._resources.ring_buffer_resources.delete_ring_variables()
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
