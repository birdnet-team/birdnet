from __future__ import annotations

import shutil
from abc import ABC
from collections.abc import Callable, Collection, Iterable
from pathlib import Path
from typing import Any, ContextManager, Generic, Literal, Self, cast

import numpy as np
import numpy.typing as npt
from ordered_set import OrderedSet

from birdnet.acoustic.inference.benchmarking import handle_statistics
from birdnet.acoustic.inference.configs import (
  ConfigType,
  EncodingConfig,
  FilteringConfig,
  InferenceConfig,
  ModelConfig,
  OutputConfig,
  PredictionConfig,
  ProcessingConfig,
  ResultType,
  TensorType,
)
from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticDataEncodingResult,
  AcousticFileEncodingResult,
)
from birdnet.acoustic.inference.core.logs import get_logger_from_session
from birdnet.acoustic.inference.core.perf_tracker import AcousticProgressStats
from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticDataPredictionResult,
  AcousticFilePredictionResult,
)
from birdnet.acoustic.inference.encoding_strategy import (
  EncodingStrategy,
)
from birdnet.acoustic.inference.prediction_strategy import (
  PredictionStrategy,
)
from birdnet.acoustic.inference.processes import ProcessManager
from birdnet.acoustic.inference.resources import (
  PipelineResources,
  ResourceManager,
)
from birdnet.acoustic.inference.strategy import InferenceStrategyBase
from birdnet.core.backends import VersionedAcousticBackendProtocol
from birdnet.core.base import SessionBase
from birdnet.globals import ACOUSTIC_MODEL_VERSIONS


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

    self._process_manager.start()

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

    self._resources.stats_resources.save_end_time()

    # Finish processing and wait for processes to finish ("join") their main loops
    self._resources.processing_resources.processing_finished_event.set()
    self._process_manager.join_processing()

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
    self._process_manager.join()

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


class AcousticEncodingSession(AcousticSessionBase):
  def __init__(
    self,
    species_list: OrderedSet[str],
    model_path: Path,
    model_segment_size_s: float,
    model_sample_rate: int,
    model_is_custom: bool,
    model_sig_fmin: int,
    model_sig_fmax: int,
    model_version: ACOUSTIC_MODEL_VERSIONS,
    model_backend_type: type[VersionedAcousticBackendProtocol],
    model_backend_custom_kwargs: dict[str, Any],
    model_emb_dim: int,
    *,
    n_producers: int,
    n_workers: int | None,
    batch_size: int,
    prefetch_ratio: int,
    overlap_duration_s: float,
    speed: float,
    bandpass_fmin: int,
    bandpass_fmax: int,
    half_precision: bool,
    max_audio_duration_min: float | None,
    show_stats: None | Literal["minimal", "progress", "benchmark"],
    progress_callback: Callable[[AcousticProgressStats], None] | None,
    device: str | list[str],
    max_n_files: int,  # Limit to avoid excessive memory usage
  ) -> None:
    assert len(species_list) > 0
    assert model_path.exists()
    assert model_segment_size_s > 0
    assert model_sample_rate > 0
    assert 0 <= model_sig_fmin < model_sig_fmax
    assert model_backend_custom_kwargs is not None
    assert model_emb_dim > 0

    ModelConfig.validate_backend_supports_embeddings(model_backend_type)
    n_producers = ProcessingConfig.validate_n_producers(n_producers)
    n_workers = ProcessingConfig.validate_n_workers(n_workers)
    batch_size = ProcessingConfig.validate_batch_size(batch_size)
    prefetch_ratio = ProcessingConfig.validate_prefetch_ratio(prefetch_ratio)
    overlap_duration_s = ProcessingConfig.validate_overlap_duration(
      overlap_duration_s, model_segment_size_s
    )

    bandpass_fmin, bandpass_fmax = FilteringConfig.validate_bandpass_frequencies(
      bandpass_fmin, bandpass_fmax, model_sig_fmin, model_sig_fmax
    )

    half_precision = ProcessingConfig.validate_half_precision(half_precision)

    if max_audio_duration_min is not None:
      max_audio_duration_min = ProcessingConfig.validate_max_audio_duration_min(
        max_audio_duration_min
      )

    if show_stats is not None:
      show_stats = OutputConfig.validate_show_stats(show_stats)

    max_n_files = ProcessingConfig.validate_max_n_files(max_n_files)

    super().__init__(
      conf=InferenceConfig(
        model_conf=ModelConfig(
          species_list=species_list,
          path=model_path,
          is_custom=model_is_custom,
          version=model_version,
          segment_size_s=model_segment_size_s,
          sample_rate=model_sample_rate,
          sig_fmin=model_sig_fmin,
          sig_fmax=model_sig_fmax,
          backend_type=model_backend_type,
          backend_kwargs=model_backend_custom_kwargs,
        ),
        processing_conf=ProcessingConfig(
          producers=n_producers,
          workers=n_workers,
          batch_size=batch_size,
          prefetch_ratio=prefetch_ratio,
          overlap_duration_s=overlap_duration_s,
          half_precision=half_precision,
          max_audio_duration_min=max_audio_duration_min,
          device=device,
          max_n_files=max_n_files,
          speed=speed,
        ),
        filtering_conf=FilteringConfig(
          bandpass_fmin=bandpass_fmin,
          bandpass_fmax=bandpass_fmax,
        ),
        output_conf=OutputConfig(
          show_stats=show_stats,
          progress_callback=progress_callback,
        ),
      ),
      strategy=EncodingStrategy(),
      specific_config=EncodingConfig(
        emb_dim=model_emb_dim,
      ),
    )

  def run(
    self, inputs: Path | str | Iterable[Path | str]
  ) -> AcousticFileEncodingResult:
    inputs = InferenceConfig.validate_input_files(inputs)

    if len(inputs) > self._conf.processing_conf.max_n_files:
      raise RuntimeError(
        f"Number of input files ({len(inputs)}) exceeds the maximum "
        f"allowed ({self._conf.processing_conf.max_n_files})."
      )

    return super()._run(inputs)

  def run_arrays(
    self, inputs: tuple[npt.NDArray, int] | Iterable[tuple[npt.NDArray, int]]
  ) -> AcousticDataEncodingResult:
    data = InferenceConfig.validate_input_audio(inputs)
    return super()._run(data)


class AcousticPredictionSession(AcousticSessionBase):
  def __init__(
    self,
    species_list: OrderedSet[str],
    model_path: Path,
    model_segment_size_s: float,
    model_sample_rate: int,
    model_is_custom: bool,
    model_sig_fmin: int,
    model_sig_fmax: int,
    model_version: ACOUSTIC_MODEL_VERSIONS,
    model_backend_type: type[VersionedAcousticBackendProtocol],
    model_backend_custom_kwargs: dict[str, Any],
    *,
    top_k: int | None,
    n_producers: int,
    n_workers: int | None,
    batch_size: int = 1,
    prefetch_ratio: int = 1,
    overlap_duration_s: float,
    speed: float,
    bandpass_fmin: int,
    bandpass_fmax: int,
    apply_sigmoid: bool,
    sigmoid_sensitivity: float | None,
    default_confidence_threshold: float | None,
    custom_confidence_thresholds: dict[str, float] | None,
    custom_species_list: str | Path | Collection[str] | None,
    half_precision: bool = True,
    max_audio_duration_min: float | None,
    show_stats: Literal["minimal", "progress", "benchmark"] | None,
    progress_callback: Callable[[AcousticProgressStats], None] | None,
    device: str | list[str],
    max_n_files: int,
  ) -> None:
    assert len(species_list) > 0
    assert model_path.exists()
    assert model_segment_size_s > 0
    assert model_sample_rate > 0
    assert 0 <= model_sig_fmin < model_sig_fmax
    assert model_backend_custom_kwargs is not None

    if top_k is not None:
      top_k = PredictionConfig.validate_top_k(top_k, len(species_list))
    n_producers = ProcessingConfig.validate_n_producers(n_producers)
    n_workers = ProcessingConfig.validate_n_workers(n_workers)
    batch_size = ProcessingConfig.validate_batch_size(batch_size)
    prefetch_ratio = ProcessingConfig.validate_prefetch_ratio(prefetch_ratio)
    overlap_duration_s = ProcessingConfig.validate_overlap_duration(
      overlap_duration_s, model_segment_size_s
    )

    speed = ProcessingConfig.validate_speed(speed)

    bandpass_fmin, bandpass_fmax = FilteringConfig.validate_bandpass_frequencies(
      bandpass_fmin, bandpass_fmax, model_sig_fmin, model_sig_fmax
    )

    half_precision = ProcessingConfig.validate_half_precision(half_precision)

    if max_audio_duration_min is not None:
      max_audio_duration_min = ProcessingConfig.validate_max_audio_duration_min(
        max_audio_duration_min
      )

    if show_stats is not None:
      show_stats = OutputConfig.validate_show_stats(show_stats)

    if progress_callback is not None and show_stats not in ("progress", "benchmark"):
      raise ValueError(
        "Progress callback can only be used when 'show_stats' is set to "
        "'progress' or 'benchmark'."
      )

    if custom_confidence_thresholds is not None:
      custom_confidence_thresholds = (
        PredictionConfig.validate_custom_confidence_thresholds(
          custom_confidence_thresholds, species_list
        )
      )

    if custom_species_list is not None:
      custom_species_list = PredictionConfig.validate_custom_species_list(
        custom_species_list, species_list
      )

    if apply_sigmoid:
      sigmoid_sensitivity = PredictionConfig.validate_sigmoid_sensitivity(
        sigmoid_sensitivity
      )

    max_n_files = ProcessingConfig.validate_max_n_files(max_n_files)

    super().__init__(
      conf=InferenceConfig(
        model_conf=ModelConfig(
          species_list=species_list,
          path=model_path,
          is_custom=model_is_custom,
          version=model_version,
          segment_size_s=model_segment_size_s,
          sample_rate=model_sample_rate,
          sig_fmin=model_sig_fmin,
          sig_fmax=model_sig_fmax,
          backend_type=model_backend_type,
          backend_kwargs=model_backend_custom_kwargs,
        ),
        processing_conf=ProcessingConfig(
          producers=n_producers,
          workers=n_workers,
          batch_size=batch_size,
          prefetch_ratio=prefetch_ratio,
          overlap_duration_s=overlap_duration_s,
          half_precision=half_precision,
          max_audio_duration_min=max_audio_duration_min,
          device=device,
          max_n_files=max_n_files,
          speed=speed,
        ),
        filtering_conf=FilteringConfig(
          bandpass_fmin=bandpass_fmin,
          bandpass_fmax=bandpass_fmax,
        ),
        output_conf=OutputConfig(
          show_stats=show_stats,
          progress_callback=progress_callback,
        ),
      ),
      strategy=PredictionStrategy(),
      specific_config=PredictionConfig(
        top_k=top_k,
        default_confidence_threshold=default_confidence_threshold,
        custom_confidence_thresholds=custom_confidence_thresholds,
        apply_sigmoid=apply_sigmoid,
        sigmoid_sensitivity=sigmoid_sensitivity,
        custom_species_list=custom_species_list,
      ),
    )

  def run(
    self, inputs: Path | str | Iterable[Path | str]
  ) -> AcousticFilePredictionResult:
    inputs = InferenceConfig.validate_input_files(inputs)

    if len(inputs) > self._conf.processing_conf.max_n_files:
      raise RuntimeError(
        f"Number of input files ({len(inputs)}) exceeds the maximum "
        f"allowed ({self._conf.processing_conf.max_n_files})."
      )

    return super()._run(inputs)

  def run_arrays(
    self, inputs: tuple[npt.NDArray, int] | Iterable[tuple[npt.NDArray, int]]
  ) -> AcousticDataPredictionResult:
    data = InferenceConfig.validate_input_audio(inputs)
    return super()._run(data)
