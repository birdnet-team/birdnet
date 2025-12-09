from __future__ import annotations

from collections.abc import Callable, Collection, Iterable
from pathlib import Path
from typing import Any, Literal

import numpy.typing as npt
from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.encoding.result import (
  AcousticDataEncodingResult,
  AcousticFileEncodingResult,
)
from birdnet.acoustic_models.inference.perf_tracker import AcousticProgressStats
from birdnet.acoustic_models.inference.prediction.result import (
  AcousticDataPredictionResult,
  AcousticFilePredictionResult,
)
from birdnet.acoustic_models.inference_pipeline.configs import (
  EncodingConfig,
  FilteringConfig,
  InferenceConfig,
  ModelConfig,
  OutputConfig,
  PredictionConfig,
  ProcessingConfig,
)
from birdnet.acoustic_models.inference_pipeline.encoding_strategy import (
  EncodingStrategy,
)
from birdnet.acoustic_models.inference_pipeline.prediction_strategy import (
  PredictionStrategy,
)
from birdnet.acoustic_models.inference_pipeline.session import AcousticSessionBase
from birdnet.backends import VersionedAcousticBackendProtocol
from birdnet.globals import ACOUSTIC_MODEL_VERSIONS


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
    n_feeders: int,
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
    n_feeders = ProcessingConfig.validate_n_feeders(n_feeders)
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
          feeders=n_feeders,
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
    n_feeders: int,
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
    n_feeders = ProcessingConfig.validate_n_feeders(n_feeders)
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
          feeders=n_feeders,
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
