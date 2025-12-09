from __future__ import annotations

from collections.abc import Callable, Collection, Iterable
from pathlib import Path
from typing import Any, Literal, final

from ordered_set import OrderedSet

from birdnet.acoustic_models.base import AcousticModelBase
from birdnet.acoustic_models.inference.encoding.result import (
  AcousticEncodingResultBase,
)
from birdnet.acoustic_models.inference.perf_tracker import AcousticProgressStats
from birdnet.acoustic_models.inference.prediction.result import (
  AcousticPredictionResultBase,
)
from birdnet.acoustic_models.inference_pipeline.api import (
  AcousticEncodingSession,
  AcousticPredictionSession,
)
from birdnet.acoustic_models.inference_pipeline.configs import InferenceConfig
from birdnet.backends import BackendLoader, VersionedAcousticBackendProtocol
from birdnet.globals import ACOUSTIC_MODEL_VERSION_V2_4, ACOUSTIC_MODEL_VERSIONS
from birdnet.helper import validate_species_list


class AcousticDownloaderBaseV2_4:
  AVAILABLE_LANGUAGES: OrderedSet[str] = OrderedSet(
    (
      "af",
      "ar",
      "cs",
      "da",
      "de",
      "en_uk",
      "en_us",
      "es",
      "fi",
      "fr",
      "hu",
      "it",
      "ja",
      "ko",
      "nl",
      "no",
      "pl",
      "pt",
      "ro",
      "ru",
      "sk",
      "sl",
      "sv",
      "th",
      "tr",
      "uk",
      "zh",
    )
  )


class AcousticModelV2_4(AcousticModelBase):
  def __init__(
    self,
    model_path: Path,
    species_list: OrderedSet[str],
    use_custom_model: bool,
    backend_type: type[VersionedAcousticBackendProtocol],
    backend_kwargs: dict[str, Any],
  ) -> None:
    super().__init__(model_path, species_list, use_custom_model)
    self._backend_type = backend_type
    self._backend_custom_kwargs = backend_kwargs

  @classmethod
  def load(
    cls,
    model_path: Path,
    species_list: OrderedSet[str],
    backend_type: type[VersionedAcousticBackendProtocol],
    backend_kwargs: dict[str, Any],
  ) -> AcousticModelV2_4:
    result = AcousticModelV2_4(
      model_path,
      species_list,
      use_custom_model=False,
      backend_type=backend_type,
      backend_kwargs=backend_kwargs,
    )
    return result

  @classmethod
  def load_custom(
    cls,
    model_path: Path,
    species_list: Path,
    backend_type: type[VersionedAcousticBackendProtocol],
    backend_kwargs: dict[str, Any],
    check_validity: bool,
  ) -> AcousticModelV2_4:
    assert model_path.exists()
    assert species_list.is_file()

    loaded_species_list = validate_species_list(species_list)

    if check_validity:
      n_species_in_model = BackendLoader.check_model_can_be_loaded(
        model_path, backend_type, backend_kwargs
      )

      if n_species_in_model != len(loaded_species_list):
        raise ValueError(
          f"Model '{model_path.absolute()}' has {n_species_in_model} outputs, but "
          f"species list '{species_list.absolute()}' has "
          f"{len(loaded_species_list)} species!"
        )

    result = AcousticModelV2_4(
      model_path,
      loaded_species_list,
      use_custom_model=True,
      backend_type=backend_type,
      backend_kwargs=backend_kwargs,
    )
    return result

  @classmethod
  @final
  def get_version(cls) -> ACOUSTIC_MODEL_VERSIONS:
    return ACOUSTIC_MODEL_VERSION_V2_4

  @classmethod
  @final
  def get_sig_fmin(cls) -> int:
    return 0

  @classmethod
  @final
  def get_sig_fmax(cls) -> int:
    return 15_000

  @classmethod
  @final
  def get_sample_rate(cls) -> int:
    return 48_000

  @classmethod
  @final
  def get_segment_size_s(cls) -> float:
    return 3.0

  @classmethod
  @final
  def get_segment_size_samples(cls) -> int:
    return 144_000  # 3.0 * 48_000

  @classmethod
  @final
  def get_embeddings_dim(cls) -> int:
    return 1024

  def encode_session(
    self,
    /,
    *,
    n_feeders: int = 1,
    n_workers: int | None = None,
    batch_size: int = 1,
    prefetch_ratio: int = 1,
    overlap_duration_s: float = 0,
    speed: float = 1.0,
    bandpass_fmin: int = 0,
    bandpass_fmax: int = 15_000,
    half_precision: bool = False,
    max_audio_duration_min: float | None = None,
    show_stats: None | Literal["minimal", "progress", "benchmark"] = None,
    progress_callback: Callable[[AcousticProgressStats], None] | None = None,
    device: str | list[str] = "CPU",
    max_n_files: int = 65_536,  # Limit to avoid excessive memory usage
  ) -> AcousticEncodingSession:
    return AcousticEncodingSession(
      species_list=self.species_list,
      model_path=self.model_path,
      model_segment_size_s=self.get_segment_size_s(),
      model_sample_rate=self.get_sample_rate(),
      model_is_custom=self.use_custom_model,
      model_sig_fmin=self.get_sig_fmin(),
      model_sig_fmax=self.get_sig_fmax(),
      model_version=self.get_version(),
      model_backend_type=self._backend_type,
      model_backend_custom_kwargs=self._backend_custom_kwargs,
      model_emb_dim=self.get_embeddings_dim(),
      n_feeders=n_feeders,
      n_workers=n_workers,
      batch_size=batch_size,
      prefetch_ratio=prefetch_ratio,
      overlap_duration_s=overlap_duration_s,
      speed=speed,
      bandpass_fmin=bandpass_fmin,
      bandpass_fmax=bandpass_fmax,
      half_precision=half_precision,
      max_audio_duration_min=max_audio_duration_min,
      show_stats=show_stats,
      progress_callback=progress_callback,
      device=device,
      max_n_files=max_n_files,
    )

  def predict_session(
    self,
    /,
    *,
    top_k: int | None = 5,
    n_feeders: int = 1,
    n_workers: int | None = None,
    batch_size: int = 1,
    prefetch_ratio: int = 1,
    overlap_duration_s: float = 0,
    speed: float = 1.0,
    bandpass_fmin: int = 0,
    bandpass_fmax: int = 15_000,
    apply_sigmoid: bool = True,
    sigmoid_sensitivity: float | None = 1.0,
    default_confidence_threshold: float | None = 0.1,
    custom_confidence_thresholds: dict[str, float] | None = None,
    custom_species_list: str | Path | Collection[str] | None = None,
    half_precision: bool = False,
    max_audio_duration_min: float | None = None,
    show_stats: Literal["minimal", "progress", "benchmark"] | None = None,
    progress_callback: Callable[[AcousticProgressStats], None] | None = None,
    device: str | list[str] = "CPU",
    max_n_files: int = 65_536,  # Limit to avoid excessive memory usage
  ) -> AcousticPredictionSession:
    return AcousticPredictionSession(
      species_list=self.species_list,
      model_path=self.model_path,
      model_segment_size_s=self.get_segment_size_s(),
      model_sample_rate=self.get_sample_rate(),
      model_is_custom=self.use_custom_model,
      model_sig_fmin=self.get_sig_fmin(),
      model_sig_fmax=self.get_sig_fmax(),
      model_version=self.get_version(),
      model_backend_type=self._backend_type,
      model_backend_custom_kwargs=self._backend_custom_kwargs,
      top_k=top_k,
      n_feeders=n_feeders,
      n_workers=n_workers,
      batch_size=batch_size,
      prefetch_ratio=prefetch_ratio,
      overlap_duration_s=overlap_duration_s,
      speed=speed,
      bandpass_fmin=bandpass_fmin,
      bandpass_fmax=bandpass_fmax,
      apply_sigmoid=apply_sigmoid,
      sigmoid_sensitivity=sigmoid_sensitivity,
      default_confidence_threshold=default_confidence_threshold,
      custom_confidence_thresholds=custom_confidence_thresholds,
      custom_species_list=custom_species_list,
      half_precision=half_precision,
      max_audio_duration_min=max_audio_duration_min,
      show_stats=show_stats,
      progress_callback=progress_callback,
      device=device,
      max_n_files=max_n_files,
    )

  def encode(
    self,
    inp: Path | str | Iterable[Path | str],
    /,
    *,
    n_feeders: int = 1,
    n_workers: int | None = None,
    batch_size: int = 1,
    prefetch_ratio: int = 1,
    overlap_duration_s: float = 0,
    speed: float = 1.0,
    bandpass_fmin: int = 0,
    bandpass_fmax: int = 15_000,
    half_precision: bool = False,
    max_audio_duration_min: float | None = None,
    show_stats: None | Literal["minimal", "progress", "benchmark"] = None,
    progress_callback: Callable[[AcousticProgressStats], None] | None = None,
    device: str | list[str] = "CPU",
    max_n_files: int = 65_536,  # Limit to avoid excessive memory usage
  ) -> AcousticEncodingResultBase:
    input_files = InferenceConfig.validate_input_files(inp)
    max_n_files = len(input_files)

    with self.encode_session(
      n_feeders=n_feeders,
      n_workers=n_workers,
      batch_size=batch_size,
      prefetch_ratio=prefetch_ratio,
      overlap_duration_s=overlap_duration_s,
      speed=speed,
      bandpass_fmin=bandpass_fmin,
      bandpass_fmax=bandpass_fmax,
      half_precision=half_precision,
      max_audio_duration_min=max_audio_duration_min,
      show_stats=show_stats,
      max_n_files=max_n_files,
      device=device,
      progress_callback=progress_callback,
    ) as session:
      return session.run(input_files)

  def predict(
    self,
    inp: Path | str | Iterable[Path | str],
    /,
    *,
    top_k: int | None = 5,
    n_feeders: int = 1,
    n_workers: int | None = None,
    batch_size: int = 1,
    prefetch_ratio: int = 1,
    overlap_duration_s: float = 0,
    bandpass_fmin: int = 0,
    bandpass_fmax: int = 15_000,
    speed: float = 1.0,
    apply_sigmoid: bool = True,
    sigmoid_sensitivity: float | None = 1.0,
    default_confidence_threshold: float | None = 0.1,
    custom_confidence_thresholds: dict[str, float] | None = None,
    custom_species_list: str | Path | Collection[str] | None = None,
    half_precision: bool = False,
    max_audio_duration_min: float | None = None,
    device: str | list[str] = "CPU",
    show_stats: Literal["minimal", "progress", "benchmark"] | None = None,
    progress_callback: Callable[[AcousticProgressStats], None] | None = None,
  ) -> AcousticPredictionResultBase:
    input_files = InferenceConfig.validate_input_files(inp)
    max_n_files = len(input_files)

    with self.predict_session(
      top_k=top_k,
      n_feeders=n_feeders,
      n_workers=n_workers,
      batch_size=batch_size,
      prefetch_ratio=prefetch_ratio,
      overlap_duration_s=overlap_duration_s,
      speed=speed,
      bandpass_fmin=bandpass_fmin,
      bandpass_fmax=bandpass_fmax,
      apply_sigmoid=apply_sigmoid,
      sigmoid_sensitivity=sigmoid_sensitivity,
      default_confidence_threshold=default_confidence_threshold,
      custom_confidence_thresholds=custom_confidence_thresholds,
      custom_species_list=custom_species_list,
      half_precision=half_precision,
      max_audio_duration_min=max_audio_duration_min,
      show_stats=show_stats,
      progress_callback=progress_callback,
      max_n_files=max_n_files,
      device=device,
    ) as session:
      return session.run(input_files)
