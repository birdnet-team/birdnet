from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal, final

from ordered_set import OrderedSet

from birdnet.acoustic_models.base import (
  AcousticModelBase,
)
from birdnet.acoustic_models.inference.emb.prediction_result import (
  EmbeddingsPredictionResult,
)
from birdnet.acoustic_models.inference.pipelines import (
  predict_embeddings_from_recordings,
  predict_species_from_recordings,
)
from birdnet.acoustic_models.inference.scores.prediction_result import PredictionResult
from birdnet.globals import (
  ACOUSTIC_MODEL_VERSION_V2_4,
  ACOUSTIC_MODEL_VERSIONS,
  MODEL_PRECISIONS,
  MODEL_TYPE_ACOUSTIC,
  MODEL_TYPES,
)


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


class AcousticModelBaseV2_4(AcousticModelBase):
  def __init__(
    self,
    model_path: Path,
    species_list: OrderedSet[str],
    precision: MODEL_PRECISIONS,
    use_custom_model: bool,
  ) -> None:
    super().__init__(model_path, species_list, precision, use_custom_model)

  @classmethod
  @final
  def get_version(cls) -> ACOUSTIC_MODEL_VERSIONS:
    return ACOUSTIC_MODEL_VERSION_V2_4

  @classmethod
  @final
  def get_model_type(cls) -> MODEL_TYPES:
    return MODEL_TYPE_ACOUSTIC

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

  def _predict_embeddings(
    self,
    inp: Path | str | Iterable[Path | str],
    backend_kwargs: dict,
    /,
    *,
    feeders: int = 1,
    workers: int = 4,
    batch_size: int = 1,
    prefetch_ratio: int = 1,
    overlap_duration_s: float = 0,
    use_bandpass: bool = False,
    bandpass_fmin: int | None = None,
    bandpass_fmax: int | None = None,
    half_precision: bool = True,
    max_audio_duration_min: float | None = None,
    show_stats: Literal["no", "minimal", "progress", "benchmark"] = "no",
    device: str | list[str] = "CPU",
  ) -> EmbeddingsPredictionResult:
    return predict_embeddings_from_recordings(
      inp=inp,
      model_backend_kwargs=backend_kwargs,
      model_species_list=self.species_list,
      model_backend=self.get_backend(),
      model_version=self.get_version(),
      model_segment_size_s=self.get_segment_size_s(),
      model_sample_rate=self.get_sample_rate(),
      model_sig_fmin=self.get_sig_fmin(),
      model_sig_fmax=self.get_sig_fmax(),
      model_path=self.model_path,
      model_precision=self.precision,
      model_is_custom=self.use_custom_model,
      model_emb_dim=self.get_embeddings_dim(),
      feeders=feeders,
      workers=workers,
      batch_size=batch_size,
      prefetch_ratio=prefetch_ratio,
      overlap_duration_s=overlap_duration_s,
      use_bandpass=use_bandpass,
      bandpass_fmin=bandpass_fmin,
      bandpass_fmax=bandpass_fmax,
      half_precision=half_precision,
      max_audio_duration_min=max_audio_duration_min,
      show_stats=show_stats,
      device=device,
    )

  def _predict(
    self,
    inp: Path | str | Iterable[Path | str],
    backend_kwargs: dict,
    /,
    *,
    top_k: int | None = 5,
    feeders: int = 1,
    workers: int = 4,
    batch_size: int = 1,
    prefetch_ratio: int = 1,
    overlap_duration_s: float = 0,
    default_confidence_threshold: float | None = 0.1,
    custom_confidence_thresholds: dict[str, float] | None = None,
    use_bandpass: bool = False,
    bandpass_fmin: int | None = None,
    bandpass_fmax: int | None = None,
    apply_sigmoid: bool = True,
    sigmoid_sensitivity: float | None = 1.0,
    custom_species_list: set[str] | None = None,
    half_precision: bool = True,
    max_audio_duration_min: float | None = None,
    show_stats: Literal["no", "minimal", "progress", "benchmark"] = "no",
    device: str | list[str] = "CPU",
  ) -> PredictionResult:
    return predict_species_from_recordings(
      inp=inp,
      model_backend_kwargs=backend_kwargs,
      model_species_list=self.species_list,
      model_backend=self.get_backend(),
      model_version=self.get_version(),
      model_segment_size_s=self.get_segment_size_s(),
      model_sample_rate=self.get_sample_rate(),
      model_sig_fmin=self.get_sig_fmin(),
      model_sig_fmax=self.get_sig_fmax(),
      model_path=self.model_path,
      model_precision=self.precision,
      model_is_custom=self.use_custom_model,
      top_k=top_k,
      feeders=feeders,
      workers=workers,
      batch_size=batch_size,
      prefetch_ratio=prefetch_ratio,
      overlap_duration_s=overlap_duration_s,
      default_confidence_threshold=default_confidence_threshold,
      custom_confidence_thresholds=custom_confidence_thresholds,
      use_bandpass=use_bandpass,
      bandpass_fmin=bandpass_fmin,
      bandpass_fmax=bandpass_fmax,
      apply_sigmoid=apply_sigmoid,
      sigmoid_sensitivity=sigmoid_sensitivity,
      custom_species_list=custom_species_list,
      half_precision=half_precision,
      max_audio_duration_min=max_audio_duration_min,
      show_stats=show_stats,
      device=device,
    )
