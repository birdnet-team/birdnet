from __future__ import annotations

from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Literal, final

from ordered_set import OrderedSet

from birdnet.acoustic_models.base import (
  AcousticModelBase2,
)
from birdnet.acoustic_models.inference.backends import (
  InferenceBackendLoader2,
  VersionedInferenceBackendProtocol,
)
from birdnet.acoustic_models.inference.emb.encoding_result import EncodingResult
from birdnet.acoustic_models.inference.emb.tensor import EmbeddingsTensor
from birdnet.acoustic_models.inference.scores.prediction_result import PredictionResult
from birdnet.acoustic_models.inference.scores.tensor import ScoresTensor
from birdnet.acoustic_models.inference_pipeline.configs import (
  EmbeddingsConfig,
  FilteringConfig,
  ModelConfig,
  OutputConfig,
  PredictionConfig,
  ProcessingConfig,
  ScoresConfig,
)
from birdnet.acoustic_models.inference_pipeline.emb_strategy import EmbeddingsStrategy
from birdnet.acoustic_models.inference_pipeline.pipeline import PredictionSession
from birdnet.acoustic_models.inference_pipeline.scores_strategy import ScoresStrategy
from birdnet.globals import (
  ACOUSTIC_MODEL_VERSION_V2_4,
  ACOUSTIC_MODEL_VERSIONS,
  MODEL_PRECISIONS,
)
from birdnet.utils import get_species_from_file


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


class AcousticModelV2_4(AcousticModelBase2):
  def __init__(
    self,
    model_path: Path,
    species_list: OrderedSet[str],
    precision: MODEL_PRECISIONS,
    use_custom_model: bool,
    backend_type: type[VersionedInferenceBackendProtocol],
    backend_custom_kwargs: dict[str, object] | None,
  ) -> None:
    super().__init__(model_path, species_list, precision, use_custom_model)
    self._backend_type = backend_type
    self._backend_custom_kwargs = backend_custom_kwargs

  @classmethod
  def load(
    cls,
    model_path: Path,
    species_list: OrderedSet[str],
    precision: MODEL_PRECISIONS,
    backend_type: type[VersionedInferenceBackendProtocol],
    backend_custom_kwargs: dict[str, object] | None,
  ) -> AcousticModelV2_4:
    result = AcousticModelV2_4(
      model_path,
      species_list,
      precision,
      use_custom_model=False,
      backend_type=backend_type,
      backend_custom_kwargs=backend_custom_kwargs,
    )
    return result

  @classmethod
  def load_custom(
    cls,
    model_path: Path,
    species_list: Path,
    precision: MODEL_PRECISIONS,
    backend_type: type[VersionedInferenceBackendProtocol],
    backend_custom_kwargs: dict[str, object] | None,
    check_validity: bool,
  ) -> AcousticModelV2_4:
    assert model_path.exists()
    assert species_list.is_file()

    loaded_species_list: OrderedSet[str]
    try:
      loaded_species_list = get_species_from_file(species_list, encoding="utf8")
    except Exception as e:
      raise ValueError(
        f"Failed to read species list from '{species_list.absolute()}'. Ensure it is a valid text file."
      ) from e

    if check_validity:
      n_species_in_model = backend_type.check_model_can_be_loaded(
        model_path, **backend_custom_kwargs if backend_custom_kwargs is not None else {}
      )
      if n_species_in_model != len(loaded_species_list):
        raise ValueError(
          f"Model '{model_path.absolute()}' has {n_species_in_model} outputs, but species list '{species_list.absolute()}' has {len(loaded_species_list)} species!"
        )

    result = AcousticModelV2_4(
      model_path,
      loaded_species_list,
      precision,
      use_custom_model=True,
      backend_type=backend_type,
      backend_custom_kwargs=backend_custom_kwargs,
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
    feeders: int = 1,
    workers: int = 4,
    batch_size: int = 1,
    prefetch_ratio: int = 1,
    overlap_duration_s: float = 0,
    bandpass_fmin: int = 0,
    bandpass_fmax: int = 15_000,
    half_precision: bool = True,
    max_audio_duration_min: float | None = None,
    show_stats: None | Literal["minimal", "progress", "benchmark"] = None,
    max_n_files: int = 65_536,  # Limit to avoid excessive memory usage
  ) -> PredictionSession[EncodingResult, EmbeddingsConfig, EmbeddingsTensor]:
    feeders = ProcessingConfig.validate_feeders(feeders)
    workers = ProcessingConfig.validate_workers(workers)
    batch_size = ProcessingConfig.validate_batch_size(batch_size)
    prefetch_ratio = ProcessingConfig.validate_prefetch_ratio(prefetch_ratio)
    overlap_duration_s = ProcessingConfig.validate_overlap_duration(
      overlap_duration_s, self.get_segment_size_s()
    )

    bandpass_fmin, bandpass_fmax = FilteringConfig.validate_bandpass_frequencies(
      bandpass_fmin,
      bandpass_fmax,
      self.get_sig_fmin(),
      self.get_sig_fmax(),
    )

    half_precision = ProcessingConfig.validate_half_precision(half_precision)

    if max_audio_duration_min is not None:
      max_audio_duration_min = ProcessingConfig.validate_max_audio_duration_min(
        max_audio_duration_min
      )

    if show_stats is not None:
      show_stats = OutputConfig.validate_show_stats(show_stats)

    backend_loader = InferenceBackendLoader2(
      model_path=self.model_path,
      inference_strategy="embeddings",
      backend_type=self._backend_type,
      backend_custom_kwargs=self._backend_custom_kwargs,
    )

    return PredictionSession(
      conf=PredictionConfig(
        model_conf=ModelConfig(
          species_list=self.species_list,
          path=self.model_path,
          backend_loader=backend_loader,
          is_custom=self.use_custom_model,
          version=self.get_version(),
          precision=self.precision,
          segment_size_s=self.get_segment_size_s(),
          sample_rate=self.get_sample_rate(),
          sig_fmin=self.get_sig_fmin(),
          sig_fmax=self.get_sig_fmax(),
        ),
        processing_conf=ProcessingConfig(
          feeders=feeders,
          workers=workers,
          batch_size=batch_size,
          prefetch_ratio=prefetch_ratio,
          overlap_duration_s=overlap_duration_s,
          half_precision=half_precision,
          max_audio_duration_min=max_audio_duration_min,
          device="CPU",  # Device is always CPU for TF models
          max_n_files=max_n_files,
        ),
        filtering_conf=FilteringConfig(
          bandpass_fmin=bandpass_fmin,
          bandpass_fmax=bandpass_fmax,
        ),
        output_conf=OutputConfig(
          show_stats=show_stats,
        ),
      ),
      strategy=EmbeddingsStrategy(),
      specific_config=EmbeddingsConfig(
        emb_dim=self.get_embeddings_dim(),
      ),
    )

  def predict_session(
    self,
    /,
    *,
    top_k: int | None = 5,
    feeders: int = 1,
    workers: int = 4,
    batch_size: int = 1,
    prefetch_ratio: int = 1,
    overlap_duration_s: float = 0,
    bandpass_fmin: int = 0,
    bandpass_fmax: int = 15_000,
    apply_sigmoid: bool = True,
    sigmoid_sensitivity: float | None = 1.0,
    default_confidence_threshold: float | None = 0.1,
    custom_confidence_thresholds: dict[str, float] | None = None,
    custom_species_list: Collection[str] | None = None,
    half_precision: bool = True,
    max_audio_duration_min: float | None = None,
    show_stats: Literal["minimal", "progress", "benchmark"] | None = None,
    max_n_files: int = 65_536,  # Limit to avoid excessive memory usage
  ) -> PredictionSession[PredictionResult, ScoresConfig, ScoresTensor]:
    if top_k is not None:
      top_k = ScoresConfig.validate_top_k(top_k, len(self.species_list))
    feeders = ProcessingConfig.validate_feeders(feeders)
    workers = ProcessingConfig.validate_workers(workers)
    batch_size = ProcessingConfig.validate_batch_size(batch_size)
    prefetch_ratio = ProcessingConfig.validate_prefetch_ratio(prefetch_ratio)
    overlap_duration_s = ProcessingConfig.validate_overlap_duration(
      overlap_duration_s, self.get_segment_size_s()
    )

    bandpass_fmin, bandpass_fmax = FilteringConfig.validate_bandpass_frequencies(
      bandpass_fmin,
      bandpass_fmax,
      self.get_sig_fmin(),
      self.get_sig_fmax(),
    )

    half_precision = ProcessingConfig.validate_half_precision(half_precision)

    if max_audio_duration_min is not None:
      max_audio_duration_min = ProcessingConfig.validate_max_audio_duration_min(
        max_audio_duration_min
      )

    if show_stats is not None:
      show_stats = OutputConfig.validate_show_stats(show_stats)

    if custom_confidence_thresholds is not None:
      custom_confidence_thresholds = ScoresConfig.validate_custom_confidence_thresholds(
        custom_confidence_thresholds, self.species_list
      )

    if custom_species_list is not None:
      custom_species_list = ScoresConfig.validate_custom_species_list(
        custom_species_list, self.species_list
      )

    if apply_sigmoid:
      sigmoid_sensitivity = ScoresConfig.validate_sigmoid_sensitivity(
        sigmoid_sensitivity
      )

    max_n_files = ProcessingConfig.validate_max_n_files(max_n_files)

    backend_loader = InferenceBackendLoader2(
      model_path=self.model_path,
      inference_strategy="scores",
      backend_type=self._backend_type,
      backend_custom_kwargs=self._backend_custom_kwargs,
    )

    return PredictionSession(
      conf=PredictionConfig(
        model_conf=ModelConfig(
          species_list=self.species_list,
          path=self.model_path,
          backend_loader=backend_loader,
          is_custom=self.use_custom_model,
          version=self.get_version(),
          precision=self.precision,
          segment_size_s=self.get_segment_size_s(),
          sample_rate=self.get_sample_rate(),
          sig_fmin=self.get_sig_fmin(),
          sig_fmax=self.get_sig_fmax(),
        ),
        processing_conf=ProcessingConfig(
          feeders=feeders,
          workers=workers,
          batch_size=batch_size,
          prefetch_ratio=prefetch_ratio,
          overlap_duration_s=overlap_duration_s,
          half_precision=half_precision,
          max_audio_duration_min=max_audio_duration_min,
          device="CPU",  # Device is always CPU for TF models
          max_n_files=max_n_files,
        ),
        filtering_conf=FilteringConfig(
          bandpass_fmin=bandpass_fmin,
          bandpass_fmax=bandpass_fmax,
        ),
        output_conf=OutputConfig(
          show_stats=show_stats,
        ),
      ),
      strategy=ScoresStrategy(),
      specific_config=ScoresConfig(
        top_k=top_k,
        default_confidence_threshold=default_confidence_threshold,
        custom_confidence_thresholds=custom_confidence_thresholds,
        apply_sigmoid=apply_sigmoid,
        sigmoid_sensitivity=sigmoid_sensitivity,
        custom_species_list=custom_species_list,
      ),
    )

  def encode(
    self,
    inp: Path | str | Iterable[Path | str],
    /,
    *,
    feeders: int = 1,
    workers: int = 4,
    batch_size: int = 1,
    prefetch_ratio: int = 1,
    overlap_duration_s: float = 0,
    bandpass_fmin: int = 0,
    bandpass_fmax: int = 15_000,
    half_precision: bool = True,
    max_audio_duration_min: float | None = None,
    show_stats: None | Literal["minimal", "progress", "benchmark"] = None,
    max_n_files: int = 65_536,  # Limit to avoid excessive memory usage
  ) -> EncodingResult:
    input_files = PredictionConfig.validate_input_files(inp)
    max_n_files = len(input_files)

    with self.encode_session(
      feeders=feeders,
      workers=workers,
      batch_size=batch_size,
      prefetch_ratio=prefetch_ratio,
      overlap_duration_s=overlap_duration_s,
      bandpass_fmin=bandpass_fmin,
      bandpass_fmax=bandpass_fmax,
      half_precision=half_precision,
      max_audio_duration_min=max_audio_duration_min,
      show_stats=show_stats,
      max_n_files=max_n_files,
    ) as session:
      return session.run(input_files)

  def predict(
    self,
    inp: Path | str | Iterable[Path | str],
    /,
    *,
    top_k: int | None = 5,
    feeders: int = 1,
    workers: int = 4,
    batch_size: int = 1,
    prefetch_ratio: int = 1,
    overlap_duration_s: float = 0,
    bandpass_fmin: int = 0,
    bandpass_fmax: int = 15_000,
    apply_sigmoid: bool = True,
    sigmoid_sensitivity: float | None = 1.0,
    default_confidence_threshold: float | None = 0.1,
    custom_confidence_thresholds: dict[str, float] | None = None,
    custom_species_list: Collection[str] | None = None,
    half_precision: bool = True,
    max_audio_duration_min: float | None = None,
    show_stats: Literal["minimal", "progress", "benchmark"] | None = None,
  ) -> PredictionResult:
    input_files = PredictionConfig.validate_input_files(inp)
    max_n_files = len(input_files)

    with self.predict_session(
      top_k=top_k,
      feeders=feeders,
      workers=workers,
      batch_size=batch_size,
      prefetch_ratio=prefetch_ratio,
      overlap_duration_s=overlap_duration_s,
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
      max_n_files=max_n_files,
    ) as session:
      return session.run(input_files)
