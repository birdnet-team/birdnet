from __future__ import annotations

from pathlib import Path
from typing import final

from ordered_set import OrderedSet

from birdnet.acoustic_models.inference_pipeline.configs import ModelConfig, PredictionConfig
from birdnet.globals import (
  ACOUSTIC_MODEL_VERSION_V2_4,
  ACOUSTIC_MODEL_VERSIONS,
  MODEL_PRECISION_FP16,
  MODEL_PRECISION_FP32,
  MODEL_PRECISION_INT8,
  MODEL_PRECISIONS,
  MODEL_TYPE_ACOUSTIC,
  MODEL_TYPES,
)
from birdnet.helper import ModelInfo
from birdnet_2.acoustic_models.base import (
  AcousticModelBase,
)
from birdnet_2.acoustic_models.inference.backends import InferenceBackendLoader2


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
    precision: MODEL_PRECISIONS,
    use_custom_model: bool,
    backend_loader: InferenceBackendLoader2,
  ) -> None:
    super().__init__(model_path, species_list, precision, use_custom_model)

  def load(self):
    self._backend = self._backend_loader.load_backend()
    conf=PredictionConfig(
        model_conf=ModelConfig(
          species_list=self.species_list,
          path=self.model_path,
          backend=self.get_backend(),
          backend_kwargs={
            "inference_library": self._library,
            "in_idx": MODEL_IN_IDX,
            "out_idx": MODEL_EMB_OUT_IDX,
          },
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
        ),
        filtering_conf=FilteringConfig(
          bandpass_fmin=bandpass_fmin,
          bandpass_fmax=bandpass_fmax,
        ),
        output_conf=OutputConfig(
          show_stats=show_stats,
        ),
      ),
    )

  def predict():
    pass

  def encode():
    pass

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
