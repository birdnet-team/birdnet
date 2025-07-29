from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Collection, Literal, final

from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.backends import (
  InferenceBackend,
  TFInferenceBackend,
  check_tf_model_can_be_loaded,
)
from birdnet.acoustic_models.inference.emb.encoding_result import (
  EncodingResult,
)
from birdnet.acoustic_models.inference.scores.prediction_result import (
  PredictionResult,
)
from birdnet.acoustic_models.inference_pipeline.configs import (
  EmbeddingsConfig,
  FilteringConfig,
  ModelConfig,
  OutputConfig,
  PredictionConfig,
  ProcessingConfig,
  ScoresConfig,
)
from birdnet.acoustic_models.inference_pipeline.emb_strategy import (
  predict_embeddings_from_recordings,
)
from birdnet.acoustic_models.inference_pipeline.scores_strategy import (
  predict_species_from_recordings,
)
from birdnet.acoustic_models.v2_4.base import (
  AcousticDownloaderBaseV2_4,
  AcousticModelBaseV2_4,
)
from birdnet.globals import (
  LIBRARY_TYPES,
  MODEL_BACKEND_TF,
  MODEL_BACKENDS,
  MODEL_LANGUAGES,
  MODEL_PRECISION_FP16,
  MODEL_PRECISION_FP32,
  MODEL_PRECISION_INT8,
  MODEL_PRECISIONS,
)
from birdnet.helper import ModelInfo
from birdnet.local_data import get_lang_dir, get_model_path
from birdnet.utils import download_file_tqdm, get_species_from_file

if TYPE_CHECKING:
  pass

MODEL_IN_IDX = 0
MODEL_EMB_OUT_IDX = 545
MODEL_LOGITS_OUT_IDX = 546

models = {
  MODEL_PRECISION_INT8: ModelInfo(
    dl_url="https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite_int8.zip",
    dl_file_name="audio-model-int8.tflite",
    dl_size=45948867,
    file_size=41064296,
  ),
  MODEL_PRECISION_FP16: ModelInfo(
    dl_url="https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite_fp16.zip",
    dl_file_name="audio-model-fp16.tflite",
    dl_size=53025528,
    file_size=25932528,
  ),
  MODEL_PRECISION_FP32: ModelInfo(
    dl_url="https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite.zip",
    dl_file_name="audio-model.tflite",
    dl_size=76822925,
    file_size=51726412,
  ),
}


class AcousticTFDownloaderV2_4(AcousticDownloaderBaseV2_4):
  @classmethod
  def _get_paths(cls, precision: MODEL_PRECISIONS) -> tuple[Path, Path]:
    model_path = get_model_path(
      AcousticTFModelV2_4.get_model_type(),
      AcousticTFModelV2_4.get_version(),
      AcousticTFModelV2_4.get_backend(),
      precision,
    )
    lang_dir = get_lang_dir(
      AcousticTFModelV2_4.get_model_type(),
      AcousticTFModelV2_4.get_version(),
      AcousticTFModelV2_4.get_backend(),
    )
    return model_path, lang_dir

  @classmethod
  def _check_acoustic_model_available(cls, precision: MODEL_PRECISIONS) -> bool:
    model_path, lang_dir = cls._get_paths(precision)

    if not model_path.is_file():
      return False

    file_stats = os.stat(model_path)
    is_newest_version = file_stats.st_size == models[precision].file_size
    if not is_newest_version:
      return False

    if not lang_dir.is_dir():
      return False

    return all((lang_dir / f"{lang}.txt").is_file() for lang in cls.AVAILABLE_LANGUAGES)

  @classmethod
  def _download_acoustic_model(cls, precision: MODEL_PRECISIONS) -> None:
    with tempfile.TemporaryDirectory(prefix="birdnet_download") as temp_dir:
      zip_download_path = Path(temp_dir) / "download.zip"
      download_file_tqdm(
        models[precision].dl_url,
        zip_download_path,
        download_size=models[precision].dl_size,
        description="Downloading model",
      )

      extract_dir = Path(temp_dir) / "extracted"

      with zipfile.ZipFile(zip_download_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

      acoustic_model_dl_path = extract_dir / models[precision].dl_file_name
      species_dl_dir = extract_dir / "labels"

      acoustic_model_path, acoustic_lang_dir = cls._get_paths(precision)
      acoustic_model_path.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(acoustic_model_dl_path, acoustic_model_path)

      acoustic_lang_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.rmtree(acoustic_lang_dir, ignore_errors=True)
      shutil.move(species_dl_dir, acoustic_lang_dir)

  @classmethod
  def get_model_path_and_labels(
    cls, lang: str, precision: MODEL_PRECISIONS
  ) -> tuple[Path, OrderedSet[str]]:
    assert lang in cls.AVAILABLE_LANGUAGES
    if not cls._check_acoustic_model_available(precision):
      cls._download_acoustic_model(precision)
    assert cls._check_acoustic_model_available(precision)

    model_path, langs_path = cls._get_paths(precision)

    lang_file = langs_path / f"{lang}.txt"
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return model_path, labels


class AcousticTFModelV2_4(AcousticModelBaseV2_4):
  def __init__(
    self,
    model_path: Path,
    species_list: OrderedSet[str],
    precision: MODEL_PRECISIONS,
    use_custom_model: bool,
    library: LIBRARY_TYPES,
  ) -> None:
    super().__init__(model_path, species_list, precision, use_custom_model)
    self._library = library

  @final
  @classmethod
  def get_backend(cls) -> MODEL_BACKENDS:
    return MODEL_BACKEND_TF

  @final
  @classmethod
  def get_backend_type(cls) -> type[InferenceBackend]:
    return TFInferenceBackend

  @classmethod
  def load(
    cls,
    lang: MODEL_LANGUAGES,
    precision: MODEL_PRECISIONS,
    library: LIBRARY_TYPES,
  ) -> AcousticTFModelV2_4:
    model_path, species_list = AcousticTFDownloaderV2_4.get_model_path_and_labels(
      lang, precision
    )
    result = AcousticTFModelV2_4(
      model_path, species_list, precision, use_custom_model=False, library=library
    )
    return result

  @classmethod
  def load_custom(
    cls,
    model: Path,
    species_list: Path,
    precision: MODEL_PRECISIONS,
    check_validity: bool,
    library: LIBRARY_TYPES,
  ) -> AcousticTFModelV2_4:
    assert model.is_file()
    assert species_list.is_file()

    loaded_species_list: OrderedSet[str]
    try:
      loaded_species_list = get_species_from_file(species_list, encoding="utf8")
    except Exception as e:
      raise ValueError(
        f"Failed to read species list from '{species_list.absolute()}'. Ensure it is a valid text file."
      ) from e

    if check_validity:
      n_species_in_model = check_tf_model_can_be_loaded(
        model, library, out_idx=MODEL_LOGITS_OUT_IDX
      )
      if n_species_in_model != len(loaded_species_list):
        raise ValueError(
          f"Model '{model.absolute()}' has {n_species_in_model} outputs, but species list '{species_list.absolute()}' has {len(loaded_species_list)} species!"
        )

    result = AcousticTFModelV2_4(
      model, loaded_species_list, precision, use_custom_model=True, library=library
    )

    return result

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
  ) -> EncodingResult:
    input_files = PredictionConfig.validate_input_files(inp)

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

    return predict_embeddings_from_recordings(
      conf=PredictionConfig(
        input_files=input_files,
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
      emb_config=EmbeddingsConfig(
        emb_dim=self.get_embeddings_dim(),
      ),
    )

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

    return predict_species_from_recordings(
      conf=PredictionConfig(
        input_files=input_files,
        model_conf=ModelConfig(
          species_list=self.species_list,
          path=self.model_path,
          backend=self.get_backend(),
          backend_kwargs={
            "inference_library": self._library,
            "in_idx": MODEL_IN_IDX,
            "out_idx": MODEL_LOGITS_OUT_IDX,
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
      scores_conf=ScoresConfig(
        top_k=top_k,
        default_confidence_threshold=default_confidence_threshold,
        custom_confidence_thresholds=custom_confidence_thresholds,
        apply_sigmoid=apply_sigmoid,
        sigmoid_sensitivity=sigmoid_sensitivity,
        custom_species_list=custom_species_list,
      ),
    )
