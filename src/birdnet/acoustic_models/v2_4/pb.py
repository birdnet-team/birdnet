from __future__ import annotations

import shutil
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, final

from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.backends import (
  InferenceBackend,
  PBInferenceBackend,
  check_pb_model_can_be_loaded,
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
  MODEL_BACKEND_PB,
  MODEL_BACKENDS,
  MODEL_PRECISION_FP32,
)
from birdnet.helper import check_protobuf_model_files_exist
from birdnet.local_data import get_lang_dir, get_model_path
from birdnet.utils import download_file_tqdm, get_species_from_file


class AcousticPBDownloaderV2_4(AcousticDownloaderBaseV2_4):
  @classmethod
  def _get_paths(cls) -> tuple[Path, Path]:
    model_path = get_model_path(
      AcousticPBModelV2_4.get_model_type(),
      AcousticPBModelV2_4.get_version(),
      AcousticPBModelV2_4.get_backend(),
      MODEL_PRECISION_FP32,
    )
    lang_dir = get_lang_dir(
      AcousticPBModelV2_4.get_model_type(),
      AcousticPBModelV2_4.get_version(),
      AcousticPBModelV2_4.get_backend(),
    )
    return model_path, lang_dir

  @classmethod
  def _check_acoustic_model_available(cls) -> bool:
    model_path, lang_dir = cls._get_paths()

    model_is_downloaded = True
    model_is_downloaded &= model_path.is_dir()
    model_is_downloaded &= check_protobuf_model_files_exist(model_path)

    model_is_downloaded &= lang_dir.is_dir()
    for lang in cls.AVAILABLE_LANGUAGES:
      model_is_downloaded &= (lang_dir / f"{lang}.txt").is_file()

    return model_is_downloaded

  @classmethod
  def _download_acoustic_model(cls) -> None:
    dl_url = "https://zenodo.org/records/15050749/files/BirdNET_v2.4_protobuf.zip"
    dl_size = 124522908

    with tempfile.TemporaryDirectory(prefix="birdnet_download") as temp_dir:
      zip_download_path = Path(temp_dir) / "download.zip"
      download_file_tqdm(
        dl_url,
        zip_download_path,
        download_size=dl_size,
        description="Downloading model",
      )

      print("Extracting models...")
      extract_dir = Path(temp_dir) / "extracted"

      with zipfile.ZipFile(zip_download_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

      acoustic_model_dl_dir = extract_dir / "audio-model"
      species_dl_dir = extract_dir / "labels"

      acoustic_model_dir, acoustic_lang_dir = cls._get_paths()
      acoustic_model_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(acoustic_model_dl_dir, acoustic_model_dir)

      acoustic_lang_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(species_dl_dir, acoustic_lang_dir)
      print("Models extracted.")

  @classmethod
  def get_model_path_and_labels(
    cls,
    lang: str,
  ) -> tuple[Path, OrderedSet[str]]:
    if not cls._check_acoustic_model_available():
      cls._download_acoustic_model()
    assert cls._check_acoustic_model_available()

    model_dir, langs_path = cls._get_paths()

    lang_file = langs_path / f"{lang}.txt"
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return model_dir, labels


class AcousticPBModelV2_4(AcousticModelBaseV2_4):
  def __init__(
    self,
    model_path: Path,
    species_list: OrderedSet[str],
    use_custom_model: bool,
  ) -> None:
    super().__init__(model_path, species_list, MODEL_PRECISION_FP32, use_custom_model)

  @classmethod
  @final
  def get_backend(cls) -> MODEL_BACKENDS:
    return MODEL_BACKEND_PB

  @classmethod
  @final
  def get_backend_type(cls) -> type[InferenceBackend]:
    return PBInferenceBackend

  @classmethod
  def load(cls, lang: str) -> AcousticPBModelV2_4:
    model_path, species_list = AcousticPBDownloaderV2_4.get_model_path_and_labels(lang)
    result = AcousticPBModelV2_4(
      model_path=model_path,
      species_list=species_list,
      use_custom_model=False,
    )
    return result

  @classmethod
  def load_custom(
    cls, model: Path, species_list: Path, check_validity: bool
  ) -> AcousticPBModelV2_4:
    assert model.is_dir()
    assert species_list.is_file()

    loaded_species_list: OrderedSet[str]
    try:
      loaded_species_list = get_species_from_file(species_list, encoding="utf8")
    except Exception as e:
      raise ValueError(
        f"Failed to read species list from '{species_list.absolute()}'. Ensure it is a valid text file."
      ) from e

    if not check_protobuf_model_files_exist(model):
      raise ValueError(
        f"Model directory '{model.absolute()}' does not contain the required files for a Protobuf model!"
      )

    if check_validity:
      n_species_in_model = check_pb_model_can_be_loaded(model, "basic", "scores")
      if n_species_in_model != len(loaded_species_list):
        raise ValueError(
          f"Model '{model.absolute()}' has {n_species_in_model} outputs, but species list '{species_list.absolute()}' has {len(loaded_species_list)} species!"
        )

    result = AcousticPBModelV2_4(
      model_path=model, species_list=loaded_species_list, use_custom_model=True
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
    show_stats: Literal["minimal", "progress", "benchmark"] | None = None,
    device: str | list[str] = "CPU",
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

    device = ProcessingConfig.validate_device(device, workers)

    return predict_embeddings_from_recordings(
      conf=PredictionConfig(
        input_files=input_files,
        model_conf=ModelConfig(
          species_list=self.species_list,
          path=self.model_path,
          backend=self.get_backend(),
          backend_kwargs={
            "signature_name": "embeddings",
            "prediction_key": "embeddings",
            "input_key": "inputs",
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
          device=device,
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
    default_confidence_threshold: float | None = 0.1,
    custom_confidence_thresholds: dict[str, float] | None = None,
    bandpass_fmin: int | None = None,
    bandpass_fmax: int | None = None,
    apply_sigmoid: bool = True,
    sigmoid_sensitivity: float | None = 1.0,
    custom_species_list: set[str] | None = None,
    half_precision: bool = True,
    max_audio_duration_min: float | None = None,
    show_stats: None | Literal["minimal", "progress", "benchmark"] = None,
    device: str | list[str] = "CPU",
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

    device = ProcessingConfig.validate_device(device, workers)

    return predict_species_from_recordings(
      conf=PredictionConfig(
        input_files=input_files,
        model_conf=ModelConfig(
          species_list=self.species_list,
          path=self.model_path,
          backend=self.get_backend(),
          backend_kwargs={
            "signature_name": "basic",
            "prediction_key": "scores",
            "input_key": "inputs",
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
          device=device,
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
