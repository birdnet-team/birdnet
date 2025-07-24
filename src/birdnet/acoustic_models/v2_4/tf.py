from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, final

from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.emb.prediction_result import (
  EmbeddingsPredictionResult,
)
from birdnet.acoustic_models.inference.scores.prediction_result import PredictionResult
from birdnet.acoustic_models.v2_4.base import (
  AcousticDownloaderBaseV2_4,
  AcousticModelBaseV2_4,
)
from birdnet.backends import (
  InferenceBackend,
  TFInferenceBackend,
  check_tf_model_can_be_loaded,
  litert_installed,
  tf_installed,
)
from birdnet.globals import (
  LIBRARY_LITERT,
  LIBRARY_TF,
  LIBRARY_TYPES,
  MODEL_BACKEND_TF,
  MODEL_BACKENDS,
  MODEL_LANGUAGES,
  MODEL_PRECISION_FP16,
  MODEL_PRECISION_FP32,
  MODEL_PRECISION_INT8,
  MODEL_PRECISIONS,
  VALID_LIBRARY_TYPES,
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
  ) -> None:
    super().__init__(model_path, species_list, precision, use_custom_model)

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
    cls, lang: MODEL_LANGUAGES, precision: MODEL_PRECISIONS
  ) -> AcousticTFModelV2_4:
    model_path, species_list = AcousticTFDownloaderV2_4.get_model_path_and_labels(
      lang, precision
    )
    result = AcousticTFModelV2_4(
      model_path, species_list, precision, use_custom_model=False
    )
    return result

  @classmethod
  def load_custom(
    cls,
    model: Path,
    species_list: Path,
    precision: MODEL_PRECISIONS,
    check_validity: bool,
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
        model, out_idx=MODEL_LOGITS_OUT_IDX
      )
      if n_species_in_model != len(loaded_species_list):
        raise ValueError(
          f"Model '{model.absolute()}' has {n_species_in_model} outputs, but species list '{species_list.absolute()}' has {len(loaded_species_list)} species!"
        )

    result = AcousticTFModelV2_4(
      model, loaded_species_list, precision, use_custom_model=True
    )

    return result

  def predict_embeddings(
    self,
    inp: Path | str | Iterable[Path | str],
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
    inference_library: LIBRARY_TYPES = LIBRARY_TF,
  ) -> EmbeddingsPredictionResult:
    if inference_library not in VALID_LIBRARY_TYPES:
      raise ValueError(
        f"Unsupported inference library: {inference_library}. Supported libraries are: {', '.join(VALID_LIBRARY_TYPES)}."
      )
    if inference_library == LIBRARY_TF:
      assert tf_installed()
    elif inference_library == LIBRARY_LITERT:
      if not litert_installed():
        raise ValueError(
          f"Parameter 'inference_library': Library '{LIBRARY_LITERT}' is not available. Install birdnet with [litert] option."
        )
    else:
      raise AssertionError()

    return super()._predict_embeddings(
      inp=inp,
      backend_kwargs={
        "inference_library": inference_library,
        "in_idx": MODEL_IN_IDX,
        "out_idx": MODEL_EMB_OUT_IDX,
      },
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
      device="CPU",
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
    use_bandpass: bool = False,
    bandpass_fmin: int | None = None,
    bandpass_fmax: int | None = None,
    apply_sigmoid: bool = True,
    sigmoid_sensitivity: float | None = 1.0,
    custom_species_list: set[str] | None = None,
    half_precision: bool = True,
    max_audio_duration_min: float | None = None,
    show_stats: Literal["no", "minimal", "progress", "benchmark"] = "no",
    inference_library: LIBRARY_TYPES = LIBRARY_TF,
  ) -> PredictionResult:
    if inference_library not in VALID_LIBRARY_TYPES:
      raise ValueError(
        f"Unsupported inference library: {inference_library}. Supported libraries are: {', '.join(VALID_LIBRARY_TYPES)}."
      )
    if inference_library == LIBRARY_TF:
      assert tf_installed()
    elif inference_library == LIBRARY_LITERT:
      if not litert_installed():
        raise ValueError(
          f"Parameter 'inference_library': Library '{LIBRARY_LITERT}' is not available. Install birdnet with [litert] option."
        )
    else:
      raise AssertionError()

    return super()._predict(
      inp=inp,
      backend_kwargs={
        "inference_library": inference_library,
        "in_idx": MODEL_IN_IDX,
        "out_idx": MODEL_LOGITS_OUT_IDX,
      },
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
      device="CPU",
    )
