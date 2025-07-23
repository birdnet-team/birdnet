# birdnet_batch_inference.py – raw‑audio version
from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Literal, final


# You'll need these imports in your own code
# Next two import lines for this demo only
from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.prediction_result import PredictionResult
from birdnet.acoustic_models.v2_4.base import (
  AcousticDownloaderBaseV2_4,
  AcousticModelBaseV2_4,
)
from birdnet.backends import PBInferenceBackend
from birdnet.globals import (
  MODEL_BACKEND_PB,
  MODEL_BACKENDS,
  MODEL_PRECISION_FLOAT32,
)
from birdnet.helper import check_protobuf_model_files_exist, load_pb_model
from birdnet.local_data import get_local_model_root_dir
from birdnet.utils import download_file_tqdm, get_species_from_file


class AcousticPBDownloaderV2_4(AcousticDownloaderBaseV2_4):
  @classmethod
  def _get_paths(cls) -> tuple[Path, Path]:
    model_root = get_local_model_root_dir(
      AcousticPBModelV2_4.get_model_type(),
      AcousticPBModelV2_4.get_version(),
      AcousticPBModelV2_4.get_backend(),
    )

    model_dir = model_root / "model"
    lang_dir = model_root / "labels"
    return model_dir, lang_dir

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
    super().__init__(
      model_path, species_list, MODEL_PRECISION_FLOAT32, use_custom_model
    )

  @classmethod
  @final
  def get_backend(cls) -> MODEL_BACKENDS:
    return MODEL_BACKEND_PB

  @classmethod
  @final
  def get_backend_type(cls) -> type:
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

    # check not possible currently because of tf loading
    if False and check_validity:
      try:
        loaded_model = load_pb_model(model)
      except ValueError as e:
        raise ValueError(
          f"Failed to load model '{model.absolute()}'. Ensure it is a valid TFLite model."
        ) from e

      n_species_in_model = (
        loaded_model.signatures["basic"].output_shapes["scores"].dims[1].value  # type: ignore
      )
      if n_species_in_model != len(loaded_species_list):
        raise ValueError(
          f"Model '{model.absolute()}' has {n_species_in_model} outputs, but species list '{species_list.absolute()}' has {len(loaded_species_list)} species!"
        )

    result = AcousticPBModelV2_4(
      model_path=model, species_list=loaded_species_list, use_custom_model=True
    )

    return result

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
    device: str | list[str] = "CPU",
  ) -> PredictionResult:
    return super()._predict(
      inp,
      {
        "model_path": self.model_path,
        "signature_name": "basic",
        "prediction_key": "scores",
        "input_key": "inputs",
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
      device=device,
    )
