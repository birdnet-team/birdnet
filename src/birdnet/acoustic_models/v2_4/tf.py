# birdnet_batch_inference.py – raw‑audio version
# birdnet_batch_inference.py – raw‑audio version
from __future__ import annotations

import os
import shutil
import tempfile
import zipfile

# You'll need these imports in your own code
from pathlib import Path

# Next two import lines for this demo only
from typing import (
  final,
)

from ordered_set import OrderedSet

from birdnet.acoustic_models.base import AcousticInferenceBackend
from birdnet.acoustic_models.tf import AcousticTFBackend
from birdnet.acoustic_models.v2_4.base import AcousticModelBaseV2_4
from birdnet.base import (
  MODEL_BACKEND_TF,
  MODEL_BACKENDS,
)
from birdnet.local_data import get_local_model_root_dir

# try:
#   import tflite_runtime.interpreter as tflite
# except ImportError:  # fallback to full TF (heavier)
# from tensorflow.lite.python import interpreter as tflite
from birdnet.utils import download_file_tqdm, get_species_from_file


class AcousticTFDownloaderV2_4:
  _available_languages: OrderedSet[str] = OrderedSet(
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

  @classmethod
  def _get_paths(cls) -> tuple[Path, Path]:
    model_root = get_local_model_root_dir(
      AcousticTFModelV2_4.get_model_type(),
      AcousticTFModelV2_4.get_version(),
      AcousticTFModelV2_4.get_backend(),
    )

    model_path = model_root / "model.tflite"
    lang_dir = model_root / "labels"
    return model_path, lang_dir

  @classmethod
  def _check_acoustic_model_available(cls) -> bool:
    model_is_downloaded = True

    model_path, lang_dir = cls._get_paths()

    model_is_downloaded &= model_path.is_file()
    if model_is_downloaded:
      file_stats = os.stat(model_path)
      audio_is_newest_version = file_stats.st_size == 51726412
      model_is_downloaded &= audio_is_newest_version

    model_is_downloaded &= lang_dir.is_dir()
    for lang in cls._available_languages:
      model_is_downloaded &= (lang_dir / f"{lang}.txt").is_file()
    return model_is_downloaded

  @classmethod
  def _download_acoustic_model(cls) -> None:
    url = "https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite.zip"
    dl_size = 76822925

    with tempfile.TemporaryDirectory(prefix="birdnet_download") as temp_dir:
      zip_download_path = Path(temp_dir) / "download.zip"
      download_file_tqdm(
        url,
        zip_download_path,
        download_size=dl_size,
        description="Downloading models",
      )

      extract_dir = Path(temp_dir) / "extracted"

      with zipfile.ZipFile(zip_download_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

      acoustic_model_dl_path = extract_dir / "audio-model.tflite"
      species_dl_dir = extract_dir / "labels"

      acoustic_model_path, acoustic_lang_dir = cls._get_paths()
      acoustic_model_path.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(acoustic_model_dl_path, acoustic_model_path)

      acoustic_lang_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(species_dl_dir, acoustic_lang_dir)

  @classmethod
  def get_model_path_and_labels(
    cls,
    lang_id: str,
  ) -> tuple[Path, OrderedSet[str]]:
    if not cls._check_acoustic_model_available():
      cls._download_acoustic_model()
    assert cls._check_acoustic_model_available()

    model_path, langs_path = cls._get_paths()

    lang_file = langs_path / f"{lang_id}.txt"
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang_id}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return model_path, labels


class AcousticTFModelV2_4(AcousticModelBaseV2_4):
  def __init__(self) -> None:
    super().__init__()

  @classmethod
  @final
  def get_backend(cls) -> MODEL_BACKENDS:
    return MODEL_BACKEND_TF

  @classmethod
  @final
  def get_backend_type(cls) -> type[AcousticInferenceBackend]:
    return AcousticTFBackend

  @final
  def get_backend_args(self) -> dict:
    return {
      "model_path": self.model_path,
    }

  @classmethod
  def load_official(cls, lang_id: str) -> AcousticTFModelV2_4:
    result = cls.__new__(cls)
    result.__init__()
    result._load_official_model(lang_id)
    return result

  def _load_official_model(self, lang_id: str) -> None:
    self._model_path, self._species_list = (
      AcousticTFDownloaderV2_4.get_model_path_and_labels(lang_id)
    )
    self._use_custom_model = False

  @classmethod
  def load_custom(cls, model_path: Path, species_list: Path) -> AcousticTFModelV2_4:
    result = AcousticTFModelV2_4()
    result._load_custom_model(model_path, species_list)
    return result

  def _load_custom_model(self, model_path: Path, species_list: Path) -> None:
    if not model_path.is_file():
      raise ValueError(f"Model file '{model_path.absolute()}' does not exist!")

    if not species_list.is_file():
      raise ValueError(f"Species list file '{species_list.absolute()}' does not exist!")

    from tensorflow.lite.python import interpreter as tflite
    from tensorflow.lite.python.interpreter import OpResolverType

    try:
      interp = tflite.Interpreter(
        str(model_path.absolute()),
        num_threads=1,
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,  # tensor#187 is a dynamic-sized tensor
      )
    except ValueError as e:
      raise ValueError(
        f"Failed to load model '{model_path.absolute()}'. Ensure it is a valid TFLite model."
      ) from e

    loaded_species_list: OrderedSet[str]
    try:
      loaded_species_list = get_species_from_file(species_list, encoding="utf8")
    except Exception as e:
      raise ValueError(
        f"Failed to read species list from '{species_list.absolute()}'. Ensure it is a valid text file."
      ) from e

    output_size = interp.get_output_details()[0]["index"]
    if output_size != len(loaded_species_list):
      raise ValueError(
        f"Model '{model_path.absolute()}' has {output_size} outputs, but species list '{species_list.absolute()}' has {len(loaded_species_list)} species!"
      )

    self._model_path = model_path
    self._species_list = loaded_species_list
    self._use_custom_model = True
