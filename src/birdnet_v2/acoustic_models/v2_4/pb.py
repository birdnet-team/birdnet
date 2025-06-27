# birdnet_batch_inference.py – raw‑audio version
from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import final

import soundfile as sf  # pip install soundfile

# You'll need these imports in your own code
# Next two import lines for this demo only
from ordered_set import OrderedSet

# try:
#   import tflite_runtime.interpreter as tflite
# except ImportError:  # fallback to full TF (heavier)
from birdnet.utils import download_file_tqdm, get_species_from_file
from birdnet_v2.acoustic_models.base import AcousticInferenceBackend
from birdnet_v2.acoustic_models.pb import AcousticPBBackend
from birdnet_v2.acoustic_models.v2_4.base import AcousticModelBaseV2_4
from birdnet_v2.base import MODEL_BACKEND_PB, MODEL_BACKENDS
from birdnet_v2.local_data import get_local_model_root_dir


def check_protobuf_model_files_exist(folder: Path) -> bool:
  exists = True
  exists &= (folder / "saved_model.pb").is_file()
  exists &= (folder / "variables").is_dir()
  exists &= (folder / "variables" / "variables.data-00000-of-00001").is_file()
  exists &= (folder / "variables" / "variables.index").is_file()
  return exists


class AcousticPBDownloaderV2_4:
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
    for lang in cls._available_languages:
      model_is_downloaded &= (lang_dir / f"{lang}.txt").is_file()

    return model_is_downloaded

  @classmethod
  def _download_acoustic_model(cls) -> None:
    url = "https://zenodo.org/records/15050749/files/BirdNET_v2.4_protobuf.zip"
    dl_size = 124522908

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

      acoustic_model_dl_dir = extract_dir / "audio-model"
      species_dl_dir = extract_dir / "labels"

      acoustic_model_dir, acoustic_lang_dir = cls._get_paths()
      acoustic_model_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(acoustic_model_dl_dir, acoustic_model_dir)

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

    model_dir, langs_path = cls._get_paths()

    lang_file = langs_path / f"{lang_id}.txt"
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang_id}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return model_dir, labels


class AcousticPBModelV2_4(AcousticModelBaseV2_4):
  def __init__(self, device_name: str) -> None:
    self._device_name = device_name
    super().__init__()

  def get_backend_instance(self) -> AcousticInferenceBackend:
    return AcousticPBBackend(self.model_path, self._device_name)
  
  def get_backend_type(self) -> type[AcousticInferenceBackend]:
    return AcousticPBBackend

  def get_backend_args(self) -> dict:
    return {
      "model_path": self.model_path,
      "device": self._device_name,
    }

  @classmethod
  @final
  def get_backend(cls) -> MODEL_BACKENDS:
    return MODEL_BACKEND_PB

  @classmethod
  def load_official(cls, lang_id: str, device: str) -> AcousticPBModelV2_4:
    # result = cls.__new__(cls)
    # result.__init__(device)
    result = AcousticPBModelV2_4(device)
    result._load_official_model(lang_id)
    return result

  def _load_official_model(self, lang_id: str) -> None:
    self._model_path, self._species_list = (
      AcousticPBDownloaderV2_4.get_model_path_and_labels(lang_id)
    )
    self._use_custom_model = False

  @classmethod
  def load_custom(
    cls, model_path: Path, species_list: Path, device: str
  ) -> AcousticPBModelV2_4:
    result = AcousticPBModelV2_4(device)
    result._load_custom_model(model_path, species_list)
    return result

  def _load_custom_model(self, model_path: Path, species_list: Path) -> None:
    if not model_path.is_file():
      raise ValueError(f"Model file '{model_path.absolute()}' does not exist!")

    if not species_list.is_file():
      raise ValueError(f"Species list file '{species_list.absolute()}' does not exist!")

    try:
      import tensorflow as tf
      tf.saved_model.load(model_path)
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

    self._model_path = model_path
    self._species_list = loaded_species_list
    self._use_custom_model = True
