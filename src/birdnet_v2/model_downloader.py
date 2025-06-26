# birdnet_batch_inference.py – raw‑audio version
from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import soundfile as sf  # pip install soundfile
from ordered_set import OrderedSet

from birdnet.utils import download_file_tqdm, get_species_from_file
from birdnet_v2.acoustic_models.v2_4.tf import AcousticTFModelV2_4
from birdnet_v2.base import MODEL_BACKENDS, MODEL_TYPES, MODEL_VERSIONS
from birdnet_v2.globals import APP_DIR


def get_species_from_file(
  species_file: Path, /, *, encoding: str = "utf-8"
) -> OrderedSet[str]:
  species = OrderedSet(species_file.read_text(encoding).splitlines())
  return species


class TFDownloaderV2_4:
  _available_langugaes: OrderedSet[str] = OrderedSet(
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
  def check_acoustic_model_available(cls) -> bool:
    model_is_downloaded = True

    model_path, lang_dir = ModelDownloader.get_model_and_labels_paths(
      AcousticTFModelV2_4.get_model_type(),
      AcousticTFModelV2_4.get_version(),
      AcousticTFModelV2_4.get_backend(),
    )

    model_is_downloaded &= model_path.is_file()
    if model_is_downloaded:
      file_stats = os.stat(model_path)
      audio_is_newest_version = file_stats.st_size == 51726412
      model_is_downloaded &= audio_is_newest_version

    model_is_downloaded &= lang_dir.is_dir()
    for lang in cls._available_langugaes:
      model_is_downloaded &= (lang_dir / f"{lang}.txt").is_file()
    return model_is_downloaded

  @classmethod
  def check_geo_model_available(cls) -> bool:
    model_is_downloaded = True

    model_path, lang_dir = ModelDownloader.get_model_and_labels_paths(
      "geo", "2.4", "tf"
    )

    model_is_downloaded &= model_path.is_file()
    if model_is_downloaded:
      file_stats = os.stat(model_path)
      meta_is_newest_version = file_stats.st_size == 29526096
      model_is_downloaded &= meta_is_newest_version

    model_is_downloaded &= lang_dir.is_dir()
    for lang in cls._available_langugaes:
      model_is_downloaded &= (lang_dir / f"{lang}.txt").is_file()
    return model_is_downloaded

  @classmethod
  def download_acoustic_and_geo_model(cls) -> None:
    DOWNLOAD_URL = "https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite.zip"
    DOWNLOAD_SIZE = 76822925

    with tempfile.TemporaryDirectory(prefix="birdnet_download") as temp_dir:
      zip_download_path = Path(temp_dir) / "download.zip"
      download_file_tqdm(
        DOWNLOAD_URL,
        zip_download_path,
        download_size=DOWNLOAD_SIZE,
        description="Downloading models",
      )

      extract_dir = Path(temp_dir) / "extracted"

      with zipfile.ZipFile(zip_download_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

      acoustic_model_dl_path = extract_dir / "audio-model.tflite"
      geo_model_dl_path = extract_dir / "meta-model.tflite"
      species_dl_dir = extract_dir / "labels"

      acoustic_model_path, acoustic_lang_dir = (
        ModelDownloader.get_model_and_labels_paths(
          AcousticTFModelV2_4.get_model_type(),
          AcousticTFModelV2_4.get_version(),
          AcousticTFModelV2_4.get_backend(),
        )
      )
      acoustic_model_path.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(acoustic_model_dl_path, acoustic_model_path)

      acoustic_lang_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(species_dl_dir, acoustic_lang_dir)

      geo_model_path, geo_lang_dir = ModelDownloader.get_model_and_labels_paths(
        "geo", "2.4", "tf"
      )

      geo_model_path.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(geo_model_dl_path, geo_model_path)

      geo_lang_dir.mkdir(parents=True, exist_ok=True)
      shutil.copytree(acoustic_lang_dir, geo_lang_dir, dirs_exist_ok=True)


class ModelDownloader:
  checkers = {
    ("acoustic", "2.4", "tf"): TFDownloaderV2_4.check_acoustic_model_available,
    ("geo", "2.4", "tf"): TFDownloaderV2_4.check_geo_model_available,
  }
  downloaders = {
    ("acoustic", "2.4", "tf"): TFDownloaderV2_4.download_acoustic_and_geo_model,
    ("geo", "2.4", "tf"): TFDownloaderV2_4.download_acoustic_and_geo_model,
  }

  @classmethod
  def get_model_and_labels_paths(
    cls,
    model: MODEL_TYPES,
    version: MODEL_VERSIONS,
    backend: MODEL_BACKENDS,
  ) -> tuple[Path, Path]:
    parent_dir = APP_DIR / f"{model}-models" / f"v{version}" / backend
    model_path = parent_dir / "model.tflite"
    lang_path = parent_dir / "labels"
    return model_path, lang_path

  @classmethod
  def get_model_path_and_labels(
    cls,
    model: MODEL_TYPES,
    version: MODEL_VERSIONS,
    backend: MODEL_BACKENDS,
    lang_id: str,
    download_if_not_available: bool = True,
  ) -> tuple[Path, OrderedSet[str]]:
    if download_if_not_available:
      cls.download_model_files(model, version, backend)

    model_path, langs_path = cls.get_model_and_labels_paths(model, version, backend)

    lang_file = langs_path / f"{lang_id}.txt"
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang_id}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return model_path, labels

  @classmethod
  def download_model_files(
    cls,
    model: MODEL_TYPES,
    version: MODEL_VERSIONS,
    backend: MODEL_BACKENDS,
  ) -> None:
    checker = cls.checkers.get((model, version, backend))
    assert checker is not None

    model_available = checker()
    if not model_available:
      downloader = cls.downloaders.get((model, version, backend))
      assert downloader is not None
      downloader()


if __name__ == "__main__":
  # Example usage
  ModelDownloader.download_model_files("acoustic", "2.4", "tf")
  ModelDownloader.download_model_files("geo", "2.4", "tf")
  print("Models downloaded successfully.")
