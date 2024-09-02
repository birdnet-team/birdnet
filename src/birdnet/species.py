import os
import zipfile
from pathlib import Path

from birdnet.models.v2m4.model_v2m4_base import AVAILABLE_LANGUAGES
from birdnet.types import Language
from birdnet.utils import download_file_tqdm


class SpeciesDownloader():
  def __init__(self, parent_folder: Path) -> None:
    self._version_path = parent_folder
    self._lang_path = self._version_path / "labels"

  @property
  def version_path(self) -> Path:
    return self._version_path

  def get_language_path(self, language: Language) -> Path:
    return self._lang_path / f"{language}.txt"

  def _check_model_files_exist(self) -> bool:
    model_is_downloaded = True

    model_is_downloaded &= self._lang_path.is_dir()
    for lang in AVAILABLE_LANGUAGES:
      model_is_downloaded &= self.get_language_path(lang).is_file()
    return model_is_downloaded

  def _download_species_files(self) -> None:
    dl_path = "https://tuc.cloud/index.php/s/45KmTcpHH8iDDA2/download/BirdNET_v2.4.zip"
    dl_size = 76823623
    self._version_path.mkdir(parents=True, exist_ok=True)

    zip_download_path = self._version_path / "download.zip"
    download_file_tqdm(dl_path, zip_download_path, download_size=dl_size,
                       description="Downloading models")

    with zipfile.ZipFile(zip_download_path, 'r') as zip_ref:
      zip_ref.extractall(self._version_path)

    os.remove(zip_download_path)

  def ensure_species_are_available(self) -> None:
    if not self._check_model_files_exist():
      self._download_species_files()
      assert self._check_model_files_exist()


class SpeciesNames():
  def __init__(self) -> None:
    pass
