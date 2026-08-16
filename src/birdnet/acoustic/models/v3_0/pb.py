import shutil
import tempfile
import zipfile
from pathlib import Path

from ordered_set import OrderedSet

from birdnet.acoustic.models.v3_0.model import AcousticDownloaderBaseV3_0
from birdnet.core.backends import (
  PBBackend,
  VersionedAcousticBackendProtocol,
)
from birdnet.globals import (
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
)
from birdnet.utils.helper import (
  check_protobuf_model_files_exist,
  download_file_tqdm,
  get_species_from_file,
)
from birdnet.utils.local_data import get_lang_dir, get_model_path


class AcousticPBDownloaderV3_0(AcousticDownloaderBaseV3_0):
  @classmethod
  def _get_lang_dir(cls) -> Path:
    return get_lang_dir("acoustic", "3.0", "pb")

  @classmethod
  def _get_paths(cls) -> tuple[Path, Path]:
    model_path = get_model_path("acoustic", "3.0", "pb", MODEL_PRECISION_FP32)
    lang_dir = get_lang_dir("acoustic", "3.0", "pb")
    return model_path, lang_dir

  @classmethod
  def _check_model_files_available(cls) -> bool:
    model_path, _ = cls._get_paths()
    return model_path.is_dir() and check_protobuf_model_files_exist(model_path)

  @classmethod
  def _check_acoustic_model_available(cls) -> bool:
    if not cls._check_model_files_available():
      return False

    _, lang_dir = cls._get_paths()
    if not lang_dir.is_dir():
      return False
    return all((lang_dir / f"{lang}.txt").is_file() for lang in cls.AVAILABLE_LANGUAGES)

  @classmethod
  def _download_model(cls) -> None:
    dl_url = "https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_FP32_Protobuf.zip"
    dl_size = 499609919

    with tempfile.TemporaryDirectory(prefix="birdnet_download") as temp_dir:
      zip_download_path = Path(temp_dir) / "download.zip"
      download_file_tqdm(
        dl_url,
        zip_download_path,
        download_size=dl_size,
        description="Downloading acoustic model v3.0 (pb)",
      )

      print("Extracting...")  # noqa: T201
      extract_dir = Path(temp_dir) / "extracted"

      with zipfile.ZipFile(zip_download_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

      acoustic_model_dir, acoustic_lang_dir = cls._get_paths()
      acoustic_model_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.rmtree(acoustic_model_dir, ignore_errors=True)
      shutil.move(extract_dir, acoustic_model_dir)

      acoustic_lang_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.rmtree(acoustic_lang_dir, ignore_errors=True)
      print("Extracted.")  # noqa: T201

  @classmethod
  def get_model_path_and_labels(
    cls,
    lang: str,
  ) -> tuple[Path, OrderedSet[str]]:
    assert lang in cls.AVAILABLE_LANGUAGES

    # Only the model files gate the (large) download; labels are (re)generated
    # afterwards because _download_model() wipes the language directory.
    if not cls._check_model_files_available():
      cls._download_model()
    cls.ensure_labels_available()
    assert cls._check_acoustic_model_available()

    model_dir, langs_path = cls._get_paths()

    lang_file = langs_path / f"{lang}.txt"
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return model_dir, labels


class AcousticPBBackendFP32V3_0(PBBackend, VersionedAcousticBackendProtocol):
  # Unlike the v2.4 SavedModel (separate "basic" and "embeddings" signatures),
  # the v3.0 export exposes a single "serving_default" signature whose input is
  # "x" and which returns both "predictions" and "embeddings".
  def __init__(
    self, model_path: Path, device_name: str, half_precision: bool, **kwargs: dict
  ) -> None:
    super().__init__(model_path, device_name, half_precision, **kwargs)

  @classmethod
  def input_key(cls) -> str:
    return "x"

  @classmethod
  def prediction_signature_name(cls) -> str:
    return "serving_default"

  @classmethod
  def prediction_key(cls) -> str:
    return "predictions"

  @classmethod
  def supports_encoding(cls) -> bool:
    return True

  @classmethod
  def encoding_signature_name(cls) -> str | None:
    return "serving_default"

  @classmethod
  def encoding_key(cls) -> str | None:
    return "embeddings"

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP32
