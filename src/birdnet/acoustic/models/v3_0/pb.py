import shutil
from pathlib import Path

from ordered_set import OrderedSet

from birdnet.acoustic.models.v3_0.model import AcousticDownloaderBaseV3_0
from birdnet.core.backends import (
  PBBackend,
  VersionedAcousticBackendProtocol,
)
from birdnet.globals import (
  MODEL_BACKEND_PB,
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
)
from birdnet.utils.downloads import (
  download_zip_members,
  ensure_download,
  saved_model_is_current,
)
from birdnet.utils.helper import get_species_from_file, write_source_marker
from birdnet.utils.local_data import get_lang_dir, get_model_path, get_model_root_dir

_PB_DL_URL = "https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_FP32_Protobuf.zip"
_PB_DL_SIZE = 499609919


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
  def _lock_dir(cls) -> Path:
    return get_model_root_dir("acoustic", "3.0", "pb") / ".download.lock"

  @classmethod
  def _check_model_files_available(cls) -> bool:
    model_path, _ = cls._get_paths()
    return saved_model_is_current(model_path, _PB_DL_URL)

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
    model_dir, lang_dir = cls._get_paths()
    download_zip_members(
      _PB_DL_URL, _PB_DL_SIZE, "acoustic model v3.0 (pb)", [(None, model_dir)]
    )
    # Before the marker: once the model reads as current, another process may
    # generate labels, and a wipe after that would take them from under it.
    shutil.rmtree(lang_dir, ignore_errors=True)
    write_source_marker(model_dir, _PB_DL_URL)

  @classmethod
  def get_model_path_and_labels(
    cls,
    lang: str,
  ) -> tuple[Path, OrderedSet[str]]:
    assert lang in cls.AVAILABLE_LANGUAGES

    # Only the model files gate the (large) download; labels are (re)generated
    # afterwards because _download_model() wipes the language directory.
    ensure_download(
      cls._check_model_files_available,
      cls._download_model,
      cls._lock_dir(),
      "the acoustic model v3.0 (pb) download",
    )
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

  @classmethod
  def applies_sigmoid(cls) -> bool:
    return True

  @classmethod
  def max_batch_size(cls) -> int | None:
    # XLA-compiled export with a static batch of 1: a larger batch fails shape
    # refinement inside the worker.
    return 1

  @classmethod
  def fixed_input_samples(cls) -> int | None:
    # The compilation fixes the input to 3.0 s as well; another length fails the
    # same shape refinement.
    return 96_000


class AcousticRavenBackendFP32V3_0(PBBackend, VersionedAcousticBackendProtocol):
  """A Raven export from BirdNET-Analyzer on the 3.0 base.

  Signature "basic" maps "inputs" (batch, 96000 samples) to "scores", with the
  sigmoid applied; there is no embeddings signature. It wraps the official
  SavedModel, so it inherits its static batch of 1 and its fixed input length.
  """

  def __init__(
    self, model_path: Path, device_name: str, half_precision: bool, **kwargs: dict
  ) -> None:
    super().__init__(model_path, device_name, half_precision, **kwargs)

  @classmethod
  def name(cls) -> str:
    return f"{MODEL_BACKEND_PB}-raven"

  @classmethod
  def input_key(cls) -> str:
    return "inputs"

  @classmethod
  def prediction_signature_name(cls) -> str:
    return "basic"

  @classmethod
  def prediction_key(cls) -> str:
    return "scores"

  @classmethod
  def supports_encoding(cls) -> bool:
    return False

  @classmethod
  def encoding_signature_name(cls) -> str | None:
    return None

  @classmethod
  def encoding_key(cls) -> str | None:
    return None

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return "fp32"

  @classmethod
  def applies_sigmoid(cls) -> bool:
    return True

  @classmethod
  def max_batch_size(cls) -> int | None:
    return 1

  @classmethod
  def fixed_input_samples(cls) -> int | None:
    return 96_000
