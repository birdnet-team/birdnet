from pathlib import Path

from ordered_set import OrderedSet

from birdnet.acoustic.models.custom_tflite import CustomTFLiteBackend
from birdnet.acoustic.models.v3_0.model import AcousticDownloaderBaseV3_0
from birdnet.core.backends import TFBackend, VersionedAcousticBackendProtocol
from birdnet.globals import (
  ACOUSTIC_MODEL_VERSIONS,
  MODEL_BACKEND_TF,
  MODEL_PRECISION_FP16,
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
)
from birdnet.utils.helper import (
  ModelInfo,
  ensure_single_file_model,
  get_species_from_file,
)
from birdnet.utils.local_data import get_lang_dir, get_model_path

models = {
  MODEL_PRECISION_FP16: ModelInfo(
    dl_url="https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_FP16.tflite",
    dl_file_name="BirdNET+_V3.0-preview3.1_Global_11K_FP16.tflite",
    dl_size=270496424,
    file_size=270496424,
    sha256="5f6528f1456be3c682b776d72a4faa256a6710fa47ca47ebaeed4c69898cf5cf",
  ),
  MODEL_PRECISION_FP32: ModelInfo(
    dl_url="https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_FP32.tflite",
    dl_file_name="BirdNET+_V3.0-preview3.1_Global_11K_FP32.tflite",
    dl_size=540471440,
    file_size=540471440,
    sha256="a932aea50ec90984467e4c6da3d4d7bcc6a650a40e8bf35cf084219fef69fcd7",
  ),
}


class AcousticTFDownloaderV3_0(AcousticDownloaderBaseV3_0):
  @classmethod
  def _get_lang_dir(cls) -> Path:
    return get_lang_dir(
      "acoustic",
      "3.0",
      MODEL_BACKEND_TF,
    )

  @classmethod
  def _get_paths(cls, precision: MODEL_PRECISIONS) -> tuple[Path, Path]:
    model_path = get_model_path(
      "acoustic",
      "3.0",
      MODEL_BACKEND_TF,
      precision,
      content_tag=models[precision].content_tag,
    )
    lang_dir = get_lang_dir(
      "acoustic",
      "3.0",
      MODEL_BACKEND_TF,
    )
    return model_path, lang_dir

  @classmethod
  def _check_acoustic_model_available(cls, precision: MODEL_PRECISIONS) -> bool:
    model_path, lang_dir = cls._get_paths(precision)
    if not model_path.is_file():
      return False
    if model_path.stat().st_size != models[precision].file_size:
      return False
    if not lang_dir.is_dir():
      return False
    return all((lang_dir / f"{lang}.txt").is_file() for lang in cls.AVAILABLE_LANGUAGES)

  @classmethod
  def _download_model(cls, precision: MODEL_PRECISIONS) -> None:
    model_path, _lang_dir = cls._get_paths(precision)
    ensure_single_file_model(
      models[precision],
      model_path,
      legacy_path=get_model_path("acoustic", "3.0", MODEL_BACKEND_TF, precision),
      description=f"Downloading acoustic model v3.0 (tf, {precision})",
    )

  @classmethod
  def get_model_path_and_labels(
    cls, lang: str, precision: MODEL_PRECISIONS
  ) -> tuple[Path, OrderedSet[str]]:
    assert lang in cls.AVAILABLE_LANGUAGES

    cls.ensure_labels_available()
    if not cls._check_acoustic_model_available(precision):
      cls._download_model(precision)
    assert cls._check_acoustic_model_available(precision)

    model_path, _ = cls._get_paths(precision)
    lang_file = cls.get_lang_file(lang)
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return model_path, labels


class _AcousticTFBackendV3_0(TFBackend, VersionedAcousticBackendProtocol):
  @classmethod
  def in_idx(cls) -> int:
    return 0

  @classmethod
  def supports_encoding(cls) -> bool:
    return True

  @classmethod
  def probe_input_size_samples(cls) -> int:
    return 96_000

  @classmethod
  def applies_sigmoid(cls) -> bool:
    return True

  @classmethod
  def max_batch_size(cls) -> int | None:
    # The export fixes the batch dimension to 1 (input shape signature [1, -1]).
    return 1


class AcousticTFBackendFP32V3_0(_AcousticTFBackendV3_0):
  @classmethod
  def prediction_out_idx(cls) -> int:
    return 1164

  @classmethod
  def encoding_out_idx(cls) -> int | None:
    return 1090

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP32


class AcousticTFBackendFP16V3_0(_AcousticTFBackendV3_0):
  @classmethod
  def prediction_out_idx(cls) -> int:
    return 1546

  @classmethod
  def encoding_out_idx(cls) -> int | None:
    return 1472

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP16


class AcousticTFBackendCustomV3_0(
  CustomTFLiteBackend, VersionedAcousticBackendProtocol
):
  """A custom 3.0 classifier: signature outputs 'predictions' and 'embeddings'."""

  @classmethod
  def model_version(cls) -> ACOUSTIC_MODEL_VERSIONS:
    return "3.0"


class AcousticTFBackendCustomFP16V3_0(AcousticTFBackendCustomV3_0):
  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return "fp16"


class AcousticTFBackendCustomNoEmbeddingsV3_0(AcousticTFBackendCustomV3_0):
  """A custom 3.0 classifier whose file exposes no embeddings output."""

  @classmethod
  def supports_encoding(cls) -> bool:
    return False


class AcousticTFBackendCustomNoEmbeddingsFP16V3_0(
  AcousticTFBackendCustomNoEmbeddingsV3_0
):
  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return "fp16"
