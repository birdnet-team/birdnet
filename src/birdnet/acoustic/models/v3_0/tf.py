from birdnet.acoustic.models.custom_tflite import CustomTFLiteBackend
from birdnet.acoustic.models.v3_0.model import (
  AcousticModelV3_0,
  _AcousticSingleFileDownloaderV3_0,
)
from birdnet.core.backends import PinnedTFBackend, VersionedAcousticBackendProtocol
from birdnet.globals import (
  ACOUSTIC_MODEL_VERSIONS,
  MODEL_BACKEND_TF,
  MODEL_PRECISION_FP16,
  MODEL_PRECISION_FP32,
  MODEL_PRECISIONS,
)
from birdnet.utils.downloads import ModelInfo

models: dict[MODEL_PRECISIONS, ModelInfo] = {
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


class AcousticTFDownloaderV3_0(_AcousticSingleFileDownloaderV3_0):
  _backend = MODEL_BACKEND_TF
  _models = models


class _AcousticTFBackendV3_0(PinnedTFBackend, VersionedAcousticBackendProtocol):
  @classmethod
  def in_idx(cls) -> int:
    return 0

  @classmethod
  def supports_encoding(cls) -> bool:
    return True

  @classmethod
  def probe_input_size_samples(cls) -> int:
    return AcousticModelV3_0.get_segment_size_samples()

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


class _FixedInputV3_0:
  """A custom 3.0 classifier whose input shape signature fixes it to 3.0 s.

  The classifier exports wrap the XLA-compiled SavedModel; the official TFLite
  release, which load_custom also accepts, takes any length.
  """

  @classmethod
  def fixed_input_samples(cls) -> int | None:
    return AcousticModelV3_0.get_segment_size_samples()


class AcousticTFBackendCustomFixedInputV3_0(
  _FixedInputV3_0, AcousticTFBackendCustomV3_0
):
  pass


class AcousticTFBackendCustomFixedInputFP16V3_0(
  _FixedInputV3_0, AcousticTFBackendCustomFP16V3_0
):
  pass


class AcousticTFBackendCustomFixedInputNoEmbeddingsV3_0(
  _FixedInputV3_0, AcousticTFBackendCustomNoEmbeddingsV3_0
):
  pass


class AcousticTFBackendCustomFixedInputNoEmbeddingsFP16V3_0(
  _FixedInputV3_0, AcousticTFBackendCustomNoEmbeddingsFP16V3_0
):
  pass
