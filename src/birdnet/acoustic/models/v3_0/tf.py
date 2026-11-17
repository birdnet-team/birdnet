from __future__ import annotations

from birdnet.acoustic.models import custom_tflite
from birdnet.acoustic.models.custom_tflite import CustomTFLiteBackend
from birdnet.acoustic.models.v3_0.model import (
  AcousticModelV3_0,
  _AcousticSingleFileDownloaderV3_0,
)
from birdnet.core.backends import TFBackend, VersionedAcousticBackendProtocol
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
    dl_url="https://zenodo.org/records/24162518/files/BirdNET+_V3.0_Global_12K_FP16.tflite",
    dl_file_name="BirdNET+_V3.0_Global_12K_FP16.tflite",
    dl_size=172328340,
    file_size=172328340,
    sha256="78f9eb96609b2feef2f204ba81b1e6393f676c1fcf1666d922c96a5c2a7ed758",
  ),
  MODEL_PRECISION_FP32: ModelInfo(
    dl_url="https://zenodo.org/records/24162518/files/BirdNET+_V3.0_Global_12K_FP32.tflite",
    dl_file_name="BirdNET+_V3.0_Global_12K_FP32.tflite",
    dl_size=344132212,
    file_size=344132212,
    sha256="8b658ae7d3c9d7e1864833480c06f8f059c73fce5bcb7067c03a4bff439f53ce",
  ),
}


class AcousticTFDownloaderV3_0(_AcousticSingleFileDownloaderV3_0):
  _backend = MODEL_BACKEND_TF
  _models = models


class _AcousticTFBackendV3_0(TFBackend, VersionedAcousticBackendProtocol):
  @classmethod
  def supports_encoding(cls) -> bool:
    return True

  def _resolve_tensor_indices(
    self, interp: custom_tflite.Interpreter
  ) -> tuple[int, int, int | None]:
    # Read from the signature ("predictions", "embeddings"): the tensor indices
    # differ between exports of the model, and a stale one names another tensor.
    layout = custom_tflite.resolve_custom_tflite_layout(interp)
    n_classes = AcousticModelV3_0.get_n_classes()
    embeddings_dim = AcousticModelV3_0.get_embeddings_dim()
    if (
      layout.n_species != n_classes
      or not layout.embeddings_is_output
      or layout.embeddings_dim != embeddings_dim
    ):
      embeddings = (
        f"an embeddings output of width {layout.embeddings_dim}"
        if layout.embeddings_is_output
        else "no embeddings output"
      )
      raise ValueError(
        f"Model '{self._model_path.absolute()}' is not the official acoustic 3.0 "
        f"TFLite export: expected {n_classes} classes and an embeddings output of "
        f"width {embeddings_dim}, found {layout.n_species} classes and {embeddings}."
      )
    return layout.input_idx, layout.prediction_idx, layout.embeddings_idx

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
  def precision(cls) -> MODEL_PRECISIONS:
    return MODEL_PRECISION_FP32


class AcousticTFBackendFP16V3_0(_AcousticTFBackendV3_0):
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
