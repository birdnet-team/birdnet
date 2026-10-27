import numbers
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self, cast

from ordered_set import OrderedSet

from birdnet.acoustic.inference.configs import PredictionConfig
from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticEncodingResultBase,
)
from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticPredictionResultBase,
)
from birdnet.acoustic.inference.session import (
  AcousticEncodingSession,
  AcousticPredictionSession,
)
from birdnet.core.backends import BackendLoader, VersionedAcousticBackendProtocol
from birdnet.core.base import ModelBase
from birdnet.globals import ACOUSTIC_MODEL_VERSIONS
from birdnet.utils.segments import duration_as_samples
from birdnet.utils.species_lists import validate_species_list


def resolve_in_graph_sigmoid_options(
  apply_sigmoid: bool,
  apply_softmax: bool,
  sigmoid_sensitivity: float | None,
  model_desc: str,
) -> tuple[bool, float | None]:
  """Map the score options of a model with the sigmoid in its graph onto the pipeline's.

  Returns the ``(apply_sigmoid, sigmoid_sensitivity)`` the session runs with.
  The graph's own sigmoid is the result at the default sensitivity, so the
  pipeline's sigmoid stays off and the scores leave the model untouched (a
  second sigmoid would squash every score into [0.5, 0.73]). A non-default
  sensitivity turns it on: the worker recovers the logits from the scores and
  runs the pipeline sigmoid on them, exactly as on a logit-output model.
  ``apply_sigmoid=False`` returns the raw model output and ignores the
  sensitivity, as it does for a logit-output model.

  A softmax stays rejected: a score saturated at 1.0 recovers to an infinite
  logit, which the sigmoid clips but a softmax turns into NaN.
  """
  if apply_softmax:
    raise ValueError(
      f"apply_softmax is not supported for {model_desc}: the export applies a "
      "sigmoid inside the model graph, and the logits recovered from its scores "
      "are infinite wherever a score saturates, which a softmax turns into NaN."
    )
  if not apply_sigmoid:
    return False, None
  sigmoid_sensitivity = PredictionConfig.validate_sigmoid_sensitivity(
    sigmoid_sensitivity
  )
  if sigmoid_sensitivity == 1.0:
    return False, None
  return True, sigmoid_sensitivity


def validate_batch_size_for_backend(
  batch_size: object, backend_type: type[VersionedAcousticBackendProtocol]
) -> None:
  """Reject a batch the backend's export cannot process.

  Anything but an int is left to the session's own validation.
  """
  max_batch_size = backend_type.max_batch_size()
  if (
    max_batch_size is not None
    and isinstance(batch_size, int)
    and batch_size > max_batch_size
  ):
    raise ValueError(
      f"batch_size={batch_size} is not supported by the '{backend_type.name()}' "
      f"backend of this model: its export has a static batch dimension of "
      f"{max_batch_size}. Use batch_size={max_batch_size} (workers still run in "
      "parallel) or a backend with a dynamic batch, such as 'onnx' or 'pt' for "
      "the official 3.0 model."
    )


def validate_segment_size_for_backend(
  segment_size_s: object,
  sample_rate: int,
  backend_type: type[VersionedAcousticBackendProtocol],
) -> None:
  """Reject a segment length the backend's export cannot process.

  Anything but a number is left to the session's own validation.
  """
  fixed_samples = backend_type.fixed_input_samples()
  if fixed_samples is None or not isinstance(segment_size_s, numbers.Real):
    return
  # The model's input width, rounded as ModelConfig.segment_size_samples does.
  if duration_as_samples(cast(float, segment_size_s), sample_rate) != fixed_samples:
    fixed_s = fixed_samples / sample_rate
    raise ValueError(
      f"segment_size_s={segment_size_s} is not supported by the "
      f"'{backend_type.name()}' backend of this model: its export takes a fixed "
      f"input of {fixed_samples} samples ({fixed_s} s). Use "
      f"segment_size_s={fixed_s}; of the official 3.0 models, the 'tf', 'onnx' "
      "and 'pt' exports take other lengths."
    )


class AcousticModelBase(ModelBase, ABC):
  def __init__(
    self,
    model_path: Path,
    species_list: OrderedSet[str],
    is_custom_model: bool,
    backend_type: type[VersionedAcousticBackendProtocol],
    backend_kwargs: dict[str, Any],
  ) -> None:
    super().__init__(model_path, species_list, is_custom_model)
    self._backend_type = backend_type
    self._backend_kwargs = backend_kwargs

  @property
  def backend_type(self) -> type[VersionedAcousticBackendProtocol]:
    return self._backend_type

  @property
  def backend_kwargs(self) -> dict[str, Any]:
    return self._backend_kwargs

  @classmethod
  def load(
    cls,
    model_path: Path,
    species_list: OrderedSet[str],
    backend_type: type[VersionedAcousticBackendProtocol],
    backend_kwargs: dict[str, Any],
  ) -> Self:
    return cls(
      model_path,
      species_list,
      is_custom_model=False,
      backend_type=backend_type,
      backend_kwargs=backend_kwargs,
    )

  @classmethod
  def load_custom(
    cls,
    model_path: Path,
    species_list: Path,
    backend_type: type[VersionedAcousticBackendProtocol],
    backend_kwargs: dict[str, Any],
    check_validity: bool,
  ) -> Self:
    assert model_path.exists()
    assert species_list.is_file()

    loaded_species_list = validate_species_list(species_list)

    if check_validity:
      n_species_in_model = BackendLoader.check_model_can_be_loaded(
        model_path, backend_type, backend_kwargs
      )

      if n_species_in_model != len(loaded_species_list):
        raise ValueError(
          f"Model '{model_path.absolute()}' has {n_species_in_model} outputs, but "
          f"species list '{species_list.absolute()}' has "
          f"{len(loaded_species_list)} species!"
        )

    return cls(
      model_path,
      loaded_species_list,
      is_custom_model=True,
      backend_type=backend_type,
      backend_kwargs=backend_kwargs,
    )

  @classmethod
  @abstractmethod
  def get_version(cls) -> ACOUSTIC_MODEL_VERSIONS:  # noqa: ANN002
    """Return the string label that identifies the acoustic model version.

    Returns:
      ACOUSTIC_MODEL_VERSIONS: Registered enum constant for the supported version.
    """
    ...

  @classmethod
  @abstractmethod
  def get_sig_fmin(cls) -> int: ...

  @classmethod
  @abstractmethod
  def get_sig_fmax(cls) -> int: ...

  @classmethod
  @abstractmethod
  def get_sample_rate(cls) -> int: ...

  @classmethod
  @abstractmethod
  def get_segment_size_s(cls) -> float: ...

  @classmethod
  @abstractmethod
  def get_segment_size_samples(cls) -> int: ...

  @abstractmethod
  def predict(self, *args, **kwargs) -> AcousticPredictionResultBase:  # noqa: ANN002, ANN003
    ...

  @abstractmethod
  def predict_arrays(self, *args, **kwargs) -> AcousticPredictionResultBase:  # noqa: ANN002, ANN003
    ...

  @abstractmethod
  def predict_session(self, *args, **kwargs) -> AcousticPredictionSession:  # noqa: ANN002, ANN003
    ...

  @abstractmethod
  def encode(self, *args, **kwargs) -> AcousticEncodingResultBase:  # noqa: ANN002, ANN003
    ...

  @abstractmethod
  def encode_arrays(self, *args, **kwargs) -> AcousticEncodingResultBase:  # noqa: ANN002, ANN003
    ...

  @abstractmethod
  def encode_session(self, *args, **kwargs) -> AcousticEncodingSession:  # noqa: ANN002, ANN003
    ...
