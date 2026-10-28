"""Custom TFLite classifiers exported by BirdNET-Analyzer.

A custom classifier is the official model's backbone with a retrained head, so
its tensor indices depend on the head (hidden units, append vs. replace) and on
the Keras version that exported it. Nothing here relies on fixed indices: the
prediction and embeddings tensors are resolved from the file's SignatureDef,
with the pre-signature index table of old 2.4 exports as the fallback, and the
base version is derived from the input length and the embeddings width.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.core.backends import TFBackend, load_tf_model
from birdnet.globals import (
  ACOUSTIC_MODEL_VERSION_V3_0,
  ACOUSTIC_MODEL_VERSIONS,
  CUSTOM_CLASSIFIER_APPEND,
  CUSTOM_CLASSIFIER_APPEND_HIDDEN,
  CUSTOM_CLASSIFIER_REPLACE,
  CUSTOM_CLASSIFIER_REPLACE_HIDDEN,
  LIBRARY_TYPES,
  MODEL_PRECISIONS,
)

if TYPE_CHECKING:
  from ai_edge_litert.interpreter import (
    Interpreter as LiteRTInterpreter,
  )
  from tensorflow.lite.python.interpreter import Interpreter as TFInterpreter

  Interpreter: TypeAlias = LiteRTInterpreter | TFInterpreter

CUSTOM_CLASSIFIER_MODES = Literal["replace", "append"]

SERVING_SIGNATURE = "serving_default"
PREDICTIONS_OUTPUT = "predictions"
EMBEDDINGS_OUTPUT = "embeddings"

# The 2.4 backbone's pooling op. Its output tensor holds the 1024-d embeddings in
# every 2.4 export, whatever head sits on top, while the head is what shifts the
# tensor indices. The 2.4 exports never list it as a graph output.
_V2_4_EMBEDDINGS_TENSOR_MARKER = "GLOBAL_AVG_POOL"

# Prediction output index -> (classifier type, embeddings index) of 2.4 classifiers
# exported by BirdNET-Analyzer before Keras 3 (TensorFlow < 2.16). Only the keys
# are consulted, for files whose signature does not identify the prediction
# output; the embeddings index is superseded by the pooling tensor above.
LEGACY_V2_4_PREDICTION_INDICES: dict[int, tuple[str, int]] = {
  546: (CUSTOM_CLASSIFIER_REPLACE, 545),
  550: (CUSTOM_CLASSIFIER_APPEND, 547),
  549: (CUSTOM_CLASSIFIER_REPLACE_HIDDEN, 547),
  553: (CUSTOM_CLASSIFIER_APPEND_HIDDEN, 549),
}


@dataclass(frozen=True)
class CustomTFLiteInfo:
  """What `birdnet.inspect_custom_tflite` reports about a custom classifier.

  Attributes:
    version: The BirdNET model the classifier was trained on ("2.4" or "3.0").
    mode: "append" when the official species precede the custom ones in the
      output, "replace" when the output holds the custom species only.
    n_species: Width of the prediction output, i.e. the length its labels file
      must have.
    applies_sigmoid: True when the sigmoid is applied inside the graph
      (every 3.0 export); False when the output holds logits (the 2.4 exports),
      which the prediction session turns into scores itself.
    supports_encoding: Whether the file exposes the backbone's embeddings, so
      `encode(..)` works on it.
    input_samples: Length of the audio input in samples when the file fixes it,
      or None when it takes any length. A 3.0 classifier with a fixed input runs
      only with ``segment_size_s=3.0``.
  """

  version: ACOUSTIC_MODEL_VERSIONS
  mode: CUSTOM_CLASSIFIER_MODES
  n_species: int
  applies_sigmoid: bool
  supports_encoding: bool
  input_samples: int | None = None


@dataclass(frozen=True)
class CustomTFLiteLayout:
  """Tensor indices of one custom classifier file."""

  input_idx: int
  prediction_idx: int
  embeddings_idx: int | None
  # A tensor that is not a graph output is only readable after invoke() when
  # the interpreter preserves intermediate tensors.
  embeddings_is_output: bool
  n_species: int
  # From the input's shape signature; None when that dimension is dynamic.
  input_samples: int | None
  embeddings_dim: int | None


@dataclass(frozen=True)
class _BaseModel:
  version: ACOUSTIC_MODEL_VERSIONS
  segment_samples: int
  embeddings_dim: int
  # Official class count: a classifier with more outputs than this appends to
  # the official classes, one with at most this many replaces them.
  n_official_classes: int
  applies_sigmoid: bool


_BASE_MODELS = (
  _BaseModel(
    "2.4",
    AcousticModelV2_4.get_segment_size_samples(),
    AcousticModelV2_4.get_embeddings_dim(),
    n_official_classes=AcousticModelV2_4.get_n_classes(),
    applies_sigmoid=False,
  ),
  _BaseModel(
    "3.0",
    AcousticModelV3_0.get_segment_size_samples(),
    AcousticModelV3_0.get_embeddings_dim(),
    n_official_classes=AcousticModelV3_0.get_n_classes(),
    applies_sigmoid=True,
  ),
)


def _base_model(version: ACOUSTIC_MODEL_VERSIONS) -> _BaseModel:
  for base in _BASE_MODELS:
    if base.version == version:
      return base
  raise AssertionError(version)


def _shape_signature(detail: dict) -> list[int]:
  return [int(d) for d in detail.get("shape_signature", detail["shape"])]


def resolve_custom_tflite_layout(
  interp: Interpreter,
  legacy_prediction_indices: Collection[int] = (),
) -> CustomTFLiteLayout:
  """Locate the input, prediction and embeddings tensors of a classifier file.

  Works on an interpreter whose tensors are not allocated yet. The prediction
  tensor is the signature output named "predictions", else the signature's only
  output, else the file's only output, else the one legacy 2.4 index among the
  outputs. Embeddings come from the signature output named "embeddings" or, for
  2.4 files, from the backbone's pooling tensor.
  """
  inputs = interp.get_input_details()
  if len(inputs) != 1:
    raise ValueError(
      f"Expected one input tensor, found {len(inputs)}. Not a BirdNET classifier."
    )
  input_detail = inputs[0]
  input_shape = _shape_signature(input_detail)
  if len(input_shape) != 2:
    raise ValueError(
      f"Expected a 2-D (batch, samples) input, found shape {input_shape}. "
      "Not a BirdNET classifier."
    )
  input_samples = input_shape[1] if input_shape[1] > 0 else None

  outputs = {int(d["index"]): d for d in interp.get_output_details()}
  prediction_idx: int | None = None
  embeddings_idx: int | None = None

  signatures = interp.get_signature_list()
  if SERVING_SIGNATURE in signatures:
    runner = interp.get_signature_runner(SERVING_SIGNATURE)
    signature_outputs = {
      str(name): int(detail["index"])
      for name, detail in runner.get_output_details().items()
    }
    if PREDICTIONS_OUTPUT in signature_outputs:
      prediction_idx = signature_outputs[PREDICTIONS_OUTPUT]
    elif len(signature_outputs) == 1:
      (prediction_idx,) = signature_outputs.values()
    if EMBEDDINGS_OUTPUT in signature_outputs:
      embeddings_idx = signature_outputs[EMBEDDINGS_OUTPUT]

  if prediction_idx is None:
    if len(outputs) == 1:
      (prediction_idx,) = outputs
    else:
      legacy = [idx for idx in outputs if idx in legacy_prediction_indices]
      if len(legacy) != 1:
        raise ValueError(
          f"Cannot tell which of the {len(outputs)} output tensors holds the "
          "predictions: the file has no 'serving_default' signature naming a "
          "'predictions' output and none of the outputs is a known 2.4 "
          "classifier index."
        )
      (prediction_idx,) = legacy

  if prediction_idx not in outputs:
    raise ValueError(
      f"The signature's prediction tensor {prediction_idx} is not a model output."
    )
  prediction_shape = _shape_signature(outputs[prediction_idx])
  if len(prediction_shape) != 2 or prediction_shape[1] <= 0:
    raise ValueError(
      f"Expected a 2-D (batch, species) prediction output, found shape "
      f"{prediction_shape}."
    )

  if embeddings_idx is None:
    candidates = [
      d
      for d in interp.get_tensor_details()
      if _V2_4_EMBEDDINGS_TENSOR_MARKER in str(d["name"])
      and len(_shape_signature(d)) == 2
    ]
    if len(candidates) == 1:
      embeddings_idx = int(candidates[0]["index"])

  embeddings_dim: int | None = None
  if embeddings_idx is not None:
    embeddings_detail = (
      outputs[embeddings_idx]
      if embeddings_idx in outputs
      else next(
        d for d in interp.get_tensor_details() if int(d["index"]) == embeddings_idx
      )
    )
    embeddings_dim = _shape_signature(embeddings_detail)[-1]

  return CustomTFLiteLayout(
    input_idx=int(input_detail["index"]),
    prediction_idx=prediction_idx,
    embeddings_idx=embeddings_idx,
    embeddings_is_output=embeddings_idx in outputs,
    n_species=prediction_shape[1],
    input_samples=input_samples,
    embeddings_dim=embeddings_dim,
  )


def describe_custom_tflite_layout(layout: CustomTFLiteLayout) -> CustomTFLiteInfo:
  """Derive the base version and classifier mode from a resolved layout."""
  by_samples = {b.segment_samples: b for b in _BASE_MODELS}
  by_embeddings = {b.embeddings_dim: b for b in _BASE_MODELS}

  base: _BaseModel | None = None
  if layout.input_samples is not None:
    base = by_samples.get(layout.input_samples)
    if base is None:
      hint = ""
      if layout.input_samples in by_embeddings:
        hint = (
          " This looks like a detached classifier head, which takes embeddings "
          "instead of audio; birdnet needs the full classifier export."
        )
      expected = " or ".join(
        f"{b.segment_samples} (BirdNET {b.version})" for b in _BASE_MODELS
      )
      raise ValueError(
        f"The input takes {layout.input_samples} samples, but a BirdNET "
        f"classifier takes {expected}.{hint}"
      )
  if layout.embeddings_dim is not None:
    from_embeddings = by_embeddings.get(layout.embeddings_dim)
    if from_embeddings is None:
      expected = " or ".join(
        f"{b.embeddings_dim} (BirdNET {b.version})" for b in _BASE_MODELS
      )
      raise ValueError(
        f"The embeddings are {layout.embeddings_dim} wide, but BirdNET "
        f"embeddings are {expected} wide."
      )
    if base is not None and from_embeddings is not base:
      raise ValueError(
        f"The input takes {layout.input_samples} samples (BirdNET "
        f"{base.version}), but the embeddings are {layout.embeddings_dim} wide "
        f"(BirdNET {from_embeddings.version})."
      )
    base = from_embeddings
  if base is None:
    raise ValueError(
      "Neither the input length nor the embeddings width identifies the "
      "BirdNET version this classifier was trained on."
    )

  mode: CUSTOM_CLASSIFIER_MODES = (
    "append" if layout.n_species > base.n_official_classes else "replace"
  )
  return CustomTFLiteInfo(
    version=base.version,
    mode=mode,
    n_species=layout.n_species,
    applies_sigmoid=base.applies_sigmoid,
    supports_encoding=layout.embeddings_idx is not None,
    input_samples=layout.input_samples,
  )


def inspect_custom_tflite_file(
  model_path: Path,
  library: LIBRARY_TYPES,
  legacy_prediction_indices: Collection[int] = (),
) -> CustomTFLiteInfo:
  """Open the file and describe it; imports the interpreter into this process.

  The tensors are allocated as well, so a file the interpreter cannot run is
  rejected here rather than by the first prediction.
  """
  interp = load_tf_model(model_path, library, allocate_tensors=True)
  layout = resolve_custom_tflite_layout(interp, legacy_prediction_indices)
  return describe_custom_tflite_layout(layout)


class CustomTFLiteBackend(TFBackend, ABC):
  """TFLite backend whose tensors are resolved from the file at load time."""

  def __init__(
    self, model_path: Path, device_name: str, half_precision: bool, **kwargs: object
  ) -> None:
    super().__init__(model_path, device_name, half_precision, **kwargs)
    self._layout: CustomTFLiteLayout | None = None
    self._preserving_tensors = False

  @classmethod
  def model_version(cls) -> ACOUSTIC_MODEL_VERSIONS:
    raise NotImplementedError

  @classmethod
  def legacy_prediction_indices(cls) -> Collection[int]:
    return ()

  @classmethod
  def precision(cls) -> MODEL_PRECISIONS:
    return "fp32"

  @classmethod
  def supports_encoding(cls) -> bool:
    return True

  @classmethod
  def applies_sigmoid(cls) -> bool:
    return _base_model(cls.model_version()).applies_sigmoid

  @classmethod
  def max_batch_size(cls) -> int | None:
    # The 3.0 exports wrap an XLA-compiled SavedModel with a static batch of 1;
    # the interpreter accepts a larger input but returns a single row for it.
    if cls.model_version() == ACOUSTIC_MODEL_VERSION_V3_0:
      return 1
    return None

  def _resolve_tensor_indices(
    self, interp: LiteRTInterpreter | TFInterpreter
  ) -> tuple[int, int, int | None]:
    layout = resolve_custom_tflite_layout(interp, self.legacy_prediction_indices())
    info = describe_custom_tflite_layout(layout)
    if info.version != self.model_version():
      raise ValueError(
        f"Model '{self._model_path.absolute()}' is a BirdNET {info.version} "
        f"classifier, but it was loaded as version '{self.model_version()}'. "
        f"Pass version='{info.version}' or version='auto' to load_custom."
      )
    self._layout = layout
    return layout.input_idx, layout.prediction_idx, layout.embeddings_idx

  def unload(self) -> None:
    super().unload()
    self._layout = None
    self._preserving_tensors = False

  def _prepare_encoding(self) -> None:
    assert self._layout is not None
    if self._layout.embeddings_idx is None:
      raise ValueError(
        f"Model '{self._model_path.absolute()}' exposes no embeddings tensor, so "
        "it supports predict(..) but not encode(..)."
      )
    if self._layout.embeddings_is_output or self._preserving_tensors:
      return
    # Reloaded on the first encode only: preservation costs ~110 MB per
    # interpreter, which a prediction session need not pay.
    self._interp = load_tf_model(
      self._model_path,
      self._inference_library,
      allocate_tensors=True,
      preserve_all_tensors=True,
    )
    self._cached_shape = None
    self._preserving_tensors = True
