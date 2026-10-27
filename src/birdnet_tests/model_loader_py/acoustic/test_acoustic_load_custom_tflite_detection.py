"""How `load_custom` turns the inspection of a custom TFLite file into a backend.

The inspection itself is stubbed; the real files are covered under
`custom_exports/`.
"""

from pathlib import Path

import pytest

from birdnet.acoustic.models.custom_tflite import CustomTFLiteInfo
from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet.acoustic.models.v2_4.tf import (
  AcousticTFBackendCustomNoEmbeddingsV2_4,
  AcousticTFBackendCustomV2_4,
  AcousticTFBackendFP16V2_4,
  AcousticTFBackendInt8V2_4,
)
from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.acoustic.models.v3_0.tf import (
  AcousticTFBackendCustomFixedInputFP16V3_0,
  AcousticTFBackendCustomFixedInputNoEmbeddingsFP16V3_0,
  AcousticTFBackendCustomFixedInputNoEmbeddingsV3_0,
  AcousticTFBackendCustomFixedInputV3_0,
  AcousticTFBackendCustomFP16V3_0,
  AcousticTFBackendCustomNoEmbeddingsFP16V3_0,
  AcousticTFBackendCustomNoEmbeddingsV3_0,
  AcousticTFBackendCustomV3_0,
)
from birdnet.model_loader import _CUSTOM_TFLITE_BACKENDS, load_custom

pytestmark = pytest.mark.no_tf


@pytest.fixture
def model_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  monkeypatch.setattr("birdnet.model_loader.tf_installed", lambda: True)
  model = tmp_path / "classifier.tflite"
  model.write_bytes(b"tflite")
  return model


@pytest.fixture
def species_list(tmp_path: Path) -> Path:
  path = tmp_path / "labels.txt"
  path.write_text("a\nb\nc\n", encoding="utf-8")
  return path


def _inspection_returns(
  monkeypatch: pytest.MonkeyPatch, info: CustomTFLiteInfo
) -> None:
  monkeypatch.setattr(
    "birdnet.model_loader.BackendLoader.inspect_custom_tflite_model",
    lambda *args, **kwargs: info,
  )


@pytest.mark.parametrize(
  ("info", "version", "precision", "model_cls", "backend_type"),
  [
    (
      CustomTFLiteInfo("2.4", "replace", 3, False, True),
      "2.4",
      "fp32",
      AcousticModelV2_4,
      AcousticTFBackendCustomV2_4,
    ),
    (
      CustomTFLiteInfo("2.4", "append", 3, False, False),
      "2.4",
      "fp32",
      AcousticModelV2_4,
      AcousticTFBackendCustomNoEmbeddingsV2_4,
    ),
    (
      CustomTFLiteInfo("3.0", "replace", 3, True, True),
      "3.0",
      "fp32",
      AcousticModelV3_0,
      AcousticTFBackendCustomV3_0,
    ),
    (
      CustomTFLiteInfo("3.0", "replace", 3, True, True),
      "3.0",
      "fp16",
      AcousticModelV3_0,
      AcousticTFBackendCustomFP16V3_0,
    ),
    (
      CustomTFLiteInfo("3.0", "append", 3, True, False),
      "3.0",
      "fp32",
      AcousticModelV3_0,
      AcousticTFBackendCustomNoEmbeddingsV3_0,
    ),
    (
      CustomTFLiteInfo("3.0", "append", 3, True, False),
      "3.0",
      "fp16",
      AcousticModelV3_0,
      AcousticTFBackendCustomNoEmbeddingsFP16V3_0,
    ),
    (
      CustomTFLiteInfo("2.4", "replace", 3, False, True, 144_000),
      "2.4",
      "fp32",
      AcousticModelV2_4,
      AcousticTFBackendCustomV2_4,
    ),
    (
      CustomTFLiteInfo("3.0", "replace", 3, True, True, 96_000),
      "3.0",
      "fp32",
      AcousticModelV3_0,
      AcousticTFBackendCustomFixedInputV3_0,
    ),
    (
      CustomTFLiteInfo("3.0", "replace", 3, True, True, 96_000),
      "3.0",
      "fp16",
      AcousticModelV3_0,
      AcousticTFBackendCustomFixedInputFP16V3_0,
    ),
    (
      CustomTFLiteInfo("3.0", "append", 3, True, False, 96_000),
      "3.0",
      "fp32",
      AcousticModelV3_0,
      AcousticTFBackendCustomFixedInputNoEmbeddingsV3_0,
    ),
    (
      CustomTFLiteInfo("3.0", "append", 3, True, False, 96_000),
      "3.0",
      "fp16",
      AcousticModelV3_0,
      AcousticTFBackendCustomFixedInputNoEmbeddingsFP16V3_0,
    ),
  ],
)
@pytest.mark.parametrize("requested_version", ["explicit", "auto"])
def test_inspection_selects_the_backend(
  model_file: Path,
  species_list: Path,
  monkeypatch: pytest.MonkeyPatch,
  info: CustomTFLiteInfo,
  version: str,
  precision: str,
  model_cls: type,
  backend_type: type,
  requested_version: str,
) -> None:
  _inspection_returns(monkeypatch, info)
  model = load_custom(
    "acoustic",
    version if requested_version == "explicit" else "auto",
    "tf",
    model_file,
    species_list,
    precision=precision,
  )
  assert type(model) is model_cls
  assert model.backend_type is backend_type
  assert model.backend_kwargs == {"inference_library": "tflite"}


def test_every_mapped_backend_reports_what_it_was_mapped_for() -> None:
  """The result metadata carries the backend's own precision, not the key's."""
  for (
    version,
    precision,
    supports_encoding,
    fixed_input,
  ), backend in _CUSTOM_TFLITE_BACKENDS.items():
    assert backend.model_version() == version, backend
    assert backend.precision() == precision, backend
    assert backend.supports_encoding() == supports_encoding, backend
    assert (backend.fixed_input_samples() is not None) == fixed_input, backend


def test_a_fixed_input_rejects_other_segment_lengths(
  model_file: Path, species_list: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _inspection_returns(
    monkeypatch, CustomTFLiteInfo("3.0", "replace", 3, True, True, 96_000)
  )
  model = load_custom("acoustic", "3.0", "tf", model_file, species_list)
  with pytest.raises(
    ValueError, match=r"segment_size_s=2\.0 is not supported .* 96000 samples"
  ):
    model.predict_session(segment_size_s=2.0)
  # Not entered: entering would spawn workers that crash on the fake file.
  model.predict_session(top_k=None)


def test_a_dynamic_input_takes_other_segment_lengths(
  model_file: Path, species_list: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """The official 3.0 TFLite release, which load_custom also accepts."""
  _inspection_returns(monkeypatch, CustomTFLiteInfo("3.0", "replace", 3, True, True))
  model = load_custom("acoustic", "3.0", "tf", model_file, species_list)
  model.predict_session(top_k=None, segment_size_s=2.0)


def test_a_file_without_embeddings_cannot_encode(
  model_file: Path, species_list: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _inspection_returns(monkeypatch, CustomTFLiteInfo("3.0", "replace", 3, True, False))
  model = load_custom("acoustic", "3.0", "tf", model_file, species_list)
  with pytest.raises(ValueError, match=r"does not support embeddings"):
    model.encode_session(n_workers=1)


def test_the_requested_version_must_match_the_file(
  model_file: Path, species_list: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _inspection_returns(monkeypatch, CustomTFLiteInfo("3.0", "replace", 3, True, True))
  with pytest.raises(
    ValueError, match=r"is a BirdNET 3\.0 classifier, but version '2\.4'"
  ):
    load_custom("acoustic", "2.4", "tf", model_file, species_list)


def test_the_species_list_must_match_the_file(
  model_file: Path, species_list: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _inspection_returns(monkeypatch, CustomTFLiteInfo("2.4", "replace", 4, False, True))
  with pytest.raises(ValueError, match=r"has 4 outputs, but species list .* has 3"):
    load_custom("acoustic", "2.4", "tf", model_file, species_list)


def test_without_validity_check_nothing_is_inspected(
  model_file: Path, species_list: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  def _not_called(*args: object, **kwargs: object) -> None:
    raise AssertionError("inspected")

  monkeypatch.setattr(
    "birdnet.model_loader.BackendLoader.inspect_custom_tflite_model", _not_called
  )
  model = load_custom(
    "acoustic", "2.4", "tf", model_file, species_list, check_validity=False
  )
  assert model.backend_type is AcousticTFBackendCustomV2_4


@pytest.mark.parametrize(
  ("info", "backend_type"),
  [
    (CustomTFLiteInfo("3.0", "replace", 3, True, True), AcousticTFBackendCustomV3_0),
    (
      CustomTFLiteInfo("3.0", "append", 3, True, False),
      AcousticTFBackendCustomNoEmbeddingsV3_0,
    ),
    (
      CustomTFLiteInfo("3.0", "replace", 3, True, True, 96_000),
      AcousticTFBackendCustomFixedInputV3_0,
    ),
  ],
)
def test_auto_version_inspects_even_without_validity_check(
  model_file: Path,
  species_list: Path,
  monkeypatch: pytest.MonkeyPatch,
  info: CustomTFLiteInfo,
  backend_type: type,
) -> None:
  _inspection_returns(monkeypatch, info)
  model = load_custom(
    "acoustic", "auto", "tf", model_file, species_list, check_validity=False
  )
  assert type(model) is AcousticModelV3_0
  assert model.backend_type is backend_type


def test_classifier_type_is_still_validated(
  model_file: Path, species_list: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _inspection_returns(monkeypatch, CustomTFLiteInfo("3.0", "replace", 3, True, True))
  with pytest.raises(ValueError, match=r"Unknown classifier type: 'bad'"):
    load_custom(
      "acoustic", "3.0", "tf", model_file, species_list, classifier_type="bad"
    )


def test_auto_version_is_tflite_only(tmp_path: Path, species_list: Path) -> None:
  model_dir = tmp_path / "saved_model"
  model_dir.mkdir()
  with pytest.raises(ValueError, match=r"version='auto' is only supported"):
    load_custom("acoustic", "auto", "onnx", model_dir, species_list)


@pytest.mark.parametrize("check_validity", [True, False])
@pytest.mark.parametrize(
  ("precision", "backend_type"),
  [("fp16", AcousticTFBackendFP16V2_4), ("int8", AcousticTFBackendInt8V2_4)],
)
def test_quantized_2_4_precisions_take_the_official_backend(
  model_file: Path,
  species_list: Path,
  monkeypatch: pytest.MonkeyPatch,
  precision: str,
  backend_type: type,
  check_validity: bool,
) -> None:
  """Not inspected; `check_validity` still decides whether the file is opened."""

  def _not_called(*args: object, **kwargs: object) -> None:
    raise AssertionError("inspected")

  opened: list[object] = []

  def _open(*args: object, **kwargs: object) -> int:
    opened.append(args[1])
    return 3

  monkeypatch.setattr(
    "birdnet.model_loader.BackendLoader.inspect_custom_tflite_model", _not_called
  )
  monkeypatch.setattr(
    "birdnet.model_loader.BackendLoader.check_model_can_be_loaded", _open
  )
  model = load_custom(
    "acoustic",
    "2.4",
    "tf",
    model_file,
    species_list,
    precision=precision,
    check_validity=check_validity,
  )
  assert model.backend_type is backend_type
  assert opened == ([backend_type] if check_validity else [])
