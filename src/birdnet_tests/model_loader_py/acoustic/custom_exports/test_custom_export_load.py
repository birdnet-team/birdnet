from pathlib import Path

import pytest

import birdnet
from birdnet.acoustic.models.base import AcousticModelBase
from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet.acoustic.models.v2_4.tf import AcousticTFBackendCustomV2_4
from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.acoustic.models.v3_0.tf import (
  AcousticTFBackendCustomFixedInputV3_0,
  AcousticTFBackendCustomV3_0,
)
from birdnet.core.backends import VersionedAcousticBackendProtocol
from birdnet_tests.custom_classifier_exports import ExportSpec
from birdnet_tests.model_loader_py.acoustic.custom_exports.conftest import (
  EXPORT_TEST_TIMEOUT_S,
  V2_4_APPEND,
  V2_4_DETACHED,
  V2_4_REPLACE,
  V2_4_REPLACE_HIDDEN,
  V3_0_APPEND,
  V3_0_REPLACE,
  GetExport,
)

pytestmark = pytest.mark.timeout(EXPORT_TEST_TIMEOUT_S)

CASES: list[
  tuple[ExportSpec, type[AcousticModelBase], type[VersionedAcousticBackendProtocol]]
] = [
  (V2_4_REPLACE, AcousticModelV2_4, AcousticTFBackendCustomV2_4),
  (V2_4_APPEND, AcousticModelV2_4, AcousticTFBackendCustomV2_4),
  (V2_4_REPLACE_HIDDEN, AcousticModelV2_4, AcousticTFBackendCustomV2_4),
  (V3_0_REPLACE, AcousticModelV3_0, AcousticTFBackendCustomFixedInputV3_0),
  (V3_0_APPEND, AcousticModelV3_0, AcousticTFBackendCustomFixedInputV3_0),
]


@pytest.mark.parametrize(
  ("spec", "model_cls", "backend_type"), CASES, ids=lambda c: str(c)
)
def test_load_with_explicit_version(
  get_export: GetExport,
  spec: ExportSpec,
  model_cls: type[AcousticModelBase],
  backend_type: type[VersionedAcousticBackendProtocol],
) -> None:
  export = get_export(spec)
  model = birdnet.load_custom(
    "acoustic", spec.version, "tf", export.model, export.labels
  )
  assert type(model) is model_cls
  assert model.backend_type is backend_type
  assert model.is_custom_model


@pytest.mark.parametrize(
  ("spec", "model_cls", "backend_type"), CASES, ids=lambda c: str(c)
)
def test_load_with_auto_version(
  get_export: GetExport,
  spec: ExportSpec,
  model_cls: type[AcousticModelBase],
  backend_type: type[VersionedAcousticBackendProtocol],
) -> None:
  export = get_export(spec)
  model = birdnet.load_custom("acoustic", "auto", "tf", export.model, export.labels)
  assert type(model) is model_cls
  assert model.backend_type is backend_type
  assert model.get_version() == spec.version


@pytest.mark.parametrize(
  ("spec", "wrong_version"),
  [(V3_0_REPLACE, "2.4"), (V2_4_REPLACE, "3.0")],
  ids=lambda v: str(v),
)
def test_load_rejects_the_other_version(
  get_export: GetExport, spec: ExportSpec, wrong_version: str
) -> None:
  export = get_export(spec)
  with pytest.raises(
    ValueError,
    match=rf"is a BirdNET {spec.version} classifier, but version '{wrong_version}'",
  ):
    birdnet.load_custom("acoustic", wrong_version, "tf", export.model, export.labels)


def test_load_without_validity_check_skips_the_inspection(
  get_export: GetExport, monkeypatch: pytest.MonkeyPatch
) -> None:
  export = get_export(V3_0_REPLACE)

  def _not_called(*args: object, **kwargs: object) -> None:
    raise AssertionError("inspected although check_validity=False")

  monkeypatch.setattr(
    "birdnet.model_loader.BackendLoader.inspect_custom_tflite_model", _not_called
  )
  model = birdnet.load_custom(
    "acoustic", "3.0", "tf", export.model, export.labels, check_validity=False
  )
  # Not inspected, so its fixed input is unknown and not guarded.
  assert model.backend_type is AcousticTFBackendCustomV3_0


def test_a_3_0_export_rejects_another_segment_size(get_export: GetExport) -> None:
  export = get_export(V3_0_REPLACE)
  model = birdnet.load_custom("acoustic", "3.0", "tf", export.model, export.labels)
  with pytest.raises(
    ValueError, match=r"segment_size_s=2\.0 is not supported .* 96000 samples"
  ):
    model.predict_session(segment_size_s=2.0)


def test_load_rejects_a_species_list_of_the_wrong_length(
  get_export: GetExport, tmp_path: Path
) -> None:
  export = get_export(V2_4_REPLACE)
  species = tmp_path / "species.txt"
  species.write_text("a\nb\n", encoding="utf-8")
  with pytest.raises(ValueError, match=r"has 3 outputs, but species list .* has 2"):
    birdnet.load_custom("acoustic", "2.4", "tf", export.model, species)


def test_load_rejects_a_detached_head(get_export: GetExport) -> None:
  export = get_export(V2_4_DETACHED)
  with pytest.raises(ValueError, match=r"detached classifier head"):
    birdnet.load_custom("acoustic", "2.4", "tf", export.model, export.labels)


def test_auto_version_needs_the_tf_backend(
  get_export: GetExport, tmp_path: Path
) -> None:
  export = get_export(V2_4_REPLACE)
  with pytest.raises(ValueError, match=r"version='auto' is only supported"):
    birdnet.load_custom("acoustic", "auto", "pb", tmp_path, export.labels)
