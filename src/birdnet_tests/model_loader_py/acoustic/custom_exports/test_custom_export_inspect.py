from pathlib import Path

import pytest

import birdnet
from birdnet.acoustic.models.custom_tflite import CustomTFLiteInfo
from birdnet_tests.custom_classifier_exports import ExportSpec
from birdnet_tests.helper import ensure_litert_or_skip, ensure_validity_check_or_skip
from birdnet_tests.model_loader_py.acoustic.custom_exports.conftest import (
  EXPORT_TEST_TIMEOUT_S,
  V2_4_APPEND,
  V2_4_DETACHED,
  V2_4_REPLACE,
  V2_4_REPLACE_HIDDEN,
  V3_0_APPEND,
  V3_0_DETACHED,
  V3_0_REPLACE,
  GetExport,
)
from birdnet_tests.test_files import TEST_FILES_DIR

pytestmark = pytest.mark.timeout(EXPORT_TEST_TIMEOUT_S)

EXPECTED: dict[ExportSpec, CustomTFLiteInfo] = {
  V2_4_REPLACE: CustomTFLiteInfo("2.4", "replace", 3, False, True),
  V2_4_APPEND: CustomTFLiteInfo("2.4", "append", 6522 + 3, False, True),
  V2_4_REPLACE_HIDDEN: CustomTFLiteInfo("2.4", "replace", 3, False, True),
  V3_0_REPLACE: CustomTFLiteInfo("3.0", "replace", 3, True, True),
  V3_0_APPEND: CustomTFLiteInfo("3.0", "append", 11560 + 3, True, True),
}

# Exports of earlier analyzer releases, checked in under TEST_FILES.
LEGACY_DIR = TEST_FILES_DIR / "custom_models" / "tf"
EXPECTED_LEGACY: dict[str, CustomTFLiteInfo] = {
  "replace.tflite": CustomTFLiteInfo("2.4", "replace", 4, False, True),
  "append.tflite": CustomTFLiteInfo("2.4", "append", 6522 + 3, False, True),
  "replace+hidden1024.tflite": CustomTFLiteInfo("2.4", "replace", 3, False, True),
  "append+hidden1024.tflite": CustomTFLiteInfo("2.4", "append", 6522 + 3, False, True),
}


@pytest.mark.parametrize("spec", list(EXPECTED), ids=lambda s: s.name)
def test_inspect_describes_the_export(get_export: GetExport, spec: ExportSpec) -> None:
  export = get_export(spec)
  info = birdnet.inspect_custom_tflite(export.model)
  assert info == EXPECTED[spec]
  assert info.n_species == export.n_species


@pytest.mark.parametrize("name", list(EXPECTED_LEGACY))
def test_inspect_describes_the_legacy_fixtures(name: str) -> None:
  ensure_validity_check_or_skip()
  assert birdnet.inspect_custom_tflite(LEGACY_DIR / name) == EXPECTED_LEGACY[name]


@pytest.mark.litert
@pytest.mark.parametrize("spec", [V2_4_APPEND, V3_0_REPLACE], ids=lambda s: s.name)
def test_inspect_with_litert(get_export: GetExport, spec: ExportSpec) -> None:
  ensure_litert_or_skip()
  export = get_export(spec)
  assert birdnet.inspect_custom_tflite(export.model, library="litert") == EXPECTED[spec]


@pytest.mark.parametrize("spec", [V2_4_DETACHED, V3_0_DETACHED], ids=lambda s: s.name)
def test_inspect_rejects_a_detached_head(
  get_export: GetExport, spec: ExportSpec
) -> None:
  export = get_export(spec)
  with pytest.raises(ValueError, match=r"detached classifier head"):
    birdnet.inspect_custom_tflite(export.model)


def test_inspect_rejects_a_non_tflite_path(tmp_path: Path) -> None:
  model = tmp_path / "model.onnx"
  model.write_bytes(b"x")
  with pytest.raises(ValueError, match=r"is not a valid TFLite model file"):
    birdnet.inspect_custom_tflite(model)
