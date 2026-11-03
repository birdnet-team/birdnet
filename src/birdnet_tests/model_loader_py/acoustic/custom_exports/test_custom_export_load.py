import pytest

import birdnet
from birdnet.acoustic.models.base import AcousticModelBase
from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet.acoustic.models.v2_4.tf import AcousticTFBackendCustomV2_4
from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.acoustic.models.v3_0.tf import AcousticTFBackendCustomFixedInputV3_0
from birdnet.core.backends import VersionedAcousticBackendProtocol
from birdnet_tests.custom_classifier_exports import ExportSpec
from birdnet_tests.model_loader_py.acoustic.custom_exports.conftest import (
  V2_4_DETACHED,
  V2_4_REPLACE,
  V3_0_REPLACE,
  GetExport,
)


# The real-file wiring the stubbed detection suite
# (test_acoustic_load_custom_tflite_detection.py) cannot show; what the
# inspection reads from every export is test_custom_export_inspect.py's.
@pytest.mark.parametrize(
  ("spec", "requested_version", "model_cls", "backend_type"),
  [
    (V2_4_REPLACE, "2.4", AcousticModelV2_4, AcousticTFBackendCustomV2_4),
    (V3_0_REPLACE, "auto", AcousticModelV3_0, AcousticTFBackendCustomFixedInputV3_0),
  ],
  ids=["2.4-explicit", "3.0-auto"],
)
def test_load_maps_a_real_export_to_its_backend(
  get_export: GetExport,
  spec: ExportSpec,
  requested_version: str,
  model_cls: type[AcousticModelBase],
  backend_type: type[VersionedAcousticBackendProtocol],
) -> None:
  export = get_export(spec)
  model = birdnet.load_custom(
    "acoustic", requested_version, "tf", export.model, export.labels
  )
  assert type(model) is model_cls
  assert model.backend_type is backend_type
  assert model.get_version() == spec.version
  assert model.is_custom_model


def test_load_rejects_a_detached_head(get_export: GetExport) -> None:
  export = get_export(V2_4_DETACHED)
  with pytest.raises(ValueError, match=r"detached classifier head"):
    birdnet.load_custom("acoustic", "2.4", "tf", export.model, export.labels)
