"""Which TFLite backends run on XNNPACK, and which run a batch row by row.

The acoustic float models and the custom classifiers do both. The 2.4 int8
model and the 3.0 fp16 files keep the builtin kernels: XNNPACK changes the
former's logits by more than a logit, and may compute the latter in half
precision. The geo models are small and run as they always did.
"""

import pytest

from birdnet.acoustic.models.v2_4.tf import (
  AcousticTFBackendCustomV2_4,
  AcousticTFBackendFP16V2_4,
  AcousticTFBackendFP32V2_4,
  AcousticTFBackendInt8V2_4,
)
from birdnet.acoustic.models.v3_0.tf import (
  AcousticTFBackendCustomFixedInputFP16V3_0,
  AcousticTFBackendCustomFixedInputV3_0,
  AcousticTFBackendCustomFP16V3_0,
  AcousticTFBackendCustomV3_0,
  AcousticTFBackendFP16V3_0,
  AcousticTFBackendFP32V3_0,
)
from birdnet.core import backends
from birdnet.core.backends import TFBackend
from birdnet.geo.models.v2_4.tf import GeoTFBackendFP32V2_4
from birdnet.geo.models.v3_0.tf import GeoTFBackendFP32V3_0

pytestmark = pytest.mark.no_tf


@pytest.mark.parametrize(
  ("backend", "delegate", "row_by_row"),
  [
    (AcousticTFBackendFP32V2_4, True, True),
    (AcousticTFBackendFP16V2_4, True, True),
    (AcousticTFBackendInt8V2_4, False, True),
    (AcousticTFBackendCustomV2_4, True, True),
    (AcousticTFBackendFP32V3_0, True, True),
    (AcousticTFBackendFP16V3_0, False, True),
    (AcousticTFBackendCustomV3_0, True, True),
    (AcousticTFBackendCustomFixedInputV3_0, True, True),
    (AcousticTFBackendCustomFP16V3_0, False, True),
    (AcousticTFBackendCustomFixedInputFP16V3_0, False, True),
    (GeoTFBackendFP32V2_4, False, False),
    (GeoTFBackendFP32V3_0, False, False),
  ],
)
def test_choices(backend: type[TFBackend], delegate: bool, row_by_row: bool) -> None:
  assert backend.uses_default_delegates() is delegate
  assert backend.runs_rows_one_by_one() is row_by_row


class _OpResolverType:
  AUTO = "auto"
  BUILTIN_WITHOUT_DEFAULT_DELEGATES = "builtin without default delegates"


class _Interpreters:
  OpResolverType = _OpResolverType


def test_the_default_delegates_come_with_the_auto_resolver() -> None:
  assert backends._op_resolver_type(_Interpreters, True) == "auto"
  assert (
    backends._op_resolver_type(_Interpreters, False)
    == "builtin without default delegates"
  )
