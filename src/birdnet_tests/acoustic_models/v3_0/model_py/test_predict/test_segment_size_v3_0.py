"""A 3.0 export with a fixed input length rejects any other segment_size_s."""

from pathlib import Path

import pytest
from ordered_set import OrderedSet

from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.acoustic.models.v3_0.onnx import AcousticOnnxBackendFP32V3_0
from birdnet.acoustic.models.v3_0.pb import (
  AcousticPBBackendFP32V3_0,
  AcousticRavenBackendFP32V3_0,
)
from birdnet.acoustic.models.v3_0.pt import AcousticPTBackendFP32V3_0
from birdnet.acoustic.models.v3_0.tf import (
  AcousticTFBackendCustomFixedInputV3_0,
  AcousticTFBackendCustomV3_0,
  AcousticTFBackendFP32V3_0,
)
from birdnet.core.backends import VersionedAcousticBackendProtocol

pytestmark = pytest.mark.no_tf

# Measured at 2.0 and 5.0 s: the pb and Raven exports fail shape refinement and
# a classifier export the strict input resize, inside the worker.
FIXED_INPUT = [
  AcousticPBBackendFP32V3_0,
  AcousticRavenBackendFP32V3_0,
  AcousticTFBackendCustomFixedInputV3_0,
]
# Measured: the official tf, onnx and pt exports run at 2.0, 4.0 and 5.0 s, also
# through load_custom.
DYNAMIC_INPUT = [
  AcousticTFBackendFP32V3_0,
  AcousticOnnxBackendFP32V3_0,
  AcousticPTBackendFP32V3_0,
  AcousticTFBackendCustomV3_0,
]


def _model(
  tmp_path: Path, backend_type: type[VersionedAcousticBackendProtocol]
) -> AcousticModelV3_0:
  model_path = tmp_path / "model"
  model_path.write_bytes(b"fake")
  return AcousticModelV3_0.load(model_path, OrderedSet(["species_a"]), backend_type, {})


@pytest.mark.parametrize("segment_size_s", [2.0, 5.0])
@pytest.mark.parametrize("backend_type", FIXED_INPUT, ids=lambda b: b.__name__)
def test_a_fixed_input_rejects_another_segment_size(
  tmp_path: Path,
  backend_type: type[VersionedAcousticBackendProtocol],
  segment_size_s: float,
) -> None:
  model = _model(tmp_path, backend_type)
  message = (
    rf"segment_size_s={segment_size_s} is not supported .* fixed input of 96000 "
    r"samples \(3\.0 s\)\. Use segment_size_s=3\.0"
  )
  with pytest.raises(ValueError, match=message):
    model.predict_session(top_k=None, segment_size_s=segment_size_s)
  with pytest.raises(ValueError, match=message):
    model.encode_session(segment_size_s=segment_size_s)


# Sessions below are not entered: entering would spawn workers that crash on the
# fake model file.
@pytest.mark.parametrize("backend_type", FIXED_INPUT, ids=lambda b: b.__name__)
def test_a_fixed_input_takes_its_own_length(
  tmp_path: Path, backend_type: type[VersionedAcousticBackendProtocol]
) -> None:
  model = _model(tmp_path, backend_type)
  model.predict_session(top_k=None, segment_size_s=3.0)
  if backend_type.supports_encoding():
    model.encode_session(segment_size_s=3.0)


@pytest.mark.parametrize(
  ("segment_size_s", "accepted"),
  [(2.99999, True), (3.00001, True), (2.999, False), (3.001, False)],
)
def test_a_fixed_input_compares_the_rounded_input_width(
  tmp_path: Path, segment_size_s: float, accepted: bool
) -> None:
  """2.99999 s is 96000 samples at 32 kHz, which the export takes."""
  model = _model(tmp_path, AcousticPBBackendFP32V3_0)
  if accepted:
    model.predict_session(top_k=None, segment_size_s=segment_size_s)
  else:
    with pytest.raises(ValueError, match=r"is not supported"):
      model.predict_session(top_k=None, segment_size_s=segment_size_s)


@pytest.mark.parametrize("backend_type", DYNAMIC_INPUT, ids=lambda b: b.__name__)
def test_a_dynamic_input_takes_another_segment_size(
  tmp_path: Path, backend_type: type[VersionedAcousticBackendProtocol]
) -> None:
  model = _model(tmp_path, backend_type)
  model.predict_session(top_k=None, segment_size_s=2.0)
  model.encode_session(segment_size_s=2.0)
