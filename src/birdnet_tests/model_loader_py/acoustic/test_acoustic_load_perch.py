import pytest
from requests import ReadTimeout

import birdnet.model_loader as model_loader
from birdnet.acoustic.models.perch_v2 import pb as perch_pb
from birdnet.acoustic.models.perch_v2.model import AcousticModelPerchV2
from birdnet.model_loader import load_perch_v2
from birdnet_tests.helper import (
  ensure_gpu_or_skip,
  ensure_not_intel_macos_or_skip,
  ensure_tf_2_20_or_skip,
)


@pytest.mark.load_model
def test_perch_cpu_v2() -> None:
  ensure_not_intel_macos_or_skip()
  ensure_tf_2_20_or_skip()

  try:
    model = load_perch_v2("CPU")
  except ReadTimeout as e:
    pytest.fail(f"Model download timed out: {e}. Try again later.")
  assert isinstance(model, AcousticModelPerchV2)


def test_perch_cpu_v2_invalid_device_raise_error() -> None:
  ensure_not_intel_macos_or_skip()

  with pytest.raises(ValueError):
    load_perch_v2("TPU")  # type: ignore


def test_perch_cpu_v2_wrong_tf_version_raises_error(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(model_loader, "check_is_intel_macos", lambda: False)
  monkeypatch.setattr(perch_pb, "_get_tensorflow_version", lambda: "2.19.1")

  with pytest.raises(RuntimeError, match=r"TensorFlow >= 2\.20"):
    load_perch_v2("CPU")


@pytest.mark.gpu
@pytest.mark.load_model
def test_perch_gpu_v2() -> None:
  ensure_not_intel_macos_or_skip()
  ensure_tf_2_20_or_skip()
  ensure_gpu_or_skip()

  try:
    model = load_perch_v2("GPU")
  except ReadTimeout as e:
    pytest.fail(f"Model download timed out: {e}. Try again later.")
  assert isinstance(model, AcousticModelPerchV2)
