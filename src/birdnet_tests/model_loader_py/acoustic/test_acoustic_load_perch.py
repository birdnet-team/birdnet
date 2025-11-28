import pytest

from birdnet.acoustic_models.perch_v2.model import AcousticModelPerchV2
from birdnet.model_loader import load_perch_v2
from birdnet_tests.helper import ensure_gpu_or_skip


@pytest.mark.load_model
def test_perch_cpu_v2() -> None:
  model = load_perch_v2("CPU")
  assert isinstance(model, AcousticModelPerchV2)


def test_perch_cpu_v2_invalid_device_raise_error() -> None:
  with pytest.raises(ValueError):
    load_perch_v2("TPU")  # type: ignore


@pytest.mark.gpu
@pytest.mark.load_model
def test_perch_gpu_v2() -> None:
  ensure_gpu_or_skip()

  model = load_perch_v2("GPU")
  assert isinstance(model, AcousticModelPerchV2)
