"""A TFLite backend runs on the CPU only; the other backends take any device."""

from pathlib import Path

import pytest

from birdnet.acoustic.inference.configs import ProcessingConfig
from birdnet.acoustic.models.v2_4.tf import AcousticTFBackendFP32V2_4
from birdnet.acoustic.models.v3_0.onnx import AcousticOnnxBackendFP32V3_0
from birdnet_tests.fake_acoustic_backend import (
  DEFAULT_N_SPECIES,
  FakeAcousticBackend,
  fake_predict_session,
)

pytestmark = pytest.mark.no_tf


class _TFLite(AcousticTFBackendFP32V2_4):
  n_species_out = DEFAULT_N_SPECIES


@pytest.mark.parametrize("device", ["GPU", "GPU:0", "CPU:0", ["CPU", "GPU"]])
def test_a_tflite_backend_takes_the_cpu_only(device: str | list[str]) -> None:
  with pytest.raises(ValueError, match=r"not available for the 'tf' backend"):
    ProcessingConfig.validate_device_for_backend(device, AcousticTFBackendFP32V2_4)


@pytest.mark.parametrize("device", ["CPU", ["CPU", "CPU"]])
def test_a_tflite_backend_takes_the_cpu(device: str | list[str]) -> None:
  ProcessingConfig.validate_device_for_backend(device, AcousticTFBackendFP32V2_4)


@pytest.mark.parametrize("backend", [AcousticOnnxBackendFP32V3_0, FakeAcousticBackend])
def test_another_backend_takes_a_gpu(backend: type) -> None:
  ProcessingConfig.validate_device_for_backend("GPU:1", backend)


def test_a_session_rejects_a_gpu_for_a_tflite_backend(tmp_path: Path) -> None:
  with pytest.raises(ValueError, match=r"device 'GPU' is not available"):
    fake_predict_session(tmp_path, backend=_TFLite, device="GPU")
