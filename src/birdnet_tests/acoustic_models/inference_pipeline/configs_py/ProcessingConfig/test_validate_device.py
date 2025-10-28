import pytest

from birdnet.acoustic_models.inference_pipeline.configs import ProcessingConfig


def test_CPU_is_valid() -> None:
  assert ProcessingConfig.validate_device("CPU", workers=1) == "CPU"


def test_GPU_is_valid() -> None:
  assert ProcessingConfig.validate_device("GPU", workers=1) == "GPU"


def test_invalid_device_TPU_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"device must contain 'CPU' or 'GPU'",
  ):
    ProcessingConfig.validate_device("TPU", workers=1)


def test_invalid_device_empty_string_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"device must contain 'CPU' or 'GPU'",
  ):
    ProcessingConfig.validate_device("", workers=1)


def test_device_list_length_mismatch_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"length of device list \(2\) must match number of workers \(1\)",
  ):
    ProcessingConfig.validate_device(["CPU", "GPU"], workers=1)


def test_device_list_with_invalid_device_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"device must contain 'CPU' or 'GPU'",
  ):
    ProcessingConfig.validate_device(["CPU", "TPU"], workers=2)


def test_valid_device_list() -> None:
  devices = ["CPU", "GPU", "CPU"]
  assert ProcessingConfig.validate_device(devices, workers=3) == devices


def test_empty_device_list_length_mismatch_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"length of device list \(0\) must match number of workers \(1\)",
  ):
    ProcessingConfig.validate_device([], workers=1)


def test_non_string_non_list_raises_error() -> None:
  with pytest.raises(
    TypeError,
    match=r"device must be a string or a list of strings",
  ):
    ProcessingConfig.validate_device(42, workers=1)  # type: ignore


def test_device_list_with_non_string_raises_error() -> None:
  with pytest.raises(
    TypeError,
    match=r"device must be a string or a list of strings",
  ):
    ProcessingConfig.validate_device(["CPU", 42], workers=2)  # type: ignore


def test_device_list_with_empty_string_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"device must contain 'CPU' or 'GPU'",
  ):
    ProcessingConfig.validate_device(["CPU", ""], workers=2)


def test_GPU_0_is_valid() -> None:
  assert ProcessingConfig.validate_device("GPU:0", workers=1) == "GPU:0"


def test_GPU_1_is_valid() -> None:
  assert ProcessingConfig.validate_device("GPU:1", workers=1) == "GPU:1"


def test_CPU_0_is_valid() -> None:
  assert ProcessingConfig.validate_device("CPU:0", workers=1) == "CPU:0"


def test_CPU_1_is_valid() -> None:
  assert ProcessingConfig.validate_device("CPU:1", workers=1) == "CPU:1"
