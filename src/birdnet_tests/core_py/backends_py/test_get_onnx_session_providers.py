"""The onnx backend's provider lookup, with the available providers faked.

The gpu lane's onnx case decides through this lookup whether it runs, and CI
has no GPU, so nothing else reaches its GPU branches there.
"""

import pytest

from birdnet.core import backends
from birdnet.core.backends import _get_onnx_session_providers
from birdnet_tests.helper import (
  ensure_no_onnxruntime_gpu_provider_or_skip,
  ensure_onnxruntime_gpu_or_skip,
  ensure_onnxruntime_or_skip,
)

pytestmark = pytest.mark.no_tf

_NO_GPU = ["AzureExecutionProvider", "CPUExecutionProvider"]
_CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]
_DML = ["DmlExecutionProvider", "CPUExecutionProvider"]
_NO_GPU_MESSAGE = "No ONNX Runtime GPU execution provider found"


def _fake_available(monkeypatch: pytest.MonkeyPatch, providers: list[str]) -> None:
  import onnxruntime

  monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: providers)


@pytest.mark.parametrize("available", [_NO_GPU, _CUDA, _DML])
def test_cpu_gets_the_cpu_provider_whatever_is_available(
  monkeypatch: pytest.MonkeyPatch, available: list[str]
) -> None:
  ensure_onnxruntime_or_skip()
  _fake_available(monkeypatch, available)

  assert _get_onnx_session_providers("CPU") == (["CPUExecutionProvider"], [{}])


def test_gpu_without_a_gpu_provider_raises(monkeypatch: pytest.MonkeyPatch) -> None:
  ensure_onnxruntime_or_skip()
  _fake_available(monkeypatch, _NO_GPU)

  with pytest.raises(ValueError, match=_NO_GPU_MESSAGE):
    _get_onnx_session_providers("GPU")


def test_gpu_with_both_builds_installed_names_them_and_the_repair(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  ensure_onnxruntime_or_skip()
  import onnxruntime

  _fake_available(monkeypatch, _NO_GPU)
  monkeypatch.setattr(onnxruntime, "__version__", "1.32.0")
  monkeypatch.setattr(
    backends,
    "_onnxruntime_builds",
    lambda: [("onnxruntime", "1.32.0"), ("onnxruntime-gpu", "1.26.0")],
  )

  with pytest.raises(ValueError) as exc_info:
    _get_onnx_session_providers("GPU")

  assert str(exc_info.value) == (
    f"{_NO_GPU_MESSAGE}: onnxruntime 1.32.0 and onnxruntime-gpu 1.26.0 are both "
    "installed and share the folder 'onnxruntime', where the build installed last, "
    "ONNX Runtime 1.32.0, replaced the files of the other and lists no GPU "
    "provider. Reinstall the GPU build alone: 'pip uninstall -y onnxruntime "
    "onnxruntime-gpu && pip install onnxruntime-gpu==1.26.0'."
  )


@pytest.mark.parametrize(
  "builds",
  [
    [("onnxruntime", "1.32.0")],
    [("onnxruntime", "1.32.0"), ("onnxruntime-openvino", "1.24.0")],
  ],
)
def test_gpu_without_a_gpu_build_installed_asks_for_a_gpu_build(
  monkeypatch: pytest.MonkeyPatch, builds: list[tuple[str, str]]
) -> None:
  ensure_onnxruntime_or_skip()
  _fake_available(monkeypatch, _NO_GPU)
  monkeypatch.setattr(backends, "_onnxruntime_builds", lambda: builds)

  with pytest.raises(ValueError) as exc_info:
    _get_onnx_session_providers("GPU")

  assert str(exc_info.value) == (
    f"{_NO_GPU_MESSAGE}. Install an ONNX Runtime build with GPU support."
  )


def test_gpu_with_the_directml_build_installed_next_to_the_cpu_build_names_both(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  ensure_onnxruntime_or_skip()
  _fake_available(monkeypatch, _NO_GPU)
  monkeypatch.setattr(
    backends,
    "_onnxruntime_builds",
    lambda: [("onnxruntime", "1.32.0"), ("onnxruntime-directml", "1.24.0")],
  )

  with pytest.raises(ValueError) as exc_info:
    _get_onnx_session_providers("GPU")

  assert str(exc_info.value).endswith(
    "'pip uninstall -y onnxruntime onnxruntime-directml onnxruntime-gpu && "
    "pip install onnxruntime-directml==1.24.0'."
  )


@pytest.mark.parametrize(("device", "device_id"), [("GPU", "0"), ("GPU:1", "1")])
def test_gpu_gets_cuda_with_the_device_index(
  monkeypatch: pytest.MonkeyPatch, device: str, device_id: str
) -> None:
  ensure_onnxruntime_or_skip()
  _fake_available(monkeypatch, _CUDA)

  assert _get_onnx_session_providers(device) == (
    ["CUDAExecutionProvider", "CPUExecutionProvider"],
    [{"device_id": device_id}, {}],
  )


@pytest.mark.parametrize(("device", "device_id"), [("GPU", "0"), ("GPU:1", "1")])
def test_gpu_gets_dml_where_cuda_is_absent(
  monkeypatch: pytest.MonkeyPatch, device: str, device_id: str
) -> None:
  ensure_onnxruntime_or_skip()
  _fake_available(monkeypatch, _DML)

  assert _get_onnx_session_providers(device) == (
    ["DmlExecutionProvider", "CPUExecutionProvider"],
    [{"device_id": device_id}, {}],
  )


def test_cuda_is_chosen_over_dml(monkeypatch: pytest.MonkeyPatch) -> None:
  ensure_onnxruntime_or_skip()
  _fake_available(monkeypatch, ["DmlExecutionProvider", *_CUDA])

  providers, _ = _get_onnx_session_providers("GPU")

  assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_the_skip_helper_skips_with_the_lookups_message(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  ensure_onnxruntime_or_skip()
  _fake_available(monkeypatch, _NO_GPU)

  with pytest.raises(pytest.skip.Exception, match=_NO_GPU_MESSAGE):
    ensure_onnxruntime_gpu_or_skip()


def test_the_skip_helper_returns_where_a_gpu_provider_is_listed(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  ensure_onnxruntime_or_skip()
  _fake_available(monkeypatch, _CUDA)

  # Caught here: a skip that escaped would report this test as skipped, not failed.
  try:
    ensure_onnxruntime_gpu_or_skip()
  except pytest.skip.Exception as e:
    pytest.fail(f"skipped although a GPU provider is listed: {e}")


@pytest.mark.parametrize("available", [_CUDA, _DML])
def test_the_no_gpu_provider_helper_skips_where_one_is_listed(
  monkeypatch: pytest.MonkeyPatch, available: list[str]
) -> None:
  ensure_onnxruntime_or_skip()
  _fake_available(monkeypatch, available)

  with pytest.raises(pytest.skip.Exception, match=available[0]):
    ensure_no_onnxruntime_gpu_provider_or_skip()


def test_the_no_gpu_provider_helper_returns_where_none_is_listed(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  ensure_onnxruntime_or_skip()
  _fake_available(monkeypatch, _NO_GPU)

  # Caught here: a skip that escaped would report this test as skipped, not failed.
  try:
    ensure_no_onnxruntime_gpu_provider_or_skip()
  except pytest.skip.Exception as e:
    pytest.fail(f"skipped although no GPU provider is listed: {e}")
