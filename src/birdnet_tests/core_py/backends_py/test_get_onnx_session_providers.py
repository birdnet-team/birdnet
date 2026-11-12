"""The onnx backend's provider lookup, with the available providers faked.

The gpu lane's onnx case decides through this lookup whether it runs, and CI
has no GPU, so nothing else reaches its GPU branches there.
"""

import pytest

from birdnet.core.backends import _get_onnx_session_providers
from birdnet_tests.helper import (
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
