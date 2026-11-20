"""A GPU request whose session ONNX Runtime builds on the CPU is an error.

ONNX Runtime drops a GPU provider it cannot create and carries on with the
CPU provider, so the loader compares what it asked for with what it got.
"""

from pathlib import Path

import pytest

from birdnet.core.backends import load_onnx_model
from birdnet_tests.helper import (
  ensure_no_onnxruntime_gpu_provider_or_skip,
  ensure_onnxruntime_or_skip,
)

pytestmark = pytest.mark.no_tf

_GPU_PROVIDERS = ["CUDAExecutionProvider", "DmlExecutionProvider"]


def _example_model() -> Path:
  from onnxruntime.datasets import get_example

  return Path(get_example("sigmoid.onnx"))


def _runtime_version() -> str:
  import onnxruntime

  return onnxruntime.__version__


def _fake_available(monkeypatch: pytest.MonkeyPatch, gpu_provider: str) -> None:
  import onnxruntime

  monkeypatch.setattr(
    onnxruntime,
    "get_available_providers",
    lambda: [gpu_provider, "CPUExecutionProvider"],
  )


class _FakeSession:
  """Reports the providers its class is told to, whatever was requested."""

  reported: list[str] | None = None

  def __init__(
    self,
    path: str,
    sess_options: object,
    providers: list[str],
    provider_options: list[dict[str, str]],
  ) -> None:
    self._providers = providers if self.reported is None else self.reported

  def get_providers(self) -> list[str]:
    return self._providers


def _fake_session(
  monkeypatch: pytest.MonkeyPatch, reported: list[str] | None
) -> type[_FakeSession]:
  import onnxruntime

  fake = type("_FakeSessionOfThisTest", (_FakeSession,), {"reported": reported})
  monkeypatch.setattr(onnxruntime, "InferenceSession", fake)
  return fake


# The runtime's notice that it drops the provider.
@pytest.mark.filterwarnings("ignore:Specified provider:UserWarning")
@pytest.mark.parametrize("gpu_provider", _GPU_PROVIDERS)
def test_a_gpu_provider_the_runtime_cannot_create_is_an_error(
  monkeypatch: pytest.MonkeyPatch, gpu_provider: str
) -> None:
  # The real runtime, which has no such provider and falls back to the CPU.
  ensure_no_onnxruntime_gpu_provider_or_skip()
  _fake_available(monkeypatch, gpu_provider)

  with pytest.raises(ValueError) as raised:
    load_onnx_model(_example_model(), "GPU")

  message = str(raised.value)
  asked = f"ONNX Runtime {_runtime_version()} was asked for '{gpu_provider}'"
  assert asked in message, message
  assert "built the session on 'CPUExecutionProvider'" in message, message
  assert "device='CPU'" in message, message
  # DirectML needs neither.
  wants_cuda_hint = gpu_provider == "CUDAExecutionProvider"
  assert ("CUDA and cuDNN" in message) == wants_cuda_hint, message


@pytest.mark.parametrize("gpu_provider", _GPU_PROVIDERS)
def test_a_session_on_the_requested_gpu_provider_is_returned(
  monkeypatch: pytest.MonkeyPatch, gpu_provider: str
) -> None:
  ensure_onnxruntime_or_skip()
  _fake_available(monkeypatch, gpu_provider)
  fake = _fake_session(monkeypatch, reported=None)

  session = load_onnx_model(_example_model(), "GPU")

  assert isinstance(session, fake)
  assert session.get_providers() == [gpu_provider, "CPUExecutionProvider"]


def test_a_session_that_reports_the_cpu_only_is_an_error_naming_the_device_index(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # What the runtime reports after a provider failed while it was created.
  ensure_onnxruntime_or_skip()
  _fake_available(monkeypatch, "CUDAExecutionProvider")
  _fake_session(monkeypatch, reported=["CPUExecutionProvider"])

  with pytest.raises(ValueError) as raised:
    load_onnx_model(_example_model(), "GPU:7")

  message = str(raised.value)
  assert "asked for 'CUDAExecutionProvider' (device 'GPU:7')" in message, message
  assert "built the session on 'CPUExecutionProvider'" in message, message
  assert "CUDA and cuDNN" in message, message
  assert "the device index 7 exists" in message, message
  assert "device='CPU'" in message, message


def test_a_session_on_any_provider_but_the_requested_one_is_an_error(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # The runtime's only substitute is the CPU provider; the loader accepts none.
  ensure_onnxruntime_or_skip()
  _fake_available(monkeypatch, "CUDAExecutionProvider")
  _fake_session(
    monkeypatch, reported=["OtherExecutionProvider", "CPUExecutionProvider"]
  )

  with pytest.raises(ValueError) as raised:
    load_onnx_model(_example_model(), "GPU")

  message = str(raised.value)
  assert "asked for 'CUDAExecutionProvider'" in message, message
  assert "built the session on 'OtherExecutionProvider'" in message, message
