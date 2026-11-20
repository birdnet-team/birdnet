import pytest

from birdnet.model_loader import load
from birdnet_tests.helper import ensure_no_onnxruntime_gpu_provider_or_skip

pytestmark = pytest.mark.no_tf


@pytest.mark.filterwarnings("ignore:Specified provider:UserWarning")
def test_a_gpu_request_the_runtime_serves_on_the_cpu_is_an_error(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  ensure_no_onnxruntime_gpu_provider_or_skip()
  import onnxruntime

  model = load("geo", "3.0", "onnx", precision="fp32")
  # The session loads in this process, so the fake reaches it: the real
  # runtime has no CUDA provider and falls back to the CPU.
  monkeypatch.setattr(
    onnxruntime,
    "get_available_providers",
    lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
  )

  with pytest.raises(ValueError) as raised:
    model.predict(20, 50, week=1, device="GPU")

  message = str(raised.value)
  assert "asked for 'CUDAExecutionProvider'" in message, message
  assert "built the session on 'CPUExecutionProvider'" in message, message
