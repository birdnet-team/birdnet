"""The pt backend's device lookup, with torch's answer about CUDA faked.

The gpu lane's pt cases decide through this lookup whether they run, and CI
has no GPU, so nothing else reaches its GPU branch there.
"""

import pytest

from birdnet_tests.helper import ensure_torch_gpu_or_skip, ensure_torch_or_skip

pytestmark = pytest.mark.no_tf

_NO_GPU_MESSAGE = "No GPU found"


def _fake_cuda(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
  import torch

  monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
  monkeypatch.setattr(torch.cuda, "device_count", lambda: int(available))


def test_the_skip_helper_skips_with_the_lookups_message(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  ensure_torch_or_skip()
  _fake_cuda(monkeypatch, available=False)

  with pytest.raises(pytest.skip.Exception, match=_NO_GPU_MESSAGE):
    ensure_torch_gpu_or_skip()


def test_the_skip_helper_returns_where_cuda_is_available(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  ensure_torch_or_skip()
  _fake_cuda(monkeypatch, available=True)

  # Caught here: a skip that escaped would report this test as skipped, not failed.
  try:
    ensure_torch_gpu_or_skip()
  except pytest.skip.Exception as e:
    pytest.fail(f"skipped although CUDA is available: {e}")
