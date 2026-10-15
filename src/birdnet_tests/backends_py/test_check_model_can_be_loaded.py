import time
from pathlib import Path

import pytest

from birdnet.core import backends
from birdnet.core.backends import BackendLoader
from birdnet_tests.fake_acoustic_backend import NeverLoadingBackend

pytestmark = pytest.mark.no_tf


def test_gives_up_at_its_timeout(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A model that hangs while opening must fail the check at the timeout, not
  when the hang ends."""
  monkeypatch.setattr(backends, "_SUBPROCESS_TIMEOUT_S", 2)
  model = tmp_path / "model.tflite"
  model.write_bytes(b"not a model")

  started = time.monotonic()
  with pytest.raises(ValueError, match=r"Failed to load model due to timeout"):
    BackendLoader.check_model_can_be_loaded(model, NeverLoadingBackend, {})
  assert time.monotonic() - started < 30
