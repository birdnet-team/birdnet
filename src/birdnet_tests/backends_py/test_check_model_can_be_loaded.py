import sys
import time
from pathlib import Path

import pytest

from birdnet.core import backends
from birdnet.core.backends import BackendLoader
from birdnet_tests.fake_acoustic_backend import (
  FailingToLoadBackend,
  ForeignErrorBackend,
  NeverLoadingBackend,
  TimeoutErrorBackend,
  UnpicklableErrorBackend,
)

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


def _model(tmp_path: Path) -> Path:
  model = tmp_path / "model.tflite"
  model.write_bytes(b"not a model")
  return model


def test_load_error_is_raised_with_its_message_and_cause(tmp_path: Path) -> None:
  with pytest.raises(ValueError, match=r"^Failed to load model: bad weights$") as info:
    BackendLoader.check_model_can_be_loaded(_model(tmp_path), FailingToLoadBackend, {})
  assert isinstance(info.value.__cause__, RuntimeError)
  assert str(info.value.__cause__) == "bad weights"


def test_unpicklable_load_error_still_reports_type_and_message(tmp_path: Path) -> None:
  with pytest.raises(
    ValueError, match=r"^Failed to load model: RuntimeError: "
  ) as info:
    BackendLoader.check_model_can_be_loaded(
      _model(tmp_path), UnpicklableErrorBackend, {}
    )
  assert "bad weights" in str(info.value)


def test_foreign_load_error_does_not_import_its_module(tmp_path: Path) -> None:
  """A TensorFlow error would import TensorFlow into the caller on unpickling."""
  assert "birdnet_tests.foreign_load_error" not in sys.modules
  with pytest.raises(
    ValueError, match=r"^Failed to load model: ForeignLoadError: bad weights$"
  ):
    BackendLoader.check_model_can_be_loaded(_model(tmp_path), ForeignErrorBackend, {})
  assert "birdnet_tests.foreign_load_error" not in sys.modules


def test_a_timeout_error_from_the_model_is_not_the_check_timing_out(
  tmp_path: Path,
) -> None:
  with pytest.raises(
    ValueError, match=r"^Failed to load model: network share read timed out$"
  ):
    BackendLoader.check_model_can_be_loaded(_model(tmp_path), TimeoutErrorBackend, {})
