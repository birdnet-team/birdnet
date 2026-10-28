from pathlib import Path

import pytest

from birdnet.core.backends import BackendLoader

pytestmark = pytest.mark.no_tf


def test_a_failed_inspection_raises_without_printing_it_too(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """The package logger has no handler, so a log call lands on stderr."""
  model = tmp_path / "model.tflite"
  model.write_bytes(b"not a model")
  with pytest.raises(ValueError, match=r"Failed to inspect custom TFLite model"):
    BackendLoader.inspect_custom_tflite_model(model, "tflite", ())
  assert "Failed to inspect" not in capsys.readouterr().err
