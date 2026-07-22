"""``_validate_tf_backend_runtime`` friendly error when TensorFlow is absent.

On Python 3.11-3.13 TensorFlow is installed, so the real ``no_tf`` tests
(``test_issue55``) are skipped there. Monkeypatching ``tf_installed`` lets us
exercise the guard - and both interpreter-specific advice branches - on any
interpreter.
"""

from pathlib import Path

import pytest

import birdnet.model_loader as ml
from birdnet.model_loader import load, load_custom


@pytest.fixture
def _no_tf(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(ml, "tf_installed", lambda: False)


@pytest.mark.parametrize("backend", ["tf", "pb"])
def test_load_tf_backend_without_tensorflow_raises_actionable_error(
  backend: str, _no_tf: None
) -> None:
  with pytest.raises(ValueError) as exc_info:
    load("acoustic", "3.0", backend)

  message = str(exc_info.value)
  assert "requires TensorFlow" in message
  # The advice always points at the TensorFlow-free alternative.
  assert "onnx" in message


@pytest.mark.parametrize("backend", ["tf", "pb"])
def test_load_custom_tf_backend_without_tensorflow_raises_actionable_error(
  backend: str, _no_tf: None, tmp_path: Path
) -> None:
  species = tmp_path / "species.txt"
  species.write_text("a\n", encoding="utf-8")
  with pytest.raises(ValueError, match="requires TensorFlow"):
    load_custom("acoustic", "3.0", backend, tmp_path, species)


def test_non_tf_backend_is_not_blocked_by_the_guard(_no_tf: None) -> None:
  # The guard only fires for tf/pb; onnx must fall through to its own validation
  # (here: an unsupported precision) rather than complaining about TensorFlow.
  with pytest.raises(ValueError, match=r"Unsupported model precision"):
    load("acoustic", "3.0", "onnx", precision="int8")


def _fake_version_info(major: int, minor: int) -> tuple:
  # sys.version_info compares like a tuple *and* exposes .major/.minor; a
  # namedtuple gives both.
  from collections import namedtuple

  vi = namedtuple("version_info", "major minor micro releaselevel serial")
  return vi(major, minor, 0, "final", 0)


def test_message_recommends_reinstall_below_py314(
  monkeypatch: pytest.MonkeyPatch, _no_tf: None
) -> None:
  import sys

  monkeypatch.setattr(sys, "version_info", _fake_version_info(3, 12))
  with pytest.raises(ValueError, match=r"pip install tensorflow"):
    load("acoustic", "3.0", "tf")


def test_message_explains_missing_wheels_on_py314(
  monkeypatch: pytest.MonkeyPatch, _no_tf: None
) -> None:
  import sys

  monkeypatch.setattr(sys, "version_info", _fake_version_info(3, 14))
  with pytest.raises(ValueError, match=r"no wheels for Python 3\.14"):
    load("acoustic", "3.0", "tf")
