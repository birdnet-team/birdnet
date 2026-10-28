"""Advice when the onnx or pt runtime is missing.

Python 3.20 stands for an interpreter without wheels; the table is faked.
"""

import pytest

import birdnet.model_loader as ml
from birdnet.model_loader import load
from birdnet_tests.helper import fake_wheels

pytestmark = pytest.mark.no_tf  # everything is mocked; runs with or without TensorFlow


@pytest.fixture
def _no_onnxruntime(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(ml, "onnxruntime_installed", lambda: False)


@pytest.fixture
def _no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(ml, "torch_installed", lambda: False)


def _load_error(backend: str) -> str:
  with pytest.raises(ValueError) as exc_info:
    load("acoustic", "3.0", backend)  # type: ignore[call-overload]
  return str(exc_info.value)


def test_onnx_message_recommends_installing_onnxruntime_where_it_has_wheels(
  monkeypatch: pytest.MonkeyPatch, _no_onnxruntime: None
) -> None:
  fake_wheels(monkeypatch, 20)
  message = _load_error("onnx")

  assert "onnxruntime is not installed ('pip install onnxruntime'" in message


def test_onnx_message_explains_missing_wheels(
  monkeypatch: pytest.MonkeyPatch, _no_onnxruntime: None, _no_torch: None
) -> None:
  fake_wheels(monkeypatch, 20, {"onnxruntime": range(11, 20)})
  message = _load_error("onnx")

  assert (
    "onnxruntime is not installed and it had no wheels for Python 3.20 when this "
    "birdnet version was released;"
  ) in message
  assert message.endswith(
    "use the 'pt' backend ('pip install birdnet[pt]') or Python 3.11-3.19."
  )
  assert "pip install onnxruntime" not in message


def test_onnx_message_does_not_ask_to_install_an_installed_torch(
  monkeypatch: pytest.MonkeyPatch, _no_onnxruntime: None
) -> None:
  fake_wheels(monkeypatch, 20, {"onnxruntime": range(11, 20)})
  monkeypatch.setattr(ml, "torch_installed", lambda: True)

  assert _load_error("onnx").endswith("use the 'pt' backend or Python 3.11-3.19.")


def test_onnx_message_names_the_platform_where_it_is_the_reason(
  monkeypatch: pytest.MonkeyPatch, _no_onnxruntime: None, _no_torch: None
) -> None:
  fake_wheels(
    monkeypatch,
    20,
    {
      "onnxruntime": {"macos-x86_64": range(11, 20)},
      "torch": {"macos-x86_64": range(11, 15)},
    },
    "macos-x86_64",
  )
  message = _load_error("onnx")

  assert "had no wheels for Python 3.20 on macOS x86_64 when" in message
  assert message.endswith("; use Python 3.11-3.19.")
  assert "'pt'" not in message


def test_pt_message_recommends_the_extra_where_torch_has_wheels(
  monkeypatch: pytest.MonkeyPatch, _no_torch: None
) -> None:
  fake_wheels(monkeypatch, 20)

  assert _load_error("pt").endswith("Install birdnet with [pt] option.")


def test_pt_message_explains_missing_wheels(
  monkeypatch: pytest.MonkeyPatch, _no_torch: None, _no_onnxruntime: None
) -> None:
  fake_wheels(
    monkeypatch,
    20,
    {"torch": range(11, 20), "onnxruntime": range(11, 20)},
  )
  message = _load_error("pt")

  assert "PyTorch had no wheels for Python 3.20 when this birdnet version" in message
  assert message.endswith("; use Python 3.11-3.19.")
  assert "Install birdnet with [pt]" not in message


@pytest.mark.parametrize(
  ("onnxruntime_installed", "onnx_wheels", "offered"),
  [
    (True, range(11, 20), "the 'onnx' backend or"),
    (False, range(11, 30), "the 'onnx' backend ('pip install onnxruntime') or"),
  ],
)
def test_pt_message_offers_onnx_where_it_can_run(
  monkeypatch: pytest.MonkeyPatch,
  _no_torch: None,
  onnxruntime_installed: bool,
  onnx_wheels: range,
  offered: str,
) -> None:
  fake_wheels(monkeypatch, 20, {"torch": range(11, 20), "onnxruntime": onnx_wheels})
  monkeypatch.setattr(ml, "onnxruntime_installed", lambda: onnxruntime_installed)

  assert _load_error("pt").endswith(f"; use {offered} Python 3.11-3.19.")
