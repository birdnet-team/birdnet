"""``_validate_tf_backend_runtime`` friendly error when TensorFlow is absent.

On Python 3.11-3.13 TensorFlow is installed, so the real ``no_tf`` tests
(``test_issue55``) are skipped there. Monkeypatching ``tf_installed`` (and
``litert_installed``) lets us exercise the guard on any interpreter: both
interpreter-specific advice branches and the library-aware decision -- the
'tf' backend only needs TensorFlow for the default 'tflite' interpreter, not
for ``library="litert"``. Whether the litert import graph really is
TensorFlow-free is proven by the ``no_tf`` lane, not by these mocks.
"""

from pathlib import Path

import pytest
from ordered_set import OrderedSet

import birdnet.model_loader as ml
from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet.acoustic.models.v2_4.tf import AcousticTFBackendFP32V2_4
from birdnet.core.backends import TF_BACKEND_LIB_ARG
from birdnet.model_loader import load, load_custom


@pytest.fixture
def _no_tf(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(ml, "tf_installed", lambda: False)


@pytest.fixture
def _litert(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(ml, "litert_installed", lambda: True)


@pytest.fixture
def _no_litert(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(ml, "litert_installed", lambda: False)


@pytest.fixture
def _no_download(monkeypatch: pytest.MonkeyPatch) -> None:
  # Loading only resolves the model path; stub the downloader so the test needs
  # neither the model cache nor the network.
  monkeypatch.setattr(
    ml.AcousticTFDownloaderV2_4,
    "get_model_path_and_labels",
    lambda lang, precision: (Path("model.tflite"), OrderedSet(["species_a"])),
  )


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


# ----------------------------- library-aware decision ----------------------------


def test_tf_with_litert_library_loads_without_tensorflow(
  _no_tf: None, _litert: None, _no_download: None
) -> None:
  model = load("acoustic", "2.4", "tf", library="litert")

  assert isinstance(model, AcousticModelV2_4)
  assert model.backend_type is AcousticTFBackendFP32V2_4
  assert model.backend_kwargs == {TF_BACKEND_LIB_ARG: "litert"}


def test_load_custom_tf_with_litert_library_loads_without_tensorflow(
  _no_tf: None, _litert: None, tmp_path: Path
) -> None:
  model_file = tmp_path / "m.tflite"
  model_file.write_bytes(b"x")
  species = tmp_path / "species.txt"
  species.write_text("a\n", encoding="utf-8")

  model = load_custom(
    "acoustic", "2.4", "tf", model_file, species, library="litert", check_validity=False
  )

  assert isinstance(model, AcousticModelV2_4)
  assert model.backend_kwargs == {TF_BACKEND_LIB_ARG: "litert"}


def test_tf_with_explicit_tflite_library_still_requires_tensorflow(
  _no_tf: None, _litert: None
) -> None:
  with pytest.raises(
    ValueError, match=r"Backend 'tf' with library 'tflite' \(the default\) requires"
  ):
    load("acoustic", "2.4", "tf", library="tflite")


def test_pb_still_requires_tensorflow_even_with_litert_kwarg(
  _no_tf: None, _litert: None
) -> None:
  # 'pb' has no interpreter choice; the kwarg is rejected later, TensorFlow first.
  with pytest.raises(ValueError, match=r"Backend 'pb' requires TensorFlow"):
    load("acoustic", "2.4", "pb", library="litert")


def test_invalid_library_is_reported_as_such_without_tensorflow(_no_tf: None) -> None:
  # An unknown library must not be mistaken for the default and blamed on TensorFlow.
  with pytest.raises(ValueError, match=r"Unsupported TensorFlow library: zzz"):
    load("acoustic", "2.4", "tf", library="zzz")


def test_litert_library_without_litert_names_the_missing_package(
  _no_tf: None, _no_litert: None
) -> None:
  # The caller asked for litert, so the missing runtime is litert, not TensorFlow.
  with pytest.raises(ValueError, match=r"Parameter 'library'.*ai-edge-litert"):
    load("acoustic", "2.4", "tf", library="litert")


def test_validate_library_tflite_without_tensorflow_raises_actionable_error(
  _no_tf: None,
) -> None:
  # Safety net behind the guard for internal callers.
  with pytest.raises(ValueError, match=r"requires TensorFlow"):
    ml._validate_library("tflite")


# ----------------------------- advice in the message -----------------------------


def test_tf_message_offers_litert_when_installed(_no_tf: None, _litert: None) -> None:
  with pytest.raises(ValueError) as exc_info:
    load("acoustic", "2.4", "tf")

  message = str(exc_info.value)
  assert "pass library='litert'" in message
  assert "onnx" in message


def test_pb_message_offers_tf_backend_with_litert_when_installed(
  _no_tf: None, _litert: None
) -> None:
  with pytest.raises(ValueError, match=r"'tf' backend with library='litert'"):
    load("acoustic", "2.4", "pb")


def test_message_suggests_installing_litert_when_not_installed(
  _no_tf: None, _no_litert: None
) -> None:
  with pytest.raises(ValueError) as exc_info:
    load("acoustic", "2.4", "tf")

  message = str(exc_info.value)
  assert "pip install ai-edge-litert" in message
  assert "which is installed" not in message
  assert "onnx" in message


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
