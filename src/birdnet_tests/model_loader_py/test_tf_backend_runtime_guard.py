"""``_validate_tf_backend_runtime`` friendly error when TensorFlow is absent.

The real ``no_tf`` tests (``test_issue55``) only run where TensorFlow is not
installed. Monkeypatching ``tf_installed`` (and ``litert_installed``) lets us
exercise the guard in any environment: both interpreter-specific advice
branches and the library-aware decision -- the 'tf' backend only needs
TensorFlow for the default 'tflite' interpreter, not for ``library="litert"``.
Whether the litert import graph really is TensorFlow-free is proven by the
TensorFlow-free lanes, not by these mocks.
"""

from pathlib import Path

import pytest
from ordered_set import OrderedSet

import birdnet.model_loader as ml
from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet.acoustic.models.v2_4.tf import AcousticTFBackendFP32V2_4
from birdnet.core.backends import TF_BACKEND_LIB_ARG
from birdnet.model_loader import load, load_custom
from birdnet_tests.helper import ensure_not_intel_macos_or_skip, fake_wheels

pytestmark = pytest.mark.no_tf  # everything is mocked; runs with or without TensorFlow


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
def _onnxruntime(monkeypatch: pytest.MonkeyPatch) -> None:
  # Otherwise the 'onnx' advice depends on the interpreter.
  monkeypatch.setattr(ml, "onnxruntime_installed", lambda: True)


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
  backend: str, _no_tf: None, _onnxruntime: None
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


def test_perch_without_tensorflow_names_the_model(_no_tf: None) -> None:
  ensure_not_intel_macos_or_skip()  # rejected there before any TensorFlow check
  with pytest.raises(ValueError, match=r"^The Perch V2 model requires TensorFlow"):
    ml.load_perch_v2("CPU")


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


class _ModelCheckRan(Exception):
  pass


def _model_check(library: str) -> None:
  raise _ModelCheckRan


def test_validate_library_checks_tensorflow_before_the_model_check(
  _no_tf: None,
) -> None:
  # The model's check may import TensorFlow, which then fails without a hint.
  with pytest.raises(ValueError, match=r"requires TensorFlow"):
    ml._validate_library("tflite", compatibility_check=_model_check)


def test_validate_library_runs_the_model_check_before_the_litert_check(
  _no_litert: None,
) -> None:
  # A model that rejects litert must say so, not ask for litert to be installed.
  with pytest.raises(_ModelCheckRan):
    ml._validate_library("litert", compatibility_check=_model_check)


# ----------------------------- advice in the message -----------------------------


def test_tf_message_offers_litert_when_installed(
  _no_tf: None, _litert: None, _onnxruntime: None
) -> None:
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
  monkeypatch: pytest.MonkeyPatch, _no_tf: None, _no_litert: None, _onnxruntime: None
) -> None:
  fake_wheels(monkeypatch, 13)
  with pytest.raises(ValueError) as exc_info:
    load("acoustic", "2.4", "tf")

  message = str(exc_info.value)
  assert "pip install ai-edge-litert" in message
  assert "which is installed" not in message
  assert "onnx" in message


def test_message_recommends_the_tf_extra_where_tensorflow_has_wheels(
  monkeypatch: pytest.MonkeyPatch, _no_tf: None
) -> None:
  fake_wheels(monkeypatch, 20)
  with pytest.raises(ValueError, match=r"pip install birdnet\[tf\]"):
    load("acoustic", "3.0", "tf")


def test_message_explains_missing_tensorflow_wheels(
  monkeypatch: pytest.MonkeyPatch, _no_tf: None
) -> None:
  fake_wheels(monkeypatch, 20, {"tensorflow": range(11, 20)})
  with pytest.raises(ValueError, match=r"no wheels for Python 3\.20 when"):
    load("acoustic", "3.0", "tf")


# --------------------- advice where runtimes have no wheels ----------------------
# Python 3.20 stands for an interpreter without wheels; the table is faked.


def _tf_error() -> str:
  with pytest.raises(ValueError) as exc_info:
    load("acoustic", "3.0", "tf")
  return str(exc_info.value)


def test_tf_message_offers_pt_where_onnxruntime_cannot_be_installed(
  monkeypatch: pytest.MonkeyPatch, _no_tf: None
) -> None:
  fake_wheels(monkeypatch, 20, {"onnxruntime": range(11, 20)})
  monkeypatch.setattr(ml, "onnxruntime_installed", lambda: False)
  monkeypatch.setattr(ml, "torch_installed", lambda: False)
  message = _tf_error()

  assert (
    "use the 'pt' backend ('pip install birdnet[pt]'), available for the acoustic "
    "3.0 and geo 3.0 models"
  ) in message
  assert "'onnx' backend" not in message


@pytest.mark.parametrize(
  ("onnx_wheels", "onnxruntime_installed"),
  [(range(11, 30), False), (range(11, 20), True)],
)
def test_tf_message_offers_onnx_where_onnxruntime_is_available(
  monkeypatch: pytest.MonkeyPatch,
  _no_tf: None,
  onnx_wheels: range,
  onnxruntime_installed: bool,
) -> None:
  fake_wheels(monkeypatch, 20, {"onnxruntime": onnx_wheels})
  monkeypatch.setattr(ml, "onnxruntime_installed", lambda: onnxruntime_installed)

  assert "use the 'onnx' backend" in _tf_error()


@pytest.mark.parametrize(
  ("torch_wheels", "mentions_pt"), [(range(11, 30), True), (range(11, 20), False)]
)
def test_tf_message_mentions_pt_beside_onnx_only_where_it_can_run(
  monkeypatch: pytest.MonkeyPatch,
  _no_tf: None,
  _onnxruntime: None,
  torch_wheels: range,
  mentions_pt: bool,
) -> None:
  fake_wheels(monkeypatch, 20, {"torch": torch_wheels})
  monkeypatch.setattr(ml, "torch_installed", lambda: False)

  assert ("(both also offer 'pt')" in _tf_error()) is mentions_pt


def test_tf_message_does_not_suggest_litert_where_it_has_no_wheels(
  monkeypatch: pytest.MonkeyPatch, _no_tf: None, _no_litert: None
) -> None:
  fake_wheels(monkeypatch, 20, {"ai-edge-litert": range(11, 20)})
  message = _tf_error()

  assert "ai-edge-litert" not in message
  assert "library='litert'" not in message


def test_tf_message_offers_no_alternative_where_none_installs(
  monkeypatch: pytest.MonkeyPatch, _no_tf: None, _no_litert: None
) -> None:
  fake_wheels(
    monkeypatch,
    20,
    {
      "tensorflow": {"macos-x86_64": range(11, 13)},
      "onnxruntime": {"macos-x86_64": range(11, 14)},
      "ai-edge-litert": {"macos-x86_64": range(0)},
      "torch": {"macos-x86_64": range(11, 13)},
    },
    "macos-x86_64",
  )
  monkeypatch.setattr(ml, "onnxruntime_installed", lambda: False)
  monkeypatch.setattr(ml, "torch_installed", lambda: False)
  message = _tf_error()

  assert "TensorFlow had no wheels for Python 3.20 on macOS x86_64 when" in message
  assert "; use Python 3.11-3.12 to run TensorFlow-based models." in message
  assert "Alternatively" not in message


def test_litert_message_explains_missing_wheels(
  monkeypatch: pytest.MonkeyPatch, _no_tf: None, _no_litert: None
) -> None:
  fake_wheels(monkeypatch, 20, {"ai-edge-litert": range(11, 20)})
  with pytest.raises(ValueError) as exc_info:
    load("acoustic", "2.4", "tf", library="litert")

  message = str(exc_info.value)
  assert (
    "requires ai-edge-litert, which is not installed, and it had no wheels for "
    "Python 3.20 when this birdnet version was released"
  ) in message
  assert message.endswith("; use Python 3.11-3.19.")
  assert "pip install ai-edge-litert" not in message


def test_litert_message_points_at_tflite_where_litert_has_no_wheels(
  monkeypatch: pytest.MonkeyPatch, _no_litert: None
) -> None:
  fake_wheels(
    monkeypatch, 12, {"ai-edge-litert": {"macos-x86_64": range(0)}}, "macos-x86_64"
  )
  monkeypatch.setattr(ml, "tf_installed", lambda: True)
  with pytest.raises(ValueError) as exc_info:
    ml._validate_library("litert")

  message = str(exc_info.value)
  assert "it had no wheels for macOS x86_64 when" in message
  assert message.endswith(
    "; use the default library 'tflite', which runs on the installed TensorFlow."
  )


@pytest.mark.parametrize(
  ("tf_installed", "ending"),
  [
    (
      True,
      "Install it ('pip install ai-edge-litert') or use the default library "
      "'tflite', which runs on the installed TensorFlow.",
    ),
    (False, "Install it ('pip install ai-edge-litert')."),
  ],
)
def test_litert_message_offers_tflite_only_beside_an_installed_tensorflow(
  monkeypatch: pytest.MonkeyPatch, _no_litert: None, tf_installed: bool, ending: str
) -> None:
  fake_wheels(monkeypatch, 20)
  monkeypatch.setattr(ml, "tf_installed", lambda: tf_installed)
  with pytest.raises(ValueError) as exc_info:
    ml._validate_library("litert")

  assert str(exc_info.value).endswith(ending)
