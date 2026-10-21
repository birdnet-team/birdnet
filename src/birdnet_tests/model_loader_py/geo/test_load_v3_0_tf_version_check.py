import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

import birdnet.geo.models.v3_0.tf as geo_tf
from birdnet.core.backends import tf_installed
from birdnet.geo.models.v3_0.model import GeoModelV3_0
from birdnet.geo.models.v3_0.tf import GeoTFDownloaderV3_0
from birdnet.model_loader import load, load_custom
from birdnet_tests.helper import ensure_tf_2_19_or_2_18


class _PastValidation(Exception):
  """Raised by the first step after argument validation: download or probe."""


def _raise_past_validation(*args: object, **kwargs: object) -> None:
  raise _PastValidation


@pytest.fixture(autouse=True)
def _stop_after_validation(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(
    GeoTFDownloaderV3_0, "get_model_path_and_labels", _raise_past_validation
  )
  monkeypatch.setattr(GeoModelV3_0, "load_custom", _raise_past_validation)


def _require_tf() -> None:
  if not tf_installed():
    pytest.skip("TensorFlow is not installed")


def _set_tf_version(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
  _require_tf()
  monkeypatch.setattr(geo_tf, "_installed_tf_version", lambda: version)


def _load(tmp_path: Path) -> None:
  load("geo", "3.0", "tf", precision="fp32", library="tflite")


def _load_custom(tmp_path: Path) -> None:
  model = tmp_path / "model.tflite"
  model.write_bytes(b"")
  species = tmp_path / "species.txt"
  species.write_bytes(b"a\n")
  load_custom("geo", "3.0", "tf", model, species, precision="fp32", library="tflite")


ENTRIES = pytest.mark.parametrize(
  "entry", [_load, _load_custom], ids=["load", "load_custom"]
)


@ENTRIES
@pytest.mark.parametrize("version", ["2.21.0", "2.180.0", "2.17.1"])
def test_unsupported_tf_is_rejected_before_download(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  entry: Callable[[Path], None],
  version: str,
) -> None:
  _set_tf_version(monkeypatch, version)
  with pytest.raises(RuntimeError, match=r"requires TensorFlow 2\.18 or 2\.19"):
    entry(tmp_path)


@ENTRIES
@pytest.mark.parametrize("version", ["2.18.1", "2.19.0"])
def test_supported_tf_passes_the_check(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  entry: Callable[[Path], None],
  version: str,
) -> None:
  _set_tf_version(monkeypatch, version)
  with pytest.raises(_PastValidation):
    entry(tmp_path)


@pytest.mark.parametrize(
  ("version", "supported"),
  [("2.18.1", True), ("2.19.0", True), ("2.180.0", False), ("2.21.0", False)],
)
def test_the_tests_skip_exactly_where_the_backend_refuses(
  monkeypatch: pytest.MonkeyPatch, version: str, supported: bool
) -> None:
  _set_tf_version(monkeypatch, version)
  if supported:
    ensure_tf_2_19_or_2_18()
  else:
    with pytest.raises(pytest.skip.Exception):
      ensure_tf_2_19_or_2_18()


_IMPORT_PROBE = """
import sys
from pathlib import Path

import birdnet.geo.models.v3_0.model as geo_model
import birdnet.geo.models.v3_0.tf as geo_tf
from birdnet.model_loader import load, load_custom


def past_validation(*args, **kwargs):
  raise LookupError("past validation")


if __name__ == "__main__":
  # A supported TensorFlow gets past the check; nothing may download either way.
  geo_tf.GeoTFDownloaderV3_0.get_model_path_and_labels = past_validation
  geo_model.GeoModelV3_0.load_custom = past_validation
  tmp = Path(sys.argv[1])
  (tmp / "model.tflite").write_bytes(b"")
  (tmp / "species.txt").write_bytes(b"a")
  for name, call in [
    ("load", lambda: load("geo", "3.0", "tf", precision="fp32", library="tflite")),
    ("load_custom", lambda: load_custom(
      "geo", "3.0", "tf", tmp / "model.tflite", tmp / "species.txt",
      precision="fp32", library="tflite",
    )),
  ]:
    try:
      call()
    except (RuntimeError, LookupError) as e:
      print(name, type(e).__name__, "tensorflow" in sys.modules)
"""


def test_the_check_does_not_import_tensorflow(tmp_path: Path) -> None:
  """The caller's process stays free of TensorFlow until a model is used."""
  _require_tf()
  script = tmp_path / "probe.py"
  script.write_text(_IMPORT_PROBE, encoding="utf-8")
  out = subprocess.run(
    [sys.executable, str(script), str(tmp_path)],
    capture_output=True,
    text=True,
    timeout=300,
  )
  assert out.returncode == 0, out.stderr
  lines = [line.split() for line in out.stdout.splitlines()]
  assert [line[0] for line in lines] == ["load", "load_custom"]
  assert all(line[2] == "False" for line in lines), out.stdout
