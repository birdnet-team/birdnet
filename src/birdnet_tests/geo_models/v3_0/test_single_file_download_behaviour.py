"""Where the single-file geo V3.0 downloaders look, write and fetch.

The expected paths and the download description are spelled out here rather than
derived from get_model_path or get_lang_dir, so a change to where a model lands
cannot pass by agreeing with itself. Only the checksum and the byte size come from
the downloaders' tables.
"""

import dataclasses
import inspect
import sys
from pathlib import Path

import pytest

from birdnet.geo.models.v3_0.model import GeoModelV3_0
from birdnet.geo.models.v3_0.onnx import GeoOnnxDownloaderV3_0
from birdnet.geo.models.v3_0.onnx import models as onnx_models
from birdnet.geo.models.v3_0.pt import GeoPTDownloaderV3_0
from birdnet.geo.models.v3_0.pt import models as pt_models
from birdnet.geo.models.v3_0.tf import GeoTFDownloaderV3_0
from birdnet.geo.models.v3_0.tf import models as tf_models
from birdnet.globals import MODEL_PRECISIONS
from birdnet.utils import downloads
from birdnet.utils.downloads import ModelInfo
from birdnet_tests.helper import fake_labels

_Downloader = (
  type[GeoTFDownloaderV3_0] | type[GeoOnnxDownloaderV3_0] | type[GeoPTDownloaderV3_0]
)

_SMALL_FILE_SIZE = 64
_EN_US_LABELS = fake_labels(GeoModelV3_0.get_n_classes())
_REAL_SIGNATURE = inspect.signature(downloads.ensure_single_file_model)


@dataclasses.dataclass(frozen=True)
class _Case:
  downloader: _Downloader
  models: dict[MODEL_PRECISIONS, ModelInfo]
  backend: str
  ext: str
  precision: MODEL_PRECISIONS


_LAYOUTS: list[tuple[_Downloader, dict[MODEL_PRECISIONS, ModelInfo], str, str]] = [
  (GeoTFDownloaderV3_0, tf_models, "tf", "tflite"),
  (GeoOnnxDownloaderV3_0, onnx_models, "onnx", "onnx"),
  (GeoPTDownloaderV3_0, pt_models, "pt", "pt"),
]

_CASES = [
  pytest.param(
    _Case(downloader, table, backend, ext, precision),
    id=f"{downloader.__name__}-{precision}",
  )
  for downloader, table, backend, ext in _LAYOUTS
  for precision in table
]


def _backend_dir(case: _Case, app_data: Path) -> Path:
  return app_data / "geo-models" / "v3.0" / case.backend


def _expected_model_path(case: _Case, app_data: Path) -> Path:
  sha256 = case.models[case.precision].sha256
  assert sha256 is not None
  name = f"model-{case.precision}-{sha256[:12]}.{case.ext}"
  return _backend_dir(case, app_data) / name


def _expected_legacy_path(case: _Case, app_data: Path) -> Path:
  return _backend_dir(case, app_data) / f"model-{case.precision}.{case.ext}"


def _expected_lang_dir(case: _Case, app_data: Path) -> Path:
  return _backend_dir(case, app_data) / "labels"


# pt keeps its one precision callable without an argument, and is called that way.
def _get_model_path(case: _Case) -> Path:
  if case.downloader is GeoPTDownloaderV3_0:
    return GeoPTDownloaderV3_0._get_model_path()
  return case.downloader._get_model_path(case.precision)


def _is_available(case: _Case) -> bool:
  if case.downloader is GeoPTDownloaderV3_0:
    return GeoPTDownloaderV3_0._check_geo_model_available()
  return case.downloader._check_geo_model_available(case.precision)


def _download(case: _Case) -> None:
  if case.downloader is GeoPTDownloaderV3_0:
    GeoPTDownloaderV3_0._download_model()
  else:
    case.downloader._download_model(case.precision)


class _FakeDownload:
  def __init__(self, write_size: int | None) -> None:
    self.write_size = write_size
    self.calls: list[inspect.BoundArguments] = []

  def __call__(self, *args: object, **kwargs: object) -> None:
    bound = _REAL_SIGNATURE.bind(*args, **kwargs)
    self.calls.append(bound)
    if self.write_size is not None:
      target: Path = bound.arguments["model_path"]
      target.parent.mkdir(parents=True, exist_ok=True)
      target.write_bytes(b"\0" * self.write_size)


def _install_fake_download(
  case: _Case, monkeypatch: pytest.MonkeyPatch, write_size: int | None
) -> _FakeDownload:
  fake = _FakeDownload(write_size)
  # Patched wherever a class in the hierarchy binds the name, so the test holds
  # whichever module the download step lives in.
  for klass in case.downloader.__mro__:
    module = sys.modules[klass.__module__]
    if hasattr(module, "ensure_single_file_model"):
      monkeypatch.setattr(module, "ensure_single_file_model", fake)
  monkeypatch.setattr(downloads, "ensure_single_file_model", fake)
  return fake


def _skip_label_setup(case: _Case, monkeypatch: pytest.MonkeyPatch) -> None:
  def no_op(cls: type) -> None:
    pass

  monkeypatch.setattr(case.downloader, "ensure_labels_available", classmethod(no_op))


class _LabelsFlag:
  def __init__(self) -> None:
    self.available = True


def _control_labels_check(case: _Case, monkeypatch: pytest.MonkeyPatch) -> _LabelsFlag:
  flag = _LabelsFlag()

  def check(cls: type) -> bool:
    return flag.available

  monkeypatch.setattr(case.downloader, "_check_labels_available", classmethod(check))
  return flag


def _shrink_model(case: _Case, monkeypatch: pytest.MonkeyPatch) -> int:
  # In place: the downloader reads this very dict, so rebinding the module's
  # name would not reach it.
  monkeypatch.setitem(
    case.models,
    case.precision,
    dataclasses.replace(case.models[case.precision], file_size=_SMALL_FILE_SIZE),
  )
  return _SMALL_FILE_SIZE


def _write_model(case: _Case, app_data: Path, size: int) -> None:
  path = _expected_model_path(case, app_data)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(b"\0" * size)


def _write_en_us_labels(case: _Case, app_data: Path) -> None:
  lang_dir = _expected_lang_dir(case, app_data)
  lang_dir.mkdir(parents=True, exist_ok=True)
  (lang_dir / "en_us.txt").write_text("\n".join(_EN_US_LABELS), encoding="utf-8")


@pytest.fixture
def app_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  monkeypatch.setenv("BIRDNET_APP_DATA", str(tmp_path))
  return tmp_path.resolve()


@pytest.mark.no_tf
@pytest.mark.parametrize("case", _CASES)
def test_model_path(case: _Case, app_data: Path) -> None:
  assert _get_model_path(case) == _expected_model_path(case, app_data)


@pytest.mark.no_tf
@pytest.mark.parametrize("case", _CASES)
def test_label_directory(case: _Case, app_data: Path) -> None:
  assert case.downloader._get_lang_dir() == _expected_lang_dir(case, app_data)


@pytest.mark.no_tf
@pytest.mark.parametrize("case", _CASES)
def test_download_call(
  case: _Case, app_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  fake = _install_fake_download(case, monkeypatch, write_size=None)

  _download(case)

  assert len(fake.calls) == 1
  arguments = fake.calls[0].arguments
  assert arguments["info"] is case.models[case.precision]
  assert arguments["model_path"] == _expected_model_path(case, app_data)
  assert arguments["legacy_path"] == _expected_legacy_path(case, app_data)
  assert arguments["description"] == (
    f"Downloading geo model v3.0 ({case.backend}, {case.precision})"
  )


@pytest.mark.no_tf
@pytest.mark.parametrize("case", _CASES)
def test_unavailable_on_an_empty_cache(
  case: _Case, app_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _control_labels_check(case, monkeypatch)

  assert not _is_available(case)


@pytest.mark.no_tf
@pytest.mark.parametrize("case", _CASES)
def test_unavailable_at_another_size(
  case: _Case, app_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  size = _shrink_model(case, monkeypatch)
  _control_labels_check(case, monkeypatch)

  for wrong_size in (size - 1, size + 1):
    _write_model(case, app_data, wrong_size)
    assert not _is_available(case), wrong_size

  _write_model(case, app_data, size)
  assert _is_available(case)


@pytest.mark.no_tf
@pytest.mark.parametrize("case", _CASES)
def test_available_at_the_declared_size_with_labels(
  case: _Case, app_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  size = _shrink_model(case, monkeypatch)
  _control_labels_check(case, monkeypatch)
  _write_model(case, app_data, size)

  assert _is_available(case)


@pytest.mark.no_tf
@pytest.mark.parametrize("case", _CASES)
def test_unavailable_without_labels(
  case: _Case, app_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  size = _shrink_model(case, monkeypatch)
  flag = _control_labels_check(case, monkeypatch)
  _write_model(case, app_data, size)
  assert _is_available(case)

  flag.available = False

  assert not _is_available(case)


@pytest.mark.no_tf
@pytest.mark.parametrize("case", _CASES)
def test_warm_cache_load_does_not_download(
  case: _Case, app_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  size = _shrink_model(case, monkeypatch)
  _skip_label_setup(case, monkeypatch)
  _control_labels_check(case, monkeypatch)
  _write_en_us_labels(case, app_data)
  _write_model(case, app_data, size)
  fake = _install_fake_download(case, monkeypatch, write_size=None)

  model_path, labels = case.downloader.get_model_path_and_labels(
    "en_us", case.precision
  )

  assert fake.calls == []
  assert model_path == _expected_model_path(case, app_data)
  assert list(labels) == _EN_US_LABELS


@pytest.mark.no_tf
@pytest.mark.parametrize("case", _CASES)
def test_cold_cache_load_downloads_once(
  case: _Case, app_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  size = _shrink_model(case, monkeypatch)
  _skip_label_setup(case, monkeypatch)
  _control_labels_check(case, monkeypatch)
  _write_en_us_labels(case, app_data)
  fake = _install_fake_download(case, monkeypatch, write_size=size)

  model_path, labels = case.downloader.get_model_path_and_labels(
    "en_us", case.precision
  )

  assert len(fake.calls) == 1
  assert model_path == _expected_model_path(case, app_data)
  assert list(labels) == _EN_US_LABELS
