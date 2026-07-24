"""Language-file generation for the geo v3.0 downloader.

``GeoDownloaderBaseV3_0`` turns the raw (tab-separated) labels plus the shared
taxonomy CSV into one localized species-name file per supported language. This
is pure file logic, so it can be exercised with fixtures instead of downloads.
"""

from pathlib import Path

import pytest

import birdnet.geo.models.v3_0.model as geo_model
from birdnet.geo.models.v3_0.model import GeoDownloaderBaseV3_0

RAW_LABELS = "code1\tScivia one\tRobin\ncode2\tScivia two\tSparrow\ncode9\tScivia nine\tEagle\n"  # noqa: E501

# Taxonomy has a localized German name for code1 only; code2's is blank and code9
# is absent entirely - both must fall back to the English name from the raw labels.
TAXONOMY_CSV = (
  "species_code,com_name,common_name_de\n"
  "code1,Robin US,Rotkehlchen\n"
  "code2,Sparrow US,\n"
)


@pytest.fixture
def downloader(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> type[GeoDownloaderBaseV3_0]:
  raw_path = tmp_path / "labels_raw.txt"
  raw_path.write_text(RAW_LABELS, encoding="utf-8")
  taxonomy_path = tmp_path / "taxonomy.csv"
  taxonomy_path.write_text(TAXONOMY_CSV, encoding="utf-8")
  lang_dir = tmp_path / "labels"

  monkeypatch.setattr(geo_model, "_LABELS_RAW_PATH", raw_path)
  monkeypatch.setattr(geo_model, "_LABELS_DL_SIZE", raw_path.stat().st_size)
  monkeypatch.setattr(geo_model, "get_taxonomy_v3_path", lambda: taxonomy_path)
  monkeypatch.setattr(geo_model, "taxonomy_v3_available", lambda: True)

  class _Downloader(GeoDownloaderBaseV3_0):
    @classmethod
    def _get_lang_dir(cls) -> Path:
      return lang_dir

  return _Downloader


def _read_lines(path: Path) -> list[str]:
  return path.read_text(encoding="utf-8").splitlines()


def test_generate_lang_files_uses_localized_names(
  downloader: type[GeoDownloaderBaseV3_0],
) -> None:
  downloader._generate_lang_files()

  de_file = downloader.get_lang_file("de")
  assert _read_lines(de_file) == [
    "Scivia one_Rotkehlchen",  # localized German name from taxonomy
    "Scivia two_Sparrow",  # blank localized name -> English fallback
    "Scivia nine_Eagle",  # species absent from taxonomy -> English fallback
  ]


def test_generate_lang_files_en_us_uses_com_name(
  downloader: type[GeoDownloaderBaseV3_0],
) -> None:
  downloader._generate_lang_files()

  en_file = downloader.get_lang_file("en_us")
  assert _read_lines(en_file) == [
    "Scivia one_Robin US",
    "Scivia two_Sparrow US",
    "Scivia nine_Eagle",  # not in taxonomy -> English fallback
  ]


def test_generate_lang_files_writes_all_languages(
  downloader: type[GeoDownloaderBaseV3_0],
) -> None:
  downloader._generate_lang_files()

  for lang in downloader.AVAILABLE_LANGUAGES:
    lang_file = downloader.get_lang_file(lang)
    assert lang_file.is_file(), lang
    # every generated file has one entry per raw label line
    assert len(_read_lines(lang_file)) == 3


def test_check_labels_available_true_when_consistent(
  downloader: type[GeoDownloaderBaseV3_0],
) -> None:
  assert not downloader._check_labels_available()  # nothing generated yet

  downloader._generate_lang_files()

  assert downloader._check_labels_available()


def test_check_labels_available_detects_stale_lang_file(
  downloader: type[GeoDownloaderBaseV3_0],
) -> None:
  downloader._generate_lang_files()
  assert downloader._check_labels_available()

  # Simulate a lang file left over from an older labels version: wrong line count.
  stale = downloader.get_lang_file("de")
  stale.write_text("only_one_line\n", encoding="utf-8")

  assert not downloader._check_labels_available()


def test_check_labels_available_false_when_raw_labels_missing(
  downloader: type[GeoDownloaderBaseV3_0],
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  downloader._generate_lang_files()
  # Point at a non-existent raw labels file.
  gone = geo_model._LABELS_RAW_PATH.parent / "gone.txt"
  monkeypatch.setattr(geo_model, "_LABELS_RAW_PATH", gone)
  assert not downloader._check_labels_available()


def test_write_text_atomic_leaves_no_temp_files(tmp_path: Path) -> None:
  target = tmp_path / "out.txt"
  geo_model._write_text_atomic(target, "hello world")

  assert target.read_text(encoding="utf-8") == "hello world"
  # the atomic write must not leave behind any *.tmp scratch files
  assert list(tmp_path.glob("*.tmp")) == []


def test_count_lines(tmp_path: Path) -> None:
  path = tmp_path / "f.txt"
  path.write_text("a\nb\nc\n", encoding="utf-8")
  assert geo_model._count_lines(path) == 3


def test_ensure_labels_available_returns_early_when_present(
  downloader: type[GeoDownloaderBaseV3_0],
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  downloader._generate_lang_files()  # make the cache consistent

  def _fail_download(*args: object, **kwargs: object) -> int:
    raise AssertionError("download must not be called when labels are present")

  lock_dir = geo_model._LABELS_RAW_PATH.parent / ".lock"
  monkeypatch.setattr(geo_model, "download_file_tqdm", _fail_download)
  monkeypatch.setattr(geo_model, "_SETUP_LOCK_DIR", lock_dir)

  downloader.ensure_labels_available()  # must not raise / must not download


def test_ensure_labels_available_downloads_and_generates_when_missing(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  raw_path = tmp_path / "labels_raw.txt"  # deliberately absent -> "stale"
  taxonomy_path = tmp_path / "taxonomy.csv"
  lang_dir = tmp_path / "labels"
  state = {"taxonomy_available": False}
  download_calls: list[Path] = []

  monkeypatch.setattr(geo_model, "_LABELS_RAW_PATH", raw_path)
  monkeypatch.setattr(geo_model, "_LABELS_DL_SIZE", len(RAW_LABELS.encode("utf-8")))
  monkeypatch.setattr(geo_model, "_SETUP_LOCK_DIR", tmp_path / ".lock")
  monkeypatch.setattr(geo_model, "get_taxonomy_v3_path", lambda: taxonomy_path)
  monkeypatch.setattr(
    geo_model, "taxonomy_v3_available", lambda: state["taxonomy_available"]
  )

  def fake_download(url: str, file_path: Path, **kwargs: object) -> int:
    # The real downloader writes bytes verbatim; do the same so the on-disk size
    # matches _LABELS_DL_SIZE (write_text would translate newlines on Windows).
    download_calls.append(file_path)
    file_path.write_bytes(RAW_LABELS.encode("utf-8"))
    return len(RAW_LABELS.encode("utf-8"))

  def fake_ensure_taxonomy() -> Path:
    taxonomy_path.write_text(TAXONOMY_CSV, encoding="utf-8")
    state["taxonomy_available"] = True
    return taxonomy_path

  monkeypatch.setattr(geo_model, "download_file_tqdm", fake_download)
  monkeypatch.setattr(geo_model, "ensure_taxonomy_v3_available", fake_ensure_taxonomy)

  class _Downloader(GeoDownloaderBaseV3_0):
    @classmethod
    def _get_lang_dir(cls) -> Path:
      return lang_dir

  _Downloader.ensure_labels_available()

  assert download_calls == [raw_path]
  assert raw_path.read_text(encoding="utf-8") == RAW_LABELS
  assert _Downloader._check_labels_available()
  # the lock directory is released after setup
  assert not (tmp_path / ".lock").exists()
