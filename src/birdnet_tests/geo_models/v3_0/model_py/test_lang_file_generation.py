"""Language-file generation for the geo v3.0 downloader.

``GeoDownloaderBaseV3_0`` turns the raw (tab-separated) labels plus the shared
taxonomy CSV into one localized species-name file per supported language. This
is pure file logic, so it can be exercised with fixtures instead of downloads.
"""

from pathlib import Path

import pytest

import birdnet.geo.models.v3_0.model as geo_model
import birdnet.utils.label_manifest as label_manifest
from birdnet.geo.models.v3_0.model import GeoDownloaderBaseV3_0
from birdnet.globals import VALID_MODEL_LANGUAGES_V3_0
from birdnet.utils.label_manifest import LabelInput, sha256_bytes

RAW_LABELS = (
  "code1\tScivia one\tRobin\ncode2\tScivia two\tSparrow\ncode9\tScivia nine\tEagle\n"  # noqa: E501
)

# Taxonomy has a localized German name for code1 only; code2's is blank and code9
# is absent entirely - both must fall back to the English name from the raw labels.
TAXONOMY_CSV = (
  "species_code,com_name,common_name_de\n"
  "code1,Robin US,Rotkehlchen\n"
  "code2,Sparrow US,\n"
)


@pytest.fixture
def taxonomy_file(tmp_path: Path) -> Path:
  path = tmp_path / "taxonomy.csv"
  path.write_bytes(TAXONOMY_CSV.encode("utf-8"))
  return path


@pytest.fixture
def downloader(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, taxonomy_file: Path
) -> type[GeoDownloaderBaseV3_0]:
  raw_path = tmp_path / "labels_raw.txt"
  raw_path.write_bytes(RAW_LABELS.encode("utf-8"))
  taxonomy_path = taxonomy_file
  lang_dir = tmp_path / "labels"

  monkeypatch.setattr(geo_model, "_LABELS_RAW_PATH", raw_path)
  monkeypatch.setattr(geo_model, "_LABELS_DL_SIZE", raw_path.stat().st_size)
  monkeypatch.setattr(
    geo_model, "_LABELS_DL_SHA256", sha256_bytes(raw_path.read_bytes())
  )
  monkeypatch.setattr(geo_model, "_SETUP_LOCK_DIR", tmp_path / ".lock")
  monkeypatch.setattr(geo_model, "_LEGACY_LABELS_RAW_PATH", tmp_path / "legacy.txt")
  # The taxonomy is written by this fixture; nothing may reach for the network.
  monkeypatch.setattr(geo_model, "ensure_taxonomy_v3_available", lambda: taxonomy_path)
  # The digest is pinned once, the way the real constant is: verification then
  # compares it against whatever is on that path now.
  taxonomy_input = LabelInput(
    path=taxonomy_path,
    url="https://example.test/taxonomy.csv",
    size=taxonomy_path.stat().st_size,
    sha256=sha256_bytes(taxonomy_path.read_bytes()),
  )
  monkeypatch.setattr(geo_model, "get_taxonomy_v3_input", lambda: taxonomy_input)

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


def test_generate_lang_files_disambiguates_a_reused_species_code(
  downloader: type[GeoDownloaderBaseV3_0], taxonomy_file: Path
) -> None:
  """Upstream reuses a species code for two species (e.g. y01249 in v0.2-Jun2026).

  The matching row is not necessarily the last one, so keying on the code alone
  hands this species the other one's localized names.
  """
  taxonomy_file.write_text(
    "species_code,sci_name,com_name,common_name_de\n"
    "code1,Scivia one,Robin US,Rotkehlchen\n"  # the species the labels mean
    "code1,Scivia other,Other US,Anderer Name\n",  # shares the code, wins on order
    encoding="utf-8",
  )

  downloader._generate_lang_files()

  assert _read_lines(downloader.get_lang_file("de"))[0] == "Scivia one_Rotkehlchen"


def test_available_languages_match_the_public_language_list() -> None:
  # The two are edited by hand in different files; a divergence would otherwise
  # only surface as a bare AssertionError inside the downloaders.
  assert set(GeoDownloaderBaseV3_0.AVAILABLE_LANGUAGES) == set(
    VALID_MODEL_LANGUAGES_V3_0
  )


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


def test_check_labels_available_detects_a_swapped_taxonomy(
  downloader: type[GeoDownloaderBaseV3_0], taxonomy_file: Path
) -> None:
  """The taxonomy is shared, so another installed version can replace it.

  It keeps the same path and, in the case that caused this, the same byte size -
  so only the content itself distinguishes the two releases.
  """
  downloader._generate_lang_files()
  assert downloader._check_labels_available()

  path = taxonomy_file
  size_before = path.stat().st_size
  swapped = path.read_bytes().replace(b"Rotkehlchen", b"Rotkehlchan")
  path.write_bytes(swapped)
  assert path.stat().st_size == size_before, "the swap must not change the size"

  assert not downloader._check_labels_available()


def test_a_swapped_taxonomy_is_repaired_rather_than_regenerated_forever(
  downloader: type[GeoDownloaderBaseV3_0],
  monkeypatch: pytest.MonkeyPatch,
  taxonomy_file: Path,
) -> None:
  """Detecting the swap is not enough.

  If the wrong file is left in place, every later load regenerates from it and
  serves the other release's names without ever reporting anything.
  """
  downloader._generate_lang_files()
  good = taxonomy_file.read_bytes()
  taxonomy_file.write_bytes(good.replace(b"Rotkehlchen", b"Rotkehlchan"))
  assert not downloader._check_labels_available()

  restored: list[str] = []

  def _restore_taxonomy() -> Path:
    restored.append("called")
    taxonomy_file.write_bytes(good)
    return taxonomy_file

  monkeypatch.setattr(geo_model, "ensure_taxonomy_v3_available", _restore_taxonomy)

  downloader.ensure_labels_available()

  assert restored, "the mismatching input must be fetched again, not reused"
  assert downloader._check_labels_available(), "the cache must end up current"
  assert _read_lines(downloader.get_lang_file("de"))[0] == "Scivia one_Rotkehlchen"


def test_an_interrupted_generation_leaves_a_directory_that_fails_verification(
  downloader: type[GeoDownloaderBaseV3_0], monkeypatch: pytest.MonkeyPatch
) -> None:
  """The manifest is dropped before anything is written.

  A crash halfway through must not leave a directory still vouched for by the
  previous record, because its files no longer match it.
  """
  downloader._generate_lang_files()
  assert downloader._check_labels_available()

  original = geo_model.write_text_atomic
  written: list[Path] = []

  def _die_after_the_first_file(path: Path, content: str, **kwargs: object) -> None:
    if written:
      raise OSError("interrupted")
    written.append(path)
    original(path, content, **kwargs)

  monkeypatch.setattr(geo_model, "write_text_atomic", _die_after_the_first_file)

  with pytest.raises(OSError, match="interrupted"):
    downloader._generate_lang_files()

  assert not downloader._check_labels_available()


def test_generation_that_cannot_verify_itself_raises(
  downloader: type[GeoDownloaderBaseV3_0], monkeypatch: pytest.MonkeyPatch
) -> None:
  """Both inputs are verified before generating, so a directory that still does
  not verify afterwards is a bug - and must say so instead of quietly
  regenerating on every later load."""
  monkeypatch.setattr(downloader, "_generate_lang_files", classmethod(lambda cls: None))

  with pytest.raises(RuntimeError, match="could not be generated"):
    downloader.ensure_labels_available()


def test_a_current_cache_is_served_without_writing_anything(
  downloader: type[GeoDownloaderBaseV3_0],
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Pre-populated caches ship read-only in frozen and GUI applications, so a
  load that needs nothing must not take the setup lock."""
  downloader._generate_lang_files()

  def _fail(*args: object, **kwargs: object) -> None:
    raise AssertionError("a current cache must not acquire the setup lock")

  monkeypatch.setattr(geo_model, "directory_lock", _fail)

  downloader.ensure_labels_available()


def test_check_labels_available_detects_stale_lang_file(
  downloader: type[GeoDownloaderBaseV3_0],
) -> None:
  downloader._generate_lang_files()
  assert downloader._check_labels_available()

  # Simulate a lang file left over from an older labels version.
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


def test_ensure_labels_available_returns_early_when_present(
  downloader: type[GeoDownloaderBaseV3_0],
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  downloader._generate_lang_files()  # make the cache consistent

  def _fail_download(*args: object, **kwargs: object) -> int:
    raise AssertionError("download must not be called when labels are present")

  lock_dir = geo_model._LABELS_RAW_PATH.parent / ".lock"
  monkeypatch.setattr(label_manifest, "download_file_tqdm", _fail_download)
  monkeypatch.setattr(geo_model, "_SETUP_LOCK_DIR", lock_dir)

  downloader.ensure_labels_available()  # must not raise / must not download


def test_ensure_labels_available_downloads_and_generates_when_missing(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  raw_path = tmp_path / "labels_raw.txt"  # deliberately absent -> "stale"
  taxonomy_path = tmp_path / "taxonomy.csv"
  lang_dir = tmp_path / "labels"
  download_calls: list[Path] = []

  monkeypatch.setattr(geo_model, "_LABELS_RAW_PATH", raw_path)
  monkeypatch.setattr(geo_model, "_LABELS_DL_SIZE", len(RAW_LABELS.encode("utf-8")))
  monkeypatch.setattr(
    geo_model, "_LABELS_DL_SHA256", sha256_bytes(RAW_LABELS.encode("utf-8"))
  )
  monkeypatch.setattr(geo_model, "_SETUP_LOCK_DIR", tmp_path / ".lock")
  monkeypatch.setattr(
    geo_model,
    "get_taxonomy_v3_input",
    lambda: LabelInput(
      path=taxonomy_path,
      url="https://example.test/taxonomy.csv",
      size=len(TAXONOMY_CSV.encode("utf-8")),
      sha256=sha256_bytes(TAXONOMY_CSV.encode("utf-8")),
    ),
  )

  def fake_download(url: str, file_path: Path, **kwargs: object) -> int:
    # The real downloader writes bytes verbatim; do the same so the on-disk size
    # matches _LABELS_DL_SIZE (write_text would translate newlines on Windows).
    download_calls.append(file_path)
    file_path.write_bytes(RAW_LABELS.encode("utf-8"))
    return len(RAW_LABELS.encode("utf-8"))

  def fake_ensure_taxonomy() -> Path:
    # Bytes, not text: write_text would translate newlines on Windows and the
    # on-disk size would then disagree with the size declared above.
    taxonomy_path.write_bytes(TAXONOMY_CSV.encode("utf-8"))
    return taxonomy_path

  monkeypatch.setattr(label_manifest, "download_file_tqdm", fake_download)
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
