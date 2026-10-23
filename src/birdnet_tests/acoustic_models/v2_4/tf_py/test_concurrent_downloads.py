"""Two loads of the 2.4 tf model at once must not wipe each other.

The three precisions share one labels directory and a download replaces it
wholesale, so two concurrent downloads (sibling precisions, or the same one
twice) each remove what the other just extracted. Threads stand in for
processes; the lock is a directory, so what holds here holds across processes.
"""

from __future__ import annotations

import shutil
import threading
import time
import zipfile
from pathlib import Path

import pytest

from birdnet.acoustic.models.v2_4.tf import AcousticTFDownloaderV2_4, models
from birdnet.globals import ENV_VAR_APP_DATA, MODEL_PRECISIONS
from birdnet.utils import helper
from birdnet.utils.local_data import get_model_root_dir

pytestmark = pytest.mark.no_tf

_DOWNLOAD_DURATION_S = 0.3


class _FakeTransport:
  """Builds the zip a 2.4 tf download delivers and records when it ran."""

  def __init__(self) -> None:
    self.intervals: list[tuple[float, float]] = []
    self.inside = threading.Event()
    self._lock = threading.Lock()

  def __call__(
    self,
    url: str,
    file_path: Path,
    *,
    download_size: int | None = None,
    description: str | None = None,
  ) -> int:
    start = time.perf_counter()
    self.inside.set()
    info = next(info for info in models.values() if info.dl_url == url)
    with zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED) as archive:
      archive.writestr(info.dl_file_name, b"\0" * info.file_size)
      for lang in AcousticTFDownloaderV2_4.AVAILABLE_LANGUAGES:
        archive.writestr(f"labels/{lang}.txt", "Genus_species_Species\n")
    time.sleep(_DOWNLOAD_DURATION_S)
    with self._lock:
      self.intervals.append((start, time.perf_counter()))
    return file_path.stat().st_size


class _Load(threading.Thread):
  def __init__(self, precision: MODEL_PRECISIONS) -> None:
    super().__init__(daemon=True)
    self.precision = precision
    self.error: BaseException | None = None

  def run(self) -> None:
    try:
      AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", self.precision)
    except BaseException as error:  # noqa: BLE001
      self.error = error


@pytest.fixture
def transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _FakeTransport:
  monkeypatch.setenv(ENV_VAR_APP_DATA, str(tmp_path))
  fake = _FakeTransport()
  monkeypatch.setattr(helper, "download_file_tqdm", fake)
  return fake


def _run_second_once_the_first_downloads(
  transport: _FakeTransport, first: MODEL_PRECISIONS, second: MODEL_PRECISIONS
) -> None:
  loads = [_Load(first)]
  loads[0].start()
  assert transport.inside.wait(timeout=10)
  loads.append(_Load(second))
  loads[1].start()
  for load in loads:
    load.join(timeout=30)
    assert not load.is_alive(), f"loading {load.precision} did not finish"
    assert load.error is None, f"loading {load.precision} failed: {load.error!r}"
  assert not (get_model_root_dir("acoustic", "2.4", "tf") / ".download.lock").exists()


def test_sibling_precisions_download_one_after_the_other(
  transport: _FakeTransport,
) -> None:
  _run_second_once_the_first_downloads(transport, "fp32", "int8")

  assert len(transport.intervals) == 2
  (_, first_end), (second_start, _) = sorted(transport.intervals)
  assert first_end <= second_start, "the second download overlapped the first"


def test_the_same_precision_twice_downloads_once(transport: _FakeTransport) -> None:
  _run_second_once_the_first_downloads(transport, "fp32", "fp32")

  assert len(transport.intervals) == 1


def test_a_cached_precision_loads_while_a_sibling_downloads(
  transport: _FakeTransport, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A load of a cached precision takes no lock, so a sibling's download must
  never leave the shared labels directory half replaced. The load is run at the
  worst moment: its first check before the sibling would remove the labels, its
  last one between that removal and the move of the new labels, if the sibling
  replaces them at all."""
  AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "fp32")
  _, lang_dir = AcousticTFDownloaderV2_4._get_paths("fp32")
  reader_prechecked = threading.Event()
  labels_removed = threading.Event()
  reader_done = threading.Event()
  real_check = AcousticTFDownloaderV2_4._check_acoustic_model_available
  real_rmtree = shutil.rmtree
  real_move = shutil.move
  writer = _Load("int8")

  def check(cls: type, precision: MODEL_PRECISIONS) -> bool:
    result = real_check(precision)
    is_reader = threading.current_thread() is threading.main_thread()
    if is_reader and not reader_prechecked.is_set():
      reader_prechecked.set()
      while writer.is_alive() and not labels_removed.wait(timeout=0.05):
        pass
    return result

  def rmtree(path: Path, *args: object, **kwargs: object) -> None:
    if Path(path) == lang_dir:
      reader_prechecked.wait(timeout=10)
    real_rmtree(path, *args, **kwargs)  # type: ignore[arg-type]
    if Path(path) == lang_dir:
      labels_removed.set()

  def move(src: Path, dst: Path, *args: object, **kwargs: object) -> object:
    if Path(dst) == lang_dir:
      reader_done.wait(timeout=10)
    return real_move(src, dst, *args, **kwargs)  # type: ignore[arg-type]

  monkeypatch.setattr(
    AcousticTFDownloaderV2_4, "_check_acoustic_model_available", classmethod(check)
  )
  monkeypatch.setattr(shutil, "rmtree", rmtree)
  monkeypatch.setattr(shutil, "move", move)

  writer.start()
  try:
    AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "fp32")
  finally:
    reader_done.set()
  writer.join(timeout=30)
  assert not writer.is_alive()
  assert writer.error is None, f"loading int8 failed: {writer.error!r}"
