import tempfile
from pathlib import Path

from birdnet.models.v2m4.model_v2m4_tflite import DownloaderTFLite


def test_download_creates_all_files():
  with tempfile.TemporaryDirectory(prefix="birdnet.test_downloader.") as tmp_dir:
    downloader = DownloaderTFLite(Path(tmp_dir))
    # pylint: disable=W0212:protected-access
    assert not downloader._check_model_files_exist()
    downloader.ensure_model_is_available()
    assert downloader._check_model_files_exist()

def test_meta_model_check():
  with tempfile.TemporaryDirectory(prefix="birdnet.test_downloader.") as tmp_dir:
    downloader = DownloaderTFLite(Path(tmp_dir))
    # pylint: disable=W0212:protected-access
    assert not downloader._check_model_files_exist()
    downloader.ensure_model_is_available()
    assert downloader._check_model_files_exist()
    downloader.meta_model_path.write_text("test")
    assert not downloader._check_model_files_exist()

def test_audio_model_check():
  with tempfile.TemporaryDirectory(prefix="birdnet.test_downloader.") as tmp_dir:
    downloader = DownloaderTFLite(Path(tmp_dir))
    # pylint: disable=W0212:protected-access
    assert not downloader._check_model_files_exist()
    downloader.ensure_model_is_available()
    assert downloader._check_model_files_exist()
    downloader.audio_model_path.write_text("test")
    assert not downloader._check_model_files_exist()
