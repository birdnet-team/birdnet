import tempfile
from pathlib import Path

from birdnet.models.model_v2m4_tflite import DownloaderTFLite


def test_download_creates_all_files():
  with tempfile.TemporaryDirectory(prefix="birdnet.test_downloader.") as tmp_dir:
    downloader = DownloaderTFLite(Path(tmp_dir))
    # pylint: disable=W0212:protected-access
    assert not downloader._check_model_files_exist()
    downloader.ensure_model_is_available()
    assert downloader._check_model_files_exist()
