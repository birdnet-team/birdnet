import tempfile
from pathlib import Path

import numpy.testing as npt
import pytest

from birdnet.models.model_v2p4 import Downloader, ModelV2p4


def test_download_creates_all_files():
  with tempfile.TemporaryDirectory(prefix="birdnet.test_downloader.") as tmp_dir:
    downloader = Downloader(Path(tmp_dir))
    # pylint: disable=W0212:protected-access
    assert not downloader._check_model_files_exist()
    downloader.ensure_model_is_available()
    assert downloader._check_model_files_exist()
