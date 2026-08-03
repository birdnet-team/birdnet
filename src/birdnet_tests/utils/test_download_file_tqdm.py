import re
import tempfile
from pathlib import Path

import pytest

from birdnet.utils.helper import download_file_tqdm


def test_download_geo_model_to_tmp() -> None:
  # from birdnet.geo_models.v2_4.tf import GeoTFDownloaderV2_4
  # url = GeoTFDownloaderV2_4._model_info.dl_url
  # dlsize = GeoTFDownloaderV2_4._model_info.dl_size
  url = "https://zenodo.org/records/10943500/files/recording_location.txt"
  dlsize = 142

  with tempfile.TemporaryDirectory(
    prefix="birdnet_tests.test_download_geo_model_to_tmp."
  ) as tmp_dir:
    output_path = Path(tmp_dir) / "dl.zip"
    try:
      download_file_tqdm(
        url,
        output_path,
        download_size=dlsize,
        description="Downloading model",
      )
    except ValueError as e:
      # Zenodo answers 503/504 under load; a server-side error is not a library
      # defect, so skip instead of failing the run. (The error message spans two
      # lines, so match the status line instead of comparing the whole string.)
      if re.search(r"Status code: 5\d\d", str(e)):
        pytest.skip(f"Server-side download error: {e}")
      raise
    assert output_path.is_file()
