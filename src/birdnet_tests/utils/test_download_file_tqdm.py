import tempfile
from pathlib import Path

import pytest
from requests import ReadTimeout

from birdnet.utils import download_file_tqdm


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
    except ReadTimeout as e:
      # sometimes the server is slow, so we just skip the test then
      pytest.skip(f"Download timed out: {e}")
    assert output_path.is_file()
