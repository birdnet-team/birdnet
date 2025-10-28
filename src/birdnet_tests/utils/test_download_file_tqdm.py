import tempfile
from pathlib import Path

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
    download_file_tqdm(
      url,
      output_path,
      download_size=dlsize,
      description="Downloading model",
    )
    assert output_path.is_file()
