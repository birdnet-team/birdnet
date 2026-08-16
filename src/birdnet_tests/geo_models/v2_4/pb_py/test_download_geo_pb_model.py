import pytest

from birdnet.geo.models.v2_4.pb import GeoPBDownloaderV2_4


@pytest.mark.skip(reason="re-downloads the model; run manually via __main__")
def test_double_download() -> None:
  GeoPBDownloaderV2_4._download_model()
  GeoPBDownloaderV2_4._download_model()


if __name__ == "__main__":
  test_double_download()
