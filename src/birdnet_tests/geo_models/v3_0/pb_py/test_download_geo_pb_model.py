import pytest

from birdnet.geo.models.v3_0.pb import GeoPBDownloaderV3_0


@pytest.mark.skip(reason="re-downloads the model; run manually via __main__")
def test_double_download() -> None:
  GeoPBDownloaderV3_0._download_model()
  GeoPBDownloaderV3_0._download_model()


if __name__ == "__main__":
  test_double_download()
