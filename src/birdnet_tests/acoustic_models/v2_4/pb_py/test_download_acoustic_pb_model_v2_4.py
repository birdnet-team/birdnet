import pytest

from birdnet.acoustic.models.v2_4.pb import AcousticPBDownloaderV2_4


@pytest.mark.skip(reason="re-downloads the model; run manually via __main__")
def test_double_download() -> None:
  AcousticPBDownloaderV2_4._download_model()
  AcousticPBDownloaderV2_4._download_model()


if __name__ == "__main__":
  test_double_download()
