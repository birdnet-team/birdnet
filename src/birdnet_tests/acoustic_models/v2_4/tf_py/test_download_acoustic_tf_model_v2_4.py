import pytest

from birdnet.acoustic.models.v2_4.tf import AcousticTFDownloaderV2_4


@pytest.mark.skip(reason="re-downloads the model; run manually via __main__")
def test_double_download() -> None:
  AcousticTFDownloaderV2_4._download_model("fp32")
  AcousticTFDownloaderV2_4._download_model("fp32")


if __name__ == "__main__":
  test_double_download()
