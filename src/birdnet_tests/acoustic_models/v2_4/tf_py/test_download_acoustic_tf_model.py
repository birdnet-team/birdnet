from birdnet.acoustic_models.v2_4.tf import AcousticTFDownloaderV2_4


def xtest_double_download() -> None:
  # takes too long to run normally
  AcousticTFDownloaderV2_4._download_model("fp32")
  AcousticTFDownloaderV2_4._download_model("fp32")


if __name__ == "__main__":
  xtest_double_download()
