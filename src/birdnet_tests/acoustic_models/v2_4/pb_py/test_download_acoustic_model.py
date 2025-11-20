from birdnet.acoustic_models.v2_4.pb import AcousticPBDownloaderV2_4


def xtest_double_download() -> None:
  # takes too long to run normally
  AcousticPBDownloaderV2_4._download_model()
  AcousticPBDownloaderV2_4._download_model()


if __name__ == "__main__":
  xtest_double_download()
