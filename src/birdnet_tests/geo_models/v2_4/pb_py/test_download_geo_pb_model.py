from birdnet.geo_models.v2_4.pb import GeoPBDownloaderV2_4


def xtest_double_download() -> None:
  # takes too long to run normally
  GeoPBDownloaderV2_4._download_model()
  GeoPBDownloaderV2_4._download_model()


if __name__ == "__main__":
  xtest_double_download()
