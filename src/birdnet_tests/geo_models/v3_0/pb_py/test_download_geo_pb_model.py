from birdnet.geo.models.v3_0.pb import GeoPBDownloaderV3_0


def xtest_double_download() -> None:
  # takes too long to run normally
  GeoPBDownloaderV3_0._download_model()
  GeoPBDownloaderV3_0._download_model()


if __name__ == "__main__":
  xtest_double_download()
