from birdnet.geo.models.v3_0.tf import GeoTFDownloaderV3_0
from birdnet.globals import MODEL_PRECISION_FP16, MODEL_PRECISION_FP32, MODEL_PRECISION_INT8


def xtest_double_download_fp32() -> None:
  # takes too long to run normally
  GeoTFDownloaderV3_0._download_model(MODEL_PRECISION_FP32)
  GeoTFDownloaderV3_0._download_model(MODEL_PRECISION_FP32)


def xtest_double_download_fp16() -> None:
  # takes too long to run normally
  GeoTFDownloaderV3_0._download_model(MODEL_PRECISION_FP16)
  GeoTFDownloaderV3_0._download_model(MODEL_PRECISION_FP16)


def xtest_double_download_int8() -> None:
  # takes too long to run normally
  GeoTFDownloaderV3_0._download_model(MODEL_PRECISION_INT8)
  GeoTFDownloaderV3_0._download_model(MODEL_PRECISION_INT8)


if __name__ == "__main__":
  xtest_double_download_fp32()
  xtest_double_download_fp16()
  xtest_double_download_int8()
