from birdnet.geo.models.v3_0.onnx import GeoOnnxDownloaderV3_0
from birdnet.globals import (
  MODEL_PRECISION_FP16,
  MODEL_PRECISION_FP32,
)


def xtest_double_download_fp32() -> None:
  # takes too long to run normally
  GeoOnnxDownloaderV3_0._download_model(MODEL_PRECISION_FP32)
  GeoOnnxDownloaderV3_0._download_model(MODEL_PRECISION_FP32)


def xtest_double_download_fp16() -> None:
  # takes too long to run normally
  GeoOnnxDownloaderV3_0._download_model(MODEL_PRECISION_FP16)
  GeoOnnxDownloaderV3_0._download_model(MODEL_PRECISION_FP16)


if __name__ == "__main__":
  xtest_double_download_fp32()
  xtest_double_download_fp16()
