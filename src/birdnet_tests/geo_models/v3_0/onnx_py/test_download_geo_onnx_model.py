import pytest

from birdnet.geo.models.v3_0.onnx import GeoOnnxDownloaderV3_0
from birdnet.globals import (
  MODEL_PRECISION_FP16,
  MODEL_PRECISION_FP32,
)


@pytest.mark.skip(reason="re-downloads the model; run manually via __main__")
def test_double_download_fp32() -> None:
  GeoOnnxDownloaderV3_0._download_model(MODEL_PRECISION_FP32)
  GeoOnnxDownloaderV3_0._download_model(MODEL_PRECISION_FP32)


@pytest.mark.skip(reason="re-downloads the model; run manually via __main__")
def test_double_download_fp16() -> None:
  GeoOnnxDownloaderV3_0._download_model(MODEL_PRECISION_FP16)
  GeoOnnxDownloaderV3_0._download_model(MODEL_PRECISION_FP16)


if __name__ == "__main__":
  test_double_download_fp32()
  test_double_download_fp16()
