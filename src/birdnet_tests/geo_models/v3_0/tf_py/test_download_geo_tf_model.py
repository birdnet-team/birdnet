import pytest

from birdnet.geo.models.v3_0.tf import GeoTFDownloaderV3_0
from birdnet.globals import (
  MODEL_PRECISION_FP16,
  MODEL_PRECISION_FP32,
  MODEL_PRECISION_INT8,
)


@pytest.mark.skip(reason="re-downloads the model; run manually via __main__")
def test_double_download_fp32() -> None:
  GeoTFDownloaderV3_0._download_model(MODEL_PRECISION_FP32)
  GeoTFDownloaderV3_0._download_model(MODEL_PRECISION_FP32)


@pytest.mark.skip(reason="re-downloads the model; run manually via __main__")
def test_double_download_fp16() -> None:
  GeoTFDownloaderV3_0._download_model(MODEL_PRECISION_FP16)
  GeoTFDownloaderV3_0._download_model(MODEL_PRECISION_FP16)


@pytest.mark.skip(reason="re-downloads the model; run manually via __main__")
def test_double_download_int8() -> None:
  GeoTFDownloaderV3_0._download_model(MODEL_PRECISION_INT8)
  GeoTFDownloaderV3_0._download_model(MODEL_PRECISION_INT8)


if __name__ == "__main__":
  test_double_download_fp32()
  test_double_download_fp16()
  test_double_download_int8()
