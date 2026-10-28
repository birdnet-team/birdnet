import numpy as np

from birdnet.utils.csv_format import hms_centis_fast


def test_hms_centis_fast() -> None:
  assert hms_centis_fast(0) == "00:00:00.00"
  assert hms_centis_fast(3661.5) == "01:01:01.50"
  assert hms_centis_fast(59.99) == "00:00:59.99"


def test_hms_centis_fast_carries_seconds_that_round_to_a_minute() -> None:
  assert hms_centis_fast(59.996) == "00:01:00.00"
  assert hms_centis_fast(119.999) == "00:02:00.00"
  assert hms_centis_fast(3599.996) == "01:00:00.00"
  assert hms_centis_fast(np.float32(59.996)) == "00:01:00.00"  # type: ignore[arg-type]
  assert hms_centis_fast(59.994) == "00:00:59.99"
