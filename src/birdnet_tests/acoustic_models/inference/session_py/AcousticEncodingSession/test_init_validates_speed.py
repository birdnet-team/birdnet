from pathlib import Path

import numpy as np
import pytest

from birdnet_tests.fake_acoustic_backend import fake_encode_session


@pytest.mark.no_tf
def test_non_numeric_speed_is_rejected_at_construction(tmp_path: Path) -> None:
  with pytest.raises(TypeError, match=r"speed must be a number"):
    fake_encode_session(tmp_path, speed="fast")  # type: ignore[arg-type]


@pytest.mark.no_tf
def test_out_of_range_speed_is_rejected_at_construction(tmp_path: Path) -> None:
  with pytest.raises(ValueError, match=r"speed must be in the range"):
    fake_encode_session(tmp_path, speed=1000.0)


@pytest.mark.no_tf
@pytest.mark.parametrize("speed", [2, 0.5, np.float32(1.0), np.int64(2)])
def test_numeric_speed_is_accepted(tmp_path: Path, speed: float) -> None:
  fake_encode_session(tmp_path, speed=speed)
