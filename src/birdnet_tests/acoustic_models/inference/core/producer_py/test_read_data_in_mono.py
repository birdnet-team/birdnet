import numpy as np
import pytest

from birdnet.acoustic.inference.core.producer import read_data_in_mono

pytestmark = pytest.mark.no_tf


def test_three_dimensional_array_raises_value_error() -> None:
  audio = np.zeros((48_000, 2, 2), dtype=np.float32)
  with pytest.raises(ValueError, match="1-D or 2-D"):
    read_data_in_mono(start_samples=0, end_samples=48_000, audio_data=audio)
