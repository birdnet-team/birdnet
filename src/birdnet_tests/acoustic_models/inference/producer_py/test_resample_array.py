import numpy as np
import numpy.testing as npt

from birdnet.acoustic_models.inference.producer import resample_array_by_sr
from birdnet.helper import apply_speed_to_samples


def test_same_sr__changes_nothing() -> None:
  result = resample_array_by_sr(
    array=np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32),
    sample_rate=48000,
    target_sample_rate=48000,
  )
  npt.assert_equal(result, np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32))


def test_doubletime__doubles_frame_count() -> None:
  result = resample_array_by_sr(
    array=np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32),
    sample_rate=48000,
    target_sample_rate=48000 * 2,
  )
  assert result.shape == (8,)


def test_halftime__halfes_frame_count() -> None:
  result = resample_array_by_sr(
    array=np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32),
    sample_rate=48000,
    target_sample_rate=48000 // 2,
  )
  assert result.shape == (2,)


def test_non_integer_ratio() -> None:
  sr = 48000
  speed = 1.25
  effective_sample_rate = apply_speed_to_samples(sr, speed)
  target_sr = 48000
  result = resample_array_by_sr(
    array=np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32),
    sample_rate=effective_sample_rate,
    target_sample_rate=target_sr,
  )
  assert result.shape == (3,)
