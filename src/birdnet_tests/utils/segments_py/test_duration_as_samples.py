from birdnet.utils.segments import duration_as_samples


def test_duration_as_samples() -> None:
  assert duration_as_samples(1.0, 48000) == 48000
  assert duration_as_samples(0.5, 48000) == 24000
