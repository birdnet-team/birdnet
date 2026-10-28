from birdnet.utils.segments import apply_speed_to_samples


def test_apply_speed_to_samples_rounds() -> None:
  assert apply_speed_to_samples(10, 0.5) == 5
  assert apply_speed_to_samples(3, 0.5) == 2  # 1.5 rounds to nearest even (2)
