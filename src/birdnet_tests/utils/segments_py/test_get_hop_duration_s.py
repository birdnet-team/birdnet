import pytest

from birdnet.utils.segments import get_hop_duration_s


def test_get_hop_duration_s() -> None:
  assert get_hop_duration_s(3.0, 1.0, 1.0) == 2.0
  assert get_hop_duration_s(3.0, 0.0, 2.0) == 6.0


def test_get_hop_duration_s_rejects_overlap_ge_segment() -> None:
  with pytest.raises(AssertionError):
    get_hop_duration_s(3.0, 3.0, 1.0)
