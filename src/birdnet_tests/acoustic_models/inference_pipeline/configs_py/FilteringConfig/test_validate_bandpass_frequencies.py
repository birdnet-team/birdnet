import pytest

from birdnet.acoustic.inference.configs import FilteringConfig

SUPPORTED_FMIN = 0
SUPPORTED_FMAX = 15000


def test_valid_frequencies() -> None:
  assert FilteringConfig.validate_bandpass_frequencies(
    100, 200, SUPPORTED_FMIN, SUPPORTED_FMAX
  ) == (100, 200)


def test_full_range_is_valid() -> None:
  assert FilteringConfig.validate_bandpass_frequencies(
    SUPPORTED_FMIN, SUPPORTED_FMAX, SUPPORTED_FMIN, SUPPORTED_FMAX
  ) == (SUPPORTED_FMIN, SUPPORTED_FMAX)


def test_fmin_none_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"bandpass minimum frequence must be specified"
  ):
    FilteringConfig.validate_bandpass_frequencies(
      None, 200, SUPPORTED_FMIN, SUPPORTED_FMAX
    )


def test_fmax_none_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"bandpass maximum frequence must be specified"
  ):
    FilteringConfig.validate_bandpass_frequencies(
      100, None, SUPPORTED_FMIN, SUPPORTED_FMAX
    )


def test_non_integer_raises_error() -> None:
  with pytest.raises(TypeError, match=r"bandpass frequencies must be integers"):
    FilteringConfig.validate_bandpass_frequencies(
      1.5, 200, SUPPORTED_FMIN, SUPPORTED_FMAX  # type: ignore
    )


def test_fmin_not_smaller_than_fmax_raises_error() -> None:
  with pytest.raises(ValueError, match=r"bandpass frequencies must be in the range"):
    FilteringConfig.validate_bandpass_frequencies(
      200, 100, SUPPORTED_FMIN, SUPPORTED_FMAX
    )


def test_out_of_supported_range_raises_error() -> None:
  with pytest.raises(ValueError, match=r"bandpass frequencies must be in the range"):
    FilteringConfig.validate_bandpass_frequencies(
      100, SUPPORTED_FMAX + 1, SUPPORTED_FMIN, SUPPORTED_FMAX
    )
