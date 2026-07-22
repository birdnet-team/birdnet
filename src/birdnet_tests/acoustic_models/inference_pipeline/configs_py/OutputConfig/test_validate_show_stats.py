import pytest

from birdnet.acoustic.inference.configs import OutputConfig


@pytest.mark.parametrize("value", ["minimal", "progress", "benchmark", None])
def test_valid_values(value: str | None) -> None:
  assert OutputConfig.validate_show_stats(value) == value


def test_unknown_value_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"show stats must be one of 'minimal', 'progress' or 'benchmark'",
  ):
    OutputConfig.validate_show_stats("verbose")  # type: ignore


def test_empty_string_raises_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"show stats must be one of 'minimal', 'progress' or 'benchmark'",
  ):
    OutputConfig.validate_show_stats("")  # type: ignore
