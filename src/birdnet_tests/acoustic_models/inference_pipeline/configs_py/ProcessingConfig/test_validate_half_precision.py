from birdnet.acoustic_models.inference_pipeline.configs import ProcessingConfig


def test_half_precision_is_valid() -> None:
  assert ProcessingConfig.validate_half_precision(True) is True
  assert ProcessingConfig.validate_half_precision(False) is False


def test_non_boolean_raises_error() -> None:
  from typing import Any

  invalid_values: list[Any] = [1, 0, "true", None, 3.14, [], {}]
  for invalid_value in invalid_values:
    try:
      ProcessingConfig.validate_half_precision(invalid_value)  # type: ignore
    except TypeError as e:
      assert str(e) == "half_precision must be a boolean"
    else:
      raise AssertionError(f"Expected TypeError for value: {invalid_value}")
