import numpy as np
import pytest

from birdnet.acoustic.inference.configs import InferenceConfig


def test_single_tuple_is_wrapped() -> None:
  audio = np.zeros(3, dtype=np.float32)
  result = InferenceConfig.validate_input_audio((audio, 48000))
  assert len(result) == 1
  np.testing.assert_array_equal(result[0][0], audio)
  assert result[0][1] == 48000


def test_list_of_tuples_is_valid() -> None:
  audio_a = np.zeros(3, dtype=np.float32)
  audio_b = np.ones(2, dtype=np.int16)
  result = InferenceConfig.validate_input_audio([(audio_a, 48000), (audio_b, 32000)])
  assert len(result) == 2


def test_non_iterable_raises_error() -> None:
  with pytest.raises(ValueError, match=r"Unsupported input type: <class 'int'>"):
    InferenceConfig.validate_input_audio(123)


def test_element_not_a_tuple_raises_error() -> None:
  with pytest.raises(ValueError, match=r"Unsupported input type"):
    InferenceConfig.validate_input_audio([123])


def test_tuple_wrong_length_raises_error() -> None:
  audio = np.zeros(3, dtype=np.float32)
  with pytest.raises(
    ValueError, match=r"Input audio tuple must have exactly two elements"
  ):
    InferenceConfig.validate_input_audio([(audio, 48000, "extra")])


def test_first_element_not_ndarray_raises_error() -> None:
  with pytest.raises(
    ValueError, match=r"First element of input audio tuple must be a numpy ndarray"
  ):
    InferenceConfig.validate_input_audio([("not an array", 48000)])


def test_sample_rate_not_int_raises_error() -> None:
  audio = np.zeros(3, dtype=np.float32)
  with pytest.raises(
    ValueError,
    match=r"Second element of input audio tuple must be an integer sample rate",
  ):
    InferenceConfig.validate_input_audio([(audio, 48000.0)])


def test_sample_rate_not_positive_raises_error() -> None:
  audio = np.zeros(3, dtype=np.float32)
  with pytest.raises(
    ValueError, match=r"Sample rate must be a positive integer, got 0."
  ):
    InferenceConfig.validate_input_audio([(audio, 0)])


def test_wrong_dtype_raises_error() -> None:
  audio = np.array(["a", "b"])
  with pytest.raises(
    ValueError, match=r"Audio array must have an integer or floating-point dtype"
  ):
    InferenceConfig.validate_input_audio([(audio, 48000)])
