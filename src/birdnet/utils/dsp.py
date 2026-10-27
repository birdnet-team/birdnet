"""Signal and score transforms: band-pass, padding, sigmoid, logit, softmax."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from birdnet.globals import Float32Array


def bandpass_signal(
  audio_signal: Float32Array,
  rate: int,
  fmin: int,
  fmax: int,
  new_fmin: int,
  new_fmax: int,
) -> Float32Array:
  assert rate > 0
  assert fmin >= 0
  assert fmin < fmax
  assert new_fmin >= 0
  assert new_fmin < new_fmax

  from scipy.signal import butter, lfilter

  nth_order = 5
  nyquist = rate // 2

  if fmin > new_fmin and fmax == new_fmax:
    low = fmin / nyquist
    b, a = butter(nth_order, low, btype="high")
    audio_signal = lfilter(b, a, audio_signal)  # type: ignore

  elif fmin == new_fmin and fmax < new_fmax:
    high = fmax / nyquist
    b, a = butter(nth_order, high, btype="low")
    audio_signal = lfilter(b, a, audio_signal)  # type: ignore

  elif fmin > new_fmin and fmax < new_fmax:
    low = fmin / nyquist
    high = fmax / nyquist
    b, a = butter(nth_order, [low, high], btype="band")
    audio_signal = lfilter(b, a, audio_signal)  # type: ignore

  sig_f32 = audio_signal.astype(np.float32)
  return sig_f32


def fillup_with_silence(
  audio_segment: Float32Array, target_length: int
) -> Float32Array:
  current_length = len(audio_segment)
  assert current_length <= target_length

  if current_length == target_length:
    return audio_segment

  silence_length = target_length - current_length
  silence = np.zeros(silence_length, dtype=audio_segment.dtype)
  filled_segment = np.concatenate((audio_segment, silence))

  return filled_segment


def flat_sigmoid_logaddexp_fast(
  x: npt.NDArray, sensitivity: float, clip_val: float = 15.0, bias: float = 1.0
) -> npt.NDArray:
  transformed_bias = (bias - 1.0) * 10.0
  y = sensitivity * np.clip(x + transformed_bias, -clip_val, clip_val)

  positive_mask = y >= 0
  abs_y = np.abs(y)
  exp_neg_abs = np.exp(-abs_y, dtype=x.dtype)

  one_plus_exp = 1.0 + exp_neg_abs

  return np.where(positive_mask, exp_neg_abs / one_plus_exp, 1.0 / one_plus_exp)


def flat_logit_fast(x: npt.NDArray) -> npt.NDArray:
  """Invert a plain sigmoid: the logit each score came from, ``log(p) - log1p(-p)``.

  Saturated scores (0.0 and 1.0) give ``-inf`` and ``inf``, which
  ``flat_sigmoid_logaddexp_fast`` clips like any logit beyond ``clip_val``.
  """
  with np.errstate(divide="ignore"):
    logits: npt.NDArray = np.log(x) - np.log1p(-x)
  return logits


def flat_softmax_fast(x: npt.NDArray) -> npt.NDArray:
  x_max = np.max(x, axis=1, keepdims=True)
  shifted = x - x_max
  exp_shifted = np.exp(shifted, dtype=x.dtype)
  return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
