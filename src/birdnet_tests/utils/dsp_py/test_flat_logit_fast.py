"""The logit recovered from a sigmoid score feeds the pipeline sigmoid as a logit would.

A model with the sigmoid in its graph hands the prediction worker scores, and
a sigmoid sensitivity needs the logits behind them. Recovering them is only
worth anything if the pipeline sigmoid then lands where it lands on the
logits a 2.4 model outputs, at every sensitivity and over the whole range a
float32 sigmoid can produce.
"""

import warnings

import numpy as np
import pytest

from birdnet.utils.dsp import flat_logit_fast, flat_sigmoid_logaddexp_fast

pytestmark = pytest.mark.no_tf


def _graph_sigmoid(logits: np.ndarray) -> np.ndarray:
  """A plain float32 sigmoid, as the exports apply it inside their graph."""
  with np.errstate(over="ignore"):
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def _pipeline_sigmoid(logits: np.ndarray, sensitivity: float) -> np.ndarray:
  return flat_sigmoid_logaddexp_fast(logits, sensitivity=-1.0, bias=sensitivity)


# The range a float32 sigmoid does not saturate in, densely, plus the saturated
# tails on both sides (scores of exactly 0.0 below -88.7 and 1.0 above 16.6).
_LOGITS = np.concatenate(
  [
    np.linspace(-120.0, -20.0, 100_001, dtype=np.float32),
    np.linspace(-20.0, 20.0, 2_000_001, dtype=np.float32),
    np.linspace(20.0, 120.0, 100_001, dtype=np.float32),
  ]
)
_SCORES = _graph_sigmoid(_LOGITS)


def test_the_graph_sigmoid_of_the_recovered_logit_is_the_score() -> None:
  recovered = flat_logit_fast(_SCORES)

  inside = (_SCORES > 0.0) & (_SCORES < 1.0)
  assert np.isfinite(recovered[inside]).all()
  # A recovered logit of magnitude ~90 (a score of ~1e-39) carries a float32
  # error of ~1e-5, which is the relative error of the score it maps back to.
  np.testing.assert_allclose(
    _graph_sigmoid(recovered[inside]), _SCORES[inside], rtol=1e-5, atol=0.0
  )


# Measured over _LOGITS in float32: 1.27e-5 at 0.5 and 1.2e-6 at 0.75, where
# the shift moves scores near 1 (which hold few bits of their logit) onto the
# sigmoid's steep part; at most 2.4e-7 from 1.0 upwards.
@pytest.mark.parametrize(
  ("sensitivity", "max_abs_deviation"),
  [(0.5, 2e-5), (0.75, 3e-6), (1.0, 5e-7), (1.25, 5e-7), (1.5, 5e-7)],
)
def test_matches_the_logit_path(sensitivity: float, max_abs_deviation: float) -> None:
  from_logits = _pipeline_sigmoid(_LOGITS, sensitivity)
  from_scores = _pipeline_sigmoid(flat_logit_fast(_SCORES), sensitivity)

  assert from_scores.dtype == np.float32
  deviation = np.abs(from_scores.astype(np.float64) - from_logits.astype(np.float64))
  assert deviation.max() <= max_abs_deviation


@pytest.mark.parametrize("sensitivity", [0.5, 1.0, 1.5])
def test_saturated_scores_land_on_the_clip(sensitivity: float) -> None:
  saturated = np.array([0.0, 1.0], dtype=np.float32)
  beyond_clip = np.array([-100.0, 100.0], dtype=np.float32)

  recovered = flat_logit_fast(saturated)

  np.testing.assert_array_equal(recovered, [-np.inf, np.inf])
  np.testing.assert_array_equal(
    _pipeline_sigmoid(recovered, sensitivity),
    _pipeline_sigmoid(beyond_clip, sensitivity),
  )


def test_is_silent_on_saturated_scores() -> None:
  # The worker runs this on every batch; a warning per saturated score would
  # flood the session log.
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    flat_logit_fast(np.array([0.0, 0.5, 1.0], dtype=np.float32))


def test_preserves_dtype() -> None:
  assert flat_logit_fast(np.array([0.5], dtype=np.float32)).dtype == np.float32
  assert flat_logit_fast(np.array([0.5], dtype=np.float64)).dtype == np.float64
