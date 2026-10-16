"""A sigmoid sensitivity on a model with the sigmoid in its graph, end to end.

The real exports are slow to load and not bit-reproducible across runs, so the
worker's contract is pinned on a stub that returns known scores, driven through
each model class's own ``predict_session`` (the 2.4 class carries the Raven
exports, the 3.0 class every model on that base): the default leaves the
scores untouched to the bit, a non-default sensitivity transforms them as it
transforms a 2.4 model's logits, and the threshold sees the transformed scores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ordered_set import OrderedSet

from birdnet.acoustic.models.base import AcousticModelBase
from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet_tests.fake_acoustic_backend import (
  InGraphSigmoidFakeBackend,
  fake_model_path,
)
from birdnet_tests.helper import scores_at_sigmoid_sensitivity

pytestmark = pytest.mark.no_tf

_SAMPLE_RATE = 48_000
_RAW = InGraphSigmoidFakeBackend.scores()
# Where a shift of the logit moves a score strictly: outside, the clip at
# +-15 holds it, for a 2.4 model's logits just the same.
_STRICTLY_MOVED = (_RAW > 0.01) & (_RAW < 0.99)


@pytest.fixture(params=[AcousticModelV2_4, AcousticModelV3_0], ids=["2.4", "3.0"])
def model(request: pytest.FixtureRequest, tmp_path: Path) -> AcousticModelBase:
  n = InGraphSigmoidFakeBackend.n_species_out
  return request.param(
    model_path=fake_model_path(tmp_path),
    species_list=OrderedSet([f"Fake species {i}_Fake {i}" for i in range(n)]),
    is_custom_model=False,
    backend_type=InGraphSigmoidFakeBackend,
    backend_kwargs={},
  )


def _clip() -> tuple[np.ndarray, int]:
  rng = np.random.default_rng(7)
  samples = rng.standard_normal(_SAMPLE_RATE * 3).astype(np.float32) * 0.1
  return samples, _SAMPLE_RATE


def _run(model: AcousticModelBase, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
  """Scores and masks of the one segment, species in model order."""
  with model.predict_session(
    n_workers=1,
    top_k=None,
    **kwargs,
  ) as session:
    res = session.run_arrays(_clip())
  order = np.argsort(np.asarray(res.species_ids)[0, 0])
  return (
    np.asarray(res.species_probs)[0, 0][order],
    np.asarray(res.species_masked)[0, 0][order],
  )


@pytest.mark.parametrize(
  "kwargs",
  [
    {},
    {"sigmoid_sensitivity": 1.0},
    {"apply_sigmoid": False},
    {"apply_sigmoid": False, "sigmoid_sensitivity": 0.5},
  ],
  ids=["default", "explicit 1.0", "no sigmoid", "no sigmoid, sensitivity ignored"],
)
def test_default_sensitivity_leaves_the_scores_untouched(
  model: AcousticModelBase, kwargs: dict[str, object]
) -> None:
  scores, _ = _run(model, default_confidence_threshold=-float("inf"), **kwargs)

  assert scores.dtype == np.float32
  np.testing.assert_array_equal(scores, _RAW)


@pytest.mark.parametrize("sensitivity", [0.5, 1.5])
def test_non_default_sensitivity_transforms_the_recovered_logits(
  model: AcousticModelBase, sensitivity: float
) -> None:
  scores, _ = _run(
    model,
    sigmoid_sensitivity=sensitivity,
    default_confidence_threshold=-float("inf"),
  )

  assert scores.dtype == np.float32
  # A score next to 1 holds few bits of its logit: 1e-5 of slack at 0.5.
  np.testing.assert_allclose(
    scores, scores_at_sigmoid_sensitivity(_RAW, sensitivity), atol=2e-5
  )
  if sensitivity < 1.0:
    assert np.all(scores[_STRICTLY_MOVED] < _RAW[_STRICTLY_MOVED])
  else:
    assert np.all(scores[_STRICTLY_MOVED] > _RAW[_STRICTLY_MOVED])


def test_threshold_applies_to_the_transformed_scores(
  model: AcousticModelBase,
) -> None:
  threshold = 0.5

  scores, masked = _run(
    model, sigmoid_sensitivity=0.5, default_confidence_threshold=threshold
  )
  _, masked_at_default = _run(model, default_confidence_threshold=threshold)

  np.testing.assert_array_equal(masked, scores < threshold)
  # The sensitivity moved scores below the threshold that cleared it before.
  assert 0 < (~masked).sum() < (~masked_at_default).sum()


def test_half_precision_casts_the_recovered_logits(model: AcousticModelBase) -> None:
  scores, _ = _run(
    model,
    sigmoid_sensitivity=0.5,
    half_precision=True,
    default_confidence_threshold=-float("inf"),
  )

  assert scores.dtype == np.float16
  np.testing.assert_allclose(
    scores, scores_at_sigmoid_sensitivity(_RAW, 0.5), atol=3e-3
  )
