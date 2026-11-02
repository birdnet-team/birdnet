"""``apply_softmax`` on the two model classes whose ``predict_session`` offers it.

Pinned on the fake backend: its raw scores (``linspace(0, 1, 32)`` plus the
segment marker) sum to about 16, so a worker that skips the softmax fails the
sum. The 3.0 class rejects the option (``test_v3_0_apply_softmax_raises_error``);
the real V2.4 model shows it once in
``test_tflite_fp32_softmax_probs_sum_to_one_per_segment``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ordered_set import OrderedSet

from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticPredictionResultBase,
)
from birdnet.acoustic.models.base import AcousticModelBase
from birdnet.acoustic.models.perch_v2.model import AcousticModelPerchV2
from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet_tests.fake_acoustic_backend import FakeAcousticBackend, fake_model_path

pytestmark = pytest.mark.no_tf

_SAMPLE_RATE = 48_000


@pytest.fixture(params=[AcousticModelV2_4, AcousticModelPerchV2], ids=["2.4", "perch"])
def model(request: pytest.FixtureRequest, tmp_path: Path) -> AcousticModelBase:
  n = FakeAcousticBackend.n_species_out
  species = OrderedSet([f"Fake species {i}_Fake {i}" for i in range(n)])
  return request.param.load(fake_model_path(tmp_path), species, FakeAcousticBackend, {})


def _clip() -> tuple[np.ndarray, int]:
  rng = np.random.default_rng(7)
  samples = rng.standard_normal(_SAMPLE_RATE * 3).astype(np.float32) * 0.1
  return samples, _SAMPLE_RATE


def _run(model: AcousticModelBase, **kwargs: object) -> AcousticPredictionResultBase:
  with model.predict_session(
    n_workers=1,
    apply_sigmoid=False,
    default_confidence_threshold=-float("inf"),
    **kwargs,  # type: ignore[arg-type]
  ) as session:
    return session.run_arrays(_clip())


def test_softmax_scores_sum_to_one_per_segment(model: AcousticModelBase) -> None:
  res = _run(model, apply_softmax=True, top_k=None)

  probs = np.asarray(res.species_probs)
  assert probs.shape == (1, 1, FakeAcousticBackend.n_species_out)
  assert np.all(probs > 0)
  assert np.all(probs <= 1)
  np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-5)


def _in_model_order(res: AcousticPredictionResultBase) -> np.ndarray:
  order = np.argsort(np.asarray(res.species_ids)[0, 0])
  return np.asarray(res.species_probs)[0, 0][order]


def test_softmax_keeps_the_ranking(model: AcousticModelBase) -> None:
  raw = _run(model, apply_softmax=False, top_k=1)
  softmax = _run(model, apply_softmax=True, top_k=1)
  np.testing.assert_array_equal(softmax.species_ids, raw.species_ids)

  # The whole order, not only the top species: the fake's scores are distinct,
  # so any transform that is not monotone shows up here.
  raw_all = _in_model_order(_run(model, apply_softmax=False, top_k=None))
  softmax_all = _in_model_order(_run(model, apply_softmax=True, top_k=None))
  np.testing.assert_array_equal(np.argsort(softmax_all), np.argsort(raw_all))
