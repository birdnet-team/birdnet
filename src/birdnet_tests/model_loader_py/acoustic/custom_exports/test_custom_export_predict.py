import numpy as np
import pytest

import birdnet
from birdnet_tests.custom_classifier_exports import ExportSpec
from birdnet_tests.helper import ensure_litert_or_skip
from birdnet_tests.model_loader_py.acoustic.custom_exports.conftest import (
  EXPORT_TEST_TIMEOUT_S,
  N_SEGMENTS,
  V2_4_APPEND,
  V2_4_REPLACE,
  V2_4_REPLACE_HIDDEN,
  V3_0_APPEND,
  V3_0_REPLACE,
  GetExport,
  load_export_unchecked,
  official_scores,
  scores_in_model_order,
)

pytestmark = pytest.mark.timeout(EXPORT_TEST_TIMEOUT_S)

TFLITE_EXPORTS = [
  V2_4_REPLACE,
  V2_4_APPEND,
  V2_4_REPLACE_HIDDEN,
  V3_0_REPLACE,
  V3_0_APPEND,
]

# A custom export embeds the official SavedModel's graph, which TFLite converts
# with slightly different op fusion than the official .tflite release.
OFFICIAL_SCORES_ATOL = 1e-3


@pytest.mark.parametrize("spec", TFLITE_EXPORTS, ids=lambda s: s.name)
def test_predict_returns_probabilities(get_export: GetExport, spec: ExportSpec) -> None:
  export = get_export(spec)
  model = birdnet.load_custom(
    "acoustic", spec.version, "tf", export.model, export.labels
  )
  scores = scores_in_model_order(model)
  assert scores.shape == (1, N_SEGMENTS, export.n_species)
  assert np.isfinite(scores).all()
  assert scores.min() >= 0.0
  assert scores.max() <= 1.0


@pytest.mark.parametrize("spec", [V2_4_APPEND, V3_0_APPEND], ids=lambda s: s.name)
def test_append_keeps_the_official_scores(
  get_export: GetExport, spec: ExportSpec
) -> None:
  custom = load_export_unchecked(get_export(spec))
  official = birdnet.load("acoustic", spec.version, "tf")
  n_official = len(official.species_list)
  assert list(custom.species_list)[:n_official] == list(official.species_list)

  np.testing.assert_allclose(
    scores_in_model_order(custom)[..., :n_official],
    official_scores(spec.version),
    atol=OFFICIAL_SCORES_ATOL,
  )


@pytest.mark.litert
@pytest.mark.parametrize("spec", [V2_4_APPEND, V3_0_REPLACE], ids=lambda s: s.name)
def test_predict_with_litert(get_export: GetExport, spec: ExportSpec) -> None:
  ensure_litert_or_skip()
  export = get_export(spec)
  model = load_export_unchecked(export, library="litert")
  scores = scores_in_model_order(model)
  assert scores.shape == (1, N_SEGMENTS, export.n_species)
  assert scores.min() >= 0.0
  assert scores.max() <= 1.0


def test_predict_fails_cleanly_when_the_unchecked_version_is_wrong(
  get_export: GetExport,
) -> None:
  export = get_export(V3_0_REPLACE)
  model = birdnet.load_custom(
    "acoustic", "2.4", "tf", export.model, export.labels, check_validity=False
  )
  with pytest.raises(RuntimeError, match=r"Analysis was cancelled"):
    scores_in_model_order(model)


def test_v2_4_export_accepts_batches(get_export: GetExport) -> None:
  model = load_export_unchecked(get_export(V2_4_REPLACE))
  np.testing.assert_allclose(
    scores_in_model_order(model, batch_size=2), scores_in_model_order(model), atol=1e-5
  )


def test_v3_0_export_rejects_batches(get_export: GetExport) -> None:
  model = load_export_unchecked(get_export(V3_0_REPLACE))
  with pytest.raises(
    ValueError, match=r"batch_size=2 is not supported .* static batch"
  ):
    model.predict_session(batch_size=2)


def test_v3_0_export_rejects_a_sigmoid_sensitivity(get_export: GetExport) -> None:
  model = load_export_unchecked(get_export(V3_0_REPLACE))
  with pytest.raises(ValueError, match=r"sigmoid_sensitivity is not supported"):
    model.predict_session(sigmoid_sensitivity=0.9)


def test_v3_0_export_scores_do_not_depend_on_apply_sigmoid(
  get_export: GetExport,
) -> None:
  model = load_export_unchecked(get_export(V3_0_REPLACE))
  # Two runs of one model differ by ~1e-7; a second sigmoid would move 0.50 to 0.62.
  np.testing.assert_allclose(
    scores_in_model_order(model, apply_sigmoid=False),
    scores_in_model_order(model),
    atol=1e-5,
  )
