import numpy as np
import pytest

import birdnet
from birdnet.acoustic.models.base import AcousticModelBase
from birdnet.acoustic.models.v2_4.pb import AcousticRavenBackendFP32V2_4
from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.acoustic.models.v3_0.pb import AcousticRavenBackendFP32V3_0
from birdnet.utils.helper import get_species_from_file
from birdnet_tests.helper import scores_at_sigmoid_sensitivity
from birdnet_tests.model_loader_py.acoustic.custom_exports.conftest import (
  EXPORT_TEST_TIMEOUT_S,
  N_SEGMENTS,
  V3_0_RAVEN,
  V3_0_REPLACE,
  GetExport,
  load_export_unchecked,
  load_unchecked,
  scores_in_model_order,
)
from birdnet_tests.test_files import TEST_FILES_DIR

pytestmark = pytest.mark.timeout(EXPORT_TEST_TIMEOUT_S)

# A Raven export of the same (trained) head as replace.tflite.
RAVEN_V2_4 = TEST_FILES_DIR / "custom_models" / "raven" / "CustomClassifier"
RAVEN_V2_4_LABELS = RAVEN_V2_4 / "labels" / "label_names.csv"
TFLITE_V2_4 = TEST_FILES_DIR / "custom_models" / "tf" / "replace.tflite"
TFLITE_V2_4_LABELS = TEST_FILES_DIR / "custom_models" / "tf" / "replace_Labels.txt"


def _load_raven_v3_0(
  get_export: GetExport, check_validity: bool = False
) -> AcousticModelV3_0:
  # The validity check loads the SavedModel in a subprocess (~9 s), so only the
  # test about loading asks for it. The species come from the export's own
  # Raven label file.
  export = get_export(V3_0_RAVEN)
  model = birdnet.load_custom(
    "acoustic",
    "3.0",
    "pb",
    export.model,
    export.model / "labels" / "label_names.csv",
    is_raven=True,
    check_validity=check_validity,
  )
  assert isinstance(model, AcousticModelV3_0)
  return model


def _load_raven_v2_4() -> AcousticModelBase:
  return birdnet.load_custom(
    "acoustic",
    "2.4",
    "pb",
    RAVEN_V2_4,
    RAVEN_V2_4_LABELS,
    is_raven=True,
    check_validity=False,
  )


def test_v3_0_raven_predicts_the_probabilities_of_the_tflite_export(
  get_export: GetExport,
) -> None:
  model = _load_raven_v3_0(get_export, check_validity=True)
  assert model.backend_type is AcousticRavenBackendFP32V3_0
  assert list(model.species_list) == list(
    get_species_from_file(get_export(V3_0_RAVEN).labels)
  )
  scores = scores_in_model_order(model)
  assert scores.shape == (1, N_SEGMENTS, 3)
  assert np.isfinite(scores).all()
  assert scores.min() >= 0.0
  assert scores.max() <= 1.0

  # The same head, exported as TFLite.
  tflite_model = load_export_unchecked(get_export(V3_0_REPLACE))
  np.testing.assert_allclose(scores, scores_in_model_order(tflite_model), atol=1e-3)


def test_v3_0_raven_loaded_as_the_official_layout_is_rejected(
  get_export: GetExport,
) -> None:
  export = get_export(V3_0_RAVEN)
  with pytest.raises(ValueError, match=r"has no 'serving_default' signature"):
    birdnet.load_custom("acoustic", "3.0", "pb", export.model, export.labels)


def test_v3_0_raven_applies_a_sigmoid_sensitivity(get_export: GetExport) -> None:
  model = _load_raven_v3_0(get_export)
  default = scores_in_model_order(model)
  scores = scores_in_model_order(model, sigmoid_sensitivity=0.5)
  np.testing.assert_allclose(
    scores, scores_at_sigmoid_sensitivity(default, 0.5), atol=1e-4
  )
  moved = (default > 0.01) & (default < 0.99)
  assert np.all(scores[moved] < default[moved])


def test_v3_0_raven_rejects_batches(get_export: GetExport) -> None:
  model = _load_raven_v3_0(get_export)
  with pytest.raises(
    ValueError, match=r"batch_size=2 is not supported .* static batch"
  ):
    model.predict_session(batch_size=2)


def test_v3_0_raven_cannot_encode(get_export: GetExport) -> None:
  model = _load_raven_v3_0(get_export)
  with pytest.raises(ValueError, match=r"does not support embeddings"):
    model.encode_session(n_workers=1)


def test_v2_4_raven_scores_are_the_models_probabilities() -> None:
  raven = _load_raven_v2_4()
  assert raven.backend_type is AcousticRavenBackendFP32V2_4
  tflite = load_unchecked(TFLITE_V2_4, TFLITE_V2_4_LABELS, "2.4")
  assert list(raven.species_list) == list(tflite.species_list)
  # The Raven graph applies the sigmoid the pipeline applies to the tflite logits.
  # SavedModel and TFLite backbones differ by up to ~2e-3 here; a second sigmoid
  # would lift a score of 0.03 to 0.51.
  scores = scores_in_model_order(raven)
  np.testing.assert_allclose(scores, scores_in_model_order(tflite), atol=5e-3)
  # And whatever apply_sigmoid says: two runs of one model differ by ~1e-7.
  np.testing.assert_allclose(
    scores_in_model_order(raven, apply_sigmoid=False), scores, atol=1e-5
  )


@pytest.mark.parametrize("sensitivity", [0.5, 1.5])
def test_v2_4_raven_sigmoid_sensitivity_matches_the_tflite_export(
  sensitivity: float,
) -> None:
  # The same head twice: as logits that get the pipeline's scaled sigmoid, and
  # as in-graph scores whose logits the pipeline recovers first. Same backbone
  # spread as at the default sensitivity.
  raven = _load_raven_v2_4()
  tflite = load_unchecked(TFLITE_V2_4, TFLITE_V2_4_LABELS, "2.4")
  np.testing.assert_allclose(
    scores_in_model_order(raven, sigmoid_sensitivity=sensitivity),
    scores_in_model_order(tflite, sigmoid_sensitivity=sensitivity),
    atol=5e-3,
  )


def test_v2_4_raven_rejects_a_softmax() -> None:
  raven = _load_raven_v2_4()
  with pytest.raises(ValueError, match=r"apply_softmax is not supported"):
    raven.predict_session(apply_sigmoid=False, apply_softmax=True)
