"""BirdNET-Analyzer's own test classifier: a 2.4 export that predates Keras 3.

It has pre-Keras-3 signature names and a hidden layer that shifts the
predictions to tensor 549. Downloaded on first use (`ensure_analyzer_fixture`)
rather than built, so its litert tests also run where TensorFlow is absent.
"""

from pathlib import Path

import numpy as np
import pytest

import birdnet
from birdnet.acoustic.models.custom_tflite import CustomTFLiteInfo
from birdnet.acoustic.models.v2_4.tf import AcousticTFBackendCustomV2_4
from birdnet.utils.helper import get_species_from_file
from birdnet_tests.custom_classifier_exports import (
  ANALYZER_FIXTURE_LABELS,
  ensure_analyzer_fixture,
)
from birdnet_tests.helper import ensure_litert_or_skip, ensure_validity_check_or_skip
from birdnet_tests.model_loader_py.acoustic.custom_exports.conftest import (
  N_SEGMENTS,
  embeddings,
  official_embeddings,
  scores_in_model_order,
)

EXPECTED = CustomTFLiteInfo("2.4", "replace", 2, False, True, 144_000)
OFFICIAL_EMBEDDINGS_ATOL = 1e-3


@pytest.fixture(scope="module")
def analyzer_fixture() -> tuple[Path, Path]:
  # Every test here opens the file in the validity-check subprocess.
  ensure_validity_check_or_skip()
  return ensure_analyzer_fixture()


def _assert_runs_like_the_official_model(
  analyzer_fixture: tuple[Path, Path], library: str
) -> None:
  model, labels = analyzer_fixture
  custom = birdnet.load_custom("acoustic", "auto", "tf", model, labels, library=library)
  assert custom.backend_type is AcousticTFBackendCustomV2_4
  assert list(custom.species_list) == ANALYZER_FIXTURE_LABELS

  scores = scores_in_model_order(custom)
  assert scores.shape == (1, N_SEGMENTS, len(ANALYZER_FIXTURE_LABELS))
  assert np.isfinite(scores).all()
  assert scores.min() >= 0.0
  assert scores.max() <= 1.0

  # The head overwrites the memory of the pooling tensor that holds the
  # embeddings, which is only readable once the interpreter preserves
  # intermediate tensors.
  np.testing.assert_allclose(
    embeddings(custom),
    official_embeddings("2.4", library),
    atol=OFFICIAL_EMBEDDINGS_ATOL,
  )


@pytest.mark.load_model
@pytest.mark.no_tf
def test_analyzer_fixture_download() -> None:
  model, labels = ensure_analyzer_fixture()
  assert model.is_file()
  assert list(get_species_from_file(labels)) == ANALYZER_FIXTURE_LABELS


def test_inspect_describes_the_analyzer_fixture(
  analyzer_fixture: tuple[Path, Path],
) -> None:
  assert birdnet.inspect_custom_tflite(analyzer_fixture[0]) == EXPECTED


@pytest.mark.litert
def test_inspect_the_analyzer_fixture_with_litert(
  analyzer_fixture: tuple[Path, Path],
) -> None:
  ensure_litert_or_skip()
  info = birdnet.inspect_custom_tflite(analyzer_fixture[0], library="litert")
  assert info == EXPECTED


def test_the_analyzer_fixture_predicts_and_encodes(
  analyzer_fixture: tuple[Path, Path],
) -> None:
  _assert_runs_like_the_official_model(analyzer_fixture, "tflite")


@pytest.mark.litert
def test_the_analyzer_fixture_predicts_and_encodes_with_litert(
  analyzer_fixture: tuple[Path, Path],
) -> None:
  ensure_litert_or_skip()
  _assert_runs_like_the_official_model(analyzer_fixture, "litert")
