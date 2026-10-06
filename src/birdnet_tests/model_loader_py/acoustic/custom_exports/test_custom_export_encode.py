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
  embeddings,
  load_export_unchecked,
  load_unchecked,
  official_embeddings,
)
from birdnet_tests.test_files import TEST_FILES_DIR

pytestmark = pytest.mark.timeout(EXPORT_TEST_TIMEOUT_S)

EMBEDDINGS_DIM = {"2.4": 1024, "3.0": 1280}
OFFICIAL_EMBEDDINGS_ATOL = 1e-3

LEGACY_DIR = TEST_FILES_DIR / "custom_models" / "tf"
LEGACY_FIXTURES = [
  ("replace.tflite", "replace_Labels.txt"),
  ("append.tflite", "append_Labels.txt"),
  ("replace+hidden1024.tflite", "replace+hidden1024_Labels.txt"),
  ("append+hidden1024.tflite", "append+hidden1024_Labels.txt"),
]


@pytest.mark.parametrize(
  "spec",
  [V2_4_REPLACE, V2_4_APPEND, V2_4_REPLACE_HIDDEN, V3_0_REPLACE, V3_0_APPEND],
  ids=lambda s: s.name,
)
def test_encode_gives_the_official_embeddings(
  get_export: GetExport, spec: ExportSpec
) -> None:
  export = get_export(spec)
  custom = birdnet.load_custom(
    "acoustic", spec.version, "tf", export.model, export.labels
  )

  custom_embeddings = embeddings(custom)
  assert custom_embeddings.shape == (1, N_SEGMENTS, EMBEDDINGS_DIM[spec.version])
  np.testing.assert_allclose(
    custom_embeddings,
    official_embeddings(spec.version),
    atol=OFFICIAL_EMBEDDINGS_ATOL,
  )


# An append export's prediction head overwrites the memory of the pooling tensor
# that holds the embeddings, which is only readable once the interpreter
# preserves intermediate tensors.
@pytest.mark.parametrize(("model_name", "labels_name"), LEGACY_FIXTURES)
def test_encode_legacy_fixtures_give_the_official_embeddings(
  model_name: str, labels_name: str
) -> None:
  custom = load_unchecked(LEGACY_DIR / model_name, LEGACY_DIR / labels_name, "2.4")
  np.testing.assert_allclose(
    embeddings(custom), official_embeddings("2.4"), atol=OFFICIAL_EMBEDDINGS_ATOL
  )


def test_encode_accepts_batches_on_2_4(get_export: GetExport) -> None:
  custom = load_export_unchecked(get_export(V2_4_APPEND))
  np.testing.assert_allclose(
    embeddings(custom, batch_size=2), embeddings(custom), atol=1e-5
  )


@pytest.mark.litert
def test_encode_with_litert(get_export: GetExport) -> None:
  ensure_litert_or_skip()
  custom = load_export_unchecked(get_export(V2_4_APPEND), library="litert")
  np.testing.assert_allclose(
    embeddings(custom),
    official_embeddings("2.4", "litert"),
    atol=OFFICIAL_EMBEDDINGS_ATOL,
  )
