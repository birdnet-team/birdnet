"""An official TFLite release is itself a valid input to `load_custom`.

There it runs on the custom backends, which resolve the tensors from the file
instead of from the official backends' fixed indices, so the scores and the
embeddings have to be the ones `load` gives for the same file.
"""

import numpy as np
import pytest

import birdnet
from birdnet.acoustic.models.base import AcousticModelBase
from birdnet.acoustic.models.v2_4.tf import AcousticTFBackendCustomV2_4
from birdnet.acoustic.models.v3_0.tf import (
  AcousticTFBackendCustomFP16V3_0,
  AcousticTFBackendCustomV3_0,
)
from birdnet.core.backends import VersionedAcousticBackendProtocol
from birdnet.utils.local_data import get_lang_dir
from birdnet_tests.helper import ensure_tf_2_18_or_skip, ensure_validity_check_or_skip
from birdnet_tests.model_loader_py.acoustic.custom_exports.conftest import (
  embeddings,
  official_embeddings,
  official_scores,
  scores_in_model_order,
)

# Two runs of one file differ by ~1e-7.
SAME_FILE_ATOL = 1e-5

CASES: list[tuple[str, str, type[VersionedAcousticBackendProtocol]]] = [
  ("2.4", "fp32", AcousticTFBackendCustomV2_4),
  ("3.0", "fp32", AcousticTFBackendCustomV3_0),
  ("3.0", "fp16", AcousticTFBackendCustomFP16V3_0),
]
CASE_IDS = [f"{version}-{precision}" for version, precision, _ in CASES]


def _official_and_custom(
  version: str, precision: str, requested_version: str
) -> tuple[AcousticModelBase, AcousticModelBase]:
  ensure_validity_check_or_skip()
  if version == "3.0":
    # The v3.0 exports need a newer TF than the macOS Intel pin (<2.17).
    ensure_tf_2_18_or_skip()
  official = birdnet.load("acoustic", version, "tf", precision=precision)
  custom = birdnet.load_custom(
    "acoustic",
    requested_version,
    "tf",
    official.model_path,
    get_lang_dir("acoustic", version, "tf") / "en_us.txt",  # type: ignore[arg-type]
    precision=precision,
  )
  return official, custom


@pytest.mark.parametrize(("version", "precision", "backend_type"), CASES, ids=CASE_IDS)
def test_official_tflite_through_load_custom_gives_the_official_results(
  version: str,
  precision: str,
  backend_type: type[VersionedAcousticBackendProtocol],
) -> None:
  official, custom = _official_and_custom(version, precision, version)
  assert custom.backend_type is backend_type
  assert list(custom.species_list) == list(official.species_list)

  if precision == "fp32":
    expected_scores = official_scores(version)
    expected_embeddings = official_embeddings(version)
  else:
    expected_scores = scores_in_model_order(official)
    expected_embeddings = embeddings(official)
  np.testing.assert_allclose(
    scores_in_model_order(custom), expected_scores, atol=SAME_FILE_ATOL
  )
  np.testing.assert_allclose(
    embeddings(custom), expected_embeddings, atol=SAME_FILE_ATOL
  )


@pytest.mark.parametrize(("version", "precision", "backend_type"), CASES, ids=CASE_IDS)
def test_auto_version_recognises_an_official_tflite(
  version: str,
  precision: str,
  backend_type: type[VersionedAcousticBackendProtocol],
) -> None:
  _, custom = _official_and_custom(version, precision, "auto")
  assert custom.get_version() == version
  assert custom.backend_type is backend_type
