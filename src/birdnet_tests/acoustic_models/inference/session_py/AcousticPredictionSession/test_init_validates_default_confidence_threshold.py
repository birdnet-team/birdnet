from pathlib import Path

import numpy as np
import pytest

from birdnet_tests.fake_acoustic_backend import fake_predict_session


@pytest.mark.no_tf
def test_non_numeric_threshold_is_rejected_at_construction(tmp_path: Path) -> None:
  with pytest.raises(TypeError, match=r"default confidence threshold must be a number"):
    fake_predict_session(
      tmp_path,
      default_confidence_threshold="0.5",  # type: ignore[arg-type]
    )


@pytest.mark.no_tf
@pytest.mark.parametrize(
  "threshold", [None, -np.inf, 0, 0.5, np.float32(0.5), np.int64(0)]
)
def test_numeric_or_missing_threshold_is_accepted(
  tmp_path: Path, threshold: float | None
) -> None:
  fake_predict_session(tmp_path, default_confidence_threshold=threshold)


@pytest.mark.no_tf
@pytest.mark.parametrize(
  ("threshold", "stored_type"),
  [(0, int), (0.5, float), (np.float32(0.5), float), (np.int64(0), int)],
)
def test_threshold_reaches_the_run_metadata_as_given(
  tmp_path: Path, threshold: float, stored_type: type
) -> None:
  # The benchmark statistics serialize this value with json.dump.
  session = fake_predict_session(tmp_path, default_confidence_threshold=threshold)
  stored = session._specific_config.default_confidence_threshold
  assert stored == threshold
  assert type(stored) is stored_type
