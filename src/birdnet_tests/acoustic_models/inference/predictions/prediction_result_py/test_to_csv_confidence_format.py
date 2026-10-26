"""Prediction CSV export writes confidences with six decimals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from birdnet_tests.acoustic_models.inference.predictions.prediction_result_py.test_prediction_to_structured_array import (  # noqa: E501
  create_file_prediction_result,
)

pytestmark = pytest.mark.no_tf


@pytest.mark.parametrize(
  ("dtype", "expected"), [(np.float16, "0.099976"), (np.float32, "0.100000")]
)
def test_confidences_are_written_with_six_decimals(
  tmp_path: Path, dtype: type, expected: str
) -> None:
  result = create_file_prediction_result(
    n_files=1, duration_s=6, top_k=2, segment_duration_s=3.0, overlap_duration_s=0.0
  )
  result._species_probs = np.full(result._species_probs.shape, 0.1, dtype=dtype)
  out = tmp_path / "predictions.csv"

  result.to_csv(out, silent=True)

  rows = out.read_text(encoding="utf-8").splitlines()[1:]
  assert len(rows) == 4
  assert all(row.endswith(f",{expected}") for row in rows)
