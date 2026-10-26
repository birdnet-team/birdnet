"""A prediction file result stores each input file as its absolute path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ordered_set import OrderedSet

from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticFilePredictionResult,
)
from birdnet_tests.acoustic_models.inference.predictions.prediction_result_py.test_prediction_to_structured_array import (  # noqa: E501
  create_mock_scores_tensor,
)

pytestmark = pytest.mark.no_tf


def test_relative_paths_are_stored_absolute() -> None:
  files = [Path("file_0.wav"), Path("sub") / "file_1.wav"]
  shape = (len(files), 1, 1)
  tensor = create_mock_scores_tensor(
    np.zeros(shape, dtype=np.uint8),
    np.zeros(shape, dtype=np.float32),
    np.zeros(shape, dtype=bool),
  )

  result = AcousticFilePredictionResult(
    tensor=tensor,
    files=files,
    species_list=OrderedSet(["species_0"]),
    file_durations=np.full(len(files), 3.0, dtype=np.float32),
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
    speed=1.0,
    model_path=Path("/model/path"),
    model_fmin=0,
    model_fmax=15_000,
    model_sr=48_000,
    model_precision="fp32",
    model_version="v2.4",
  )

  assert result.inputs.tolist() == [str(file.absolute()) for file in files]
