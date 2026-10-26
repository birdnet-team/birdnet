"""Prediction CSV export: the exact header and rows, in the requested encoding."""

from __future__ import annotations

import os
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


def test_header_and_rows_are_written_exactly_in_the_given_encoding(
  tmp_path: Path,
) -> None:
  wav = tmp_path / 'rec "a", ü.wav'
  tensor = create_mock_scores_tensor(
    species_ids=np.array([[[0], [1]]], dtype=np.uint8),
    species_probs=np.array([[[0.5], [0.25]]], dtype=np.float32),
    species_masked=np.zeros((1, 2, 1), dtype=bool),
  )
  result = AcousticFilePredictionResult(
    tensor=tensor,
    files=[wav],
    species_list=OrderedSet(
      ["Turdus merula_Eurasian Blackbird", "Erithacus rubecula_European Robin"]
    ),
    file_durations=np.array([4.5], dtype=np.float32),
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
  out = tmp_path / "predictions.csv"

  result.to_csv(out, encoding="utf-16", silent=True)

  quoted_input = f'"{tmp_path}{os.sep}rec ""a"", ü.wav"'
  expected = (
    "input,start_time,end_time,species_name,confidence\n"
    f'{quoted_input},"00:00:00.00","00:00:03.00",'
    '"Turdus merula_Eurasian Blackbird",0.500000\n'
    f'{quoted_input},"00:00:03.00","00:00:04.50",'
    '"Erithacus rubecula_European Robin",0.250000\n'
  )
  assert out.read_bytes() == expected.replace("\n", os.linesep).encode("utf-16")
