"""Embedding CSV export: the exact header and rows, in the requested encoding."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticFileEncodingResult,
)
from birdnet_tests.acoustic_models.inference.encoding.encoding_result_py.test_encoding_to_structured_array import (  # noqa: E501
  create_mock_encoding_tensor,
)

pytestmark = pytest.mark.no_tf


def test_header_and_rows_are_written_exactly_in_the_given_encoding(
  tmp_path: Path,
) -> None:
  wav = tmp_path / 'rec "a", ü.wav'
  tensor = create_mock_encoding_tensor(n_files=1, n_segments=2, embedding_dim=2)
  result = AcousticFileEncodingResult(
    tensor=tensor,
    files=[wav],
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
  out = tmp_path / "embeddings.csv"

  result.to_csv(out, encoding="utf-16", silent=True)

  quoted_input = f'"{tmp_path}{os.sep}rec ""a"", ü.wav"'
  expected = (
    "input,start_time,end_time,embedding\n"
    f'{quoted_input},"00:00:00.00","00:00:03.00","0.000000,1.000000"\n'
    f'{quoted_input},"00:00:03.00","00:00:04.50","2.000000,3.000000"\n'
  )
  assert out.read_bytes() == expected.replace("\n", os.linesep).encode("utf-16")
