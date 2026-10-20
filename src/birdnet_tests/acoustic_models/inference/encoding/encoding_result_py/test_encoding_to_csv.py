"""Embedding CSV export: the value format, and a progress bar that reaches the end."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tqdm import tqdm

import birdnet.acoustic.inference.core.encoding.encoding_result as encoding_result
from birdnet_tests.acoustic_models.inference.encoding.encoding_result_py.test_encoding_to_structured_array import (  # noqa: E501
  create_file_encoding_result,
)

pytestmark = pytest.mark.no_tf


@pytest.mark.parametrize(
  ("dtype", "expected"),
  [
    (np.float16, "0.099976,-1.250000,1235.000000"),
    (np.float32, "0.100000,-1.250000,1234.567749"),
    (np.float64, "0.100000,-1.250000,1234.567800"),
  ],
)
def test_embedding_values_are_written_with_six_decimals(
  tmp_path: Path, dtype: type, expected: str
) -> None:
  result = create_file_encoding_result(
    n_files=1,
    duration_s=6,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
    embedding_dim=3,
  )
  result._embeddings = np.tile(
    np.array([0.1, -1.25, 1234.5678], dtype=dtype), (1, result.max_n_segments, 1)
  )
  out = tmp_path / "embeddings.csv"

  result.to_csv(out, silent=True)

  rows = out.read_text(encoding="utf-8").splitlines()[1:]
  assert len(rows) == 2
  assert all(row.endswith(f',"{expected}"') for row in rows)


@pytest.mark.parametrize(
  ("buffer_size_kb", "embedding_dim"), [(1, 64), (1024, 64), (1, 512), (1024, 512)]
)
def test_progress_bar_reaches_the_row_count(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  buffer_size_kb: int,
  embedding_dim: int,
) -> None:
  bars: list[tqdm] = []

  def recording_tqdm(*args: Any, **kwargs: Any) -> tqdm:  # noqa: ANN401
    bar = tqdm(*args, file=io.StringIO(), **kwargs)
    bars.append(bar)
    return bar

  monkeypatch.setattr(encoding_result, "tqdm", recording_tqdm)
  result = create_file_encoding_result(
    n_files=2,
    duration_s=300,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
    embedding_dim=embedding_dim,
  )
  out = tmp_path / "embeddings.csv"

  result.to_csv(out, buffer_size_kb=buffer_size_kb)

  (bar,) = bars
  assert bar.n == bar.total == len(result.to_structured_array())
  assert bar.postfix == _flushed_size(out, buffer_size_kb)


def test_empty_result_shows_no_size(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  bars: list[tqdm] = []

  def recording_tqdm(*args: Any, **kwargs: Any) -> tqdm:  # noqa: ANN401
    bar = tqdm(*args, file=io.StringIO(), **kwargs)
    bars.append(bar)
    return bar

  monkeypatch.setattr(encoding_result, "tqdm", recording_tqdm)
  result = create_file_encoding_result(
    n_files=1,
    duration_s=6,
    segment_duration_s=3.0,
    overlap_duration_s=0.0,
    masked_segments=[(0, 0), (0, 1)],
  )

  result.to_csv(tmp_path / "embeddings.csv")

  (bar,) = bars
  assert bar.n == bar.total == 0
  assert bar.postfix is None


def _flushed_size(path: Path, buffer_size_kb: int) -> str:
  """The size the bar reports: the bytes of the blocks flushed inside the loop."""
  lines = path.read_text(encoding="utf-8").splitlines()[1:]
  flushed = block = 0
  for line in lines:
    block += len(line.encode("utf-8")) + 1
    if block >= buffer_size_kb * 1024:
      flushed += block
      block = 0
  return f"CSV={flushed / 1024**2:.0f} MB"
