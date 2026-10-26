"""The prediction CSV progress bar ends at the row count, with the size shown."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from tqdm import tqdm

import birdnet.acoustic.inference.core.result_base as result_base
from birdnet_tests.acoustic_models.inference.predictions.prediction_result_py.test_prediction_to_structured_array import (  # noqa: E501
  create_file_prediction_result,
)

pytestmark = pytest.mark.no_tf


@pytest.mark.parametrize("buffer_size_kb", [1, 1024])
def test_progress_bar_reaches_the_row_count(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, buffer_size_kb: int
) -> None:
  bars: list[tqdm] = []

  def recording_tqdm(*args: Any, **kwargs: Any) -> tqdm:  # noqa: ANN401
    bar = tqdm(*args, file=io.StringIO(), **kwargs)
    bars.append(bar)
    return bar

  monkeypatch.setattr(result_base, "tqdm", recording_tqdm)
  result = create_file_prediction_result(
    n_files=2, duration_s=3000, top_k=5, segment_duration_s=3.0, overlap_duration_s=0.0
  )
  out = tmp_path / "predictions.csv"

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

  monkeypatch.setattr(result_base, "tqdm", recording_tqdm)
  result = create_file_prediction_result(
    n_files=1, duration_s=6, top_k=2, segment_duration_s=3.0, overlap_duration_s=0.0
  )
  result._species_masked[:] = True

  result.to_csv(tmp_path / "predictions.csv")

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
