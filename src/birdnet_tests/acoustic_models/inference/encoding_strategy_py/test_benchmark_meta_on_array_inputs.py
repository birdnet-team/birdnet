"""Statistics for an encoding of in-memory arrays, which have no file format.

``show_stats`` builds its summary from the result's inputs. For a file result
those are paths; for an array result they are indices, and listing the "file
formats" of an index used to raise after inference had already finished.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticDataEncodingResult,
)
from birdnet.globals import NA
from birdnet_tests.fake_acoustic_backend import fake_encode_session

pytestmark = pytest.mark.no_tf

_SAMPLE_RATE = 48_000


def _clip() -> tuple[np.ndarray, int]:
  rng = np.random.default_rng(7)
  samples = rng.standard_normal(_SAMPLE_RATE * 3).astype(np.float32) * 0.1
  return samples, _SAMPLE_RATE


def test_minimal_stats_on_arrays_returns_the_result(tmp_path: Path) -> None:
  with fake_encode_session(tmp_path, n_workers=1, show_stats="minimal") as session:
    result = session.run_arrays(_clip())

  assert isinstance(result, AcousticDataEncodingResult)
  assert result.embeddings.shape[0] == 1


def test_benchmark_meta_on_arrays_has_no_file_formats(tmp_path: Path) -> None:
  # "progress" tracks performance, which the full meta needs, without writing
  # a benchmark directory the way "benchmark" would.
  with fake_encode_session(tmp_path, n_workers=1, show_stats="progress") as session:
    result = session.run_arrays(_clip())
    strategy = session._strategy
    minimal = strategy.create_minimal_benchmark_meta(
      session._conf, session._specific_config, session._resources, result
    )
    full = strategy.create_full_benchmark_meta(
      session._conf, session._specific_config, session._resources, result
    )

  assert minimal.file_formats == NA
  assert full.file_formats == NA
