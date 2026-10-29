"""The worker's per-batch debug lines render their NumPy arrays only at DEBUG.

Each batch logs its segment indices twice. Rendering an array costs tens of
microseconds, paid for every batch in every worker if the arguments are
formatted before `logging` decides the record is below the level, so at the
default INFO level the count must be zero. Counted inside the worker process
by `ReprCountingBackend`; the DEBUG case is the positive control for the hook.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from birdnet_tests.fake_acoustic_backend import (
  fake_predict_session,
  write_marked_audio,
)
from birdnet_tests.repr_counting_backend import (
  COUNT_DIR_ENV_VAR,
  ReprCountingBackend,
  read_counts,
)

pytestmark = pytest.mark.no_tf

N_SEGMENTS = 6


@pytest.fixture
def package_logger() -> Iterator[logging.Logger]:
  # The session captures the `birdnet` logger's level when it is created and
  # hands it to its children; restore it so other tests see the default.
  logger = logging.getLogger("birdnet")
  previous = logger.level
  yield logger
  logger.setLevel(previous)


def _count_in_one_worker(tmp_path: Path, logger: logging.Logger, level: int) -> int:
  logger.setLevel(level)
  audio = write_marked_audio(tmp_path / "in.wav", N_SEGMENTS)
  with fake_predict_session(
    tmp_path, n_workers=1, backend=ReprCountingBackend
  ) as session:
    session.run([str(audio)])
  counts = read_counts(tmp_path)
  assert len(counts) == 1, counts
  return next(iter(counts.values()))


def test_no_array_is_rendered_at_info(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, package_logger: logging.Logger
) -> None:
  monkeypatch.setenv(COUNT_DIR_ENV_VAR, str(tmp_path))

  assert _count_in_one_worker(tmp_path, package_logger, logging.INFO) == 0


def test_arrays_are_rendered_at_debug(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, package_logger: logging.Logger
) -> None:
  monkeypatch.setenv(COUNT_DIR_ENV_VAR, str(tmp_path))

  # Two lines per batch carry the array; the last batch's second line may be
  # written after the backend's final per-predict count, before exit.
  assert _count_in_one_worker(tmp_path, package_logger, logging.DEBUG) >= N_SEGMENTS
