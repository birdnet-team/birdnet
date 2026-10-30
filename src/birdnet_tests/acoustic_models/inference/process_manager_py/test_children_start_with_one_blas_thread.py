"""Every process the pipeline starts gets one OpenBLAS thread, and the parent
keeps the environment it had.

Recorded in the parent as each ``start()`` runs, which is when the child's
environment is fixed. The known roles are checked by name, and every start the
session makes is checked whatever its name, so a new start site is covered too.
"""

from __future__ import annotations

import os
from multiprocessing.process import BaseProcess

import pytest

from birdnet.core.child_env import OPENBLAS_THREADS_VAR
from birdnet_tests.fake_acoustic_backend import fake_predict_session, write_marked_audio

pytestmark = pytest.mark.no_tf

_ROLES = ("PerformanceTracker", "Producer", "Worker")
_AFTER = "after the session"


@pytest.fixture(scope="module")
def seen(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[str | None]]:
  """The parent's value at each child's start, by role, and after the session."""
  values: dict[str, list[str | None]] = {}
  original = BaseProcess.start

  def recording_start(self: BaseProcess) -> None:
    role = next((r for r in _ROLES if f"-{r}" in self.name), self.name)
    values.setdefault(role, []).append(os.environ.get(OPENBLAS_THREADS_VAR))
    original(self)

  tmp_path = tmp_path_factory.mktemp("one_blas_thread")
  wav = write_marked_audio(tmp_path / "in.wav", 4)
  with pytest.MonkeyPatch.context() as patch:
    patch.delenv(OPENBLAS_THREADS_VAR, raising=False)
    patch.setattr(BaseProcess, "start", recording_start)
    with fake_predict_session(
      tmp_path, n_workers=2, n_producers=2, show_stats="progress"
    ) as session:
      session.run(str(wav))
    values[_AFTER] = [os.environ.get(OPENBLAS_THREADS_VAR)]
  return values


@pytest.mark.parametrize(
  ("role", "n_started"), [("PerformanceTracker", 1), ("Producer", 2), ("Worker", 2)]
)
def test_each_child_starts_with_one_blas_thread(
  seen: dict[str, list[str | None]], role: str, n_started: int
) -> None:
  assert seen.get(role) == ["1"] * n_started, seen


def test_every_start_had_one_blas_thread(seen: dict[str, list[str | None]]) -> None:
  starts = [v for role, values in seen.items() if role != _AFTER for v in values]
  assert starts, seen
  assert all(v == "1" for v in starts), seen


def test_the_parent_keeps_its_environment(seen: dict[str, list[str | None]]) -> None:
  assert seen[_AFTER] == [None], seen
