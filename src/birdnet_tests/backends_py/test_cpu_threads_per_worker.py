import psutil
import pytest

from birdnet.core.backends import cpu_threads_per_worker

pytestmark = pytest.mark.no_tf


@pytest.mark.parametrize(
  ("n_cores", "n_workers", "expected"),
  [
    (8, 8, 1),
    (8, 4, 2),
    (8, 3, 2),
    (8, 1, 8),
    (8, 16, 1),
    (1, 1, 1),
  ],
)
def test_cores_are_split_between_workers(
  monkeypatch: pytest.MonkeyPatch, n_cores: int, n_workers: int, expected: int
) -> None:
  monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: n_cores)

  assert cpu_threads_per_worker(n_workers) == expected


def test_unknown_core_count_gives_one_thread(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: None)

  assert cpu_threads_per_worker(4) == 1
