from __future__ import annotations

import os
import threading

import pytest

from birdnet.core.child_env import OPENBLAS_THREADS_VAR, one_blas_thread_in_children

pytestmark = pytest.mark.no_tf


def test_set_inside_the_block_and_removed_after(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv(OPENBLAS_THREADS_VAR, raising=False)
  with one_blas_thread_in_children("spawn"):
    assert os.environ.get(OPENBLAS_THREADS_VAR) == "1"
  assert OPENBLAS_THREADS_VAR not in os.environ


@pytest.mark.parametrize("start_method", ["fork", "forkserver"])
def test_other_start_methods_are_left_alone(
  monkeypatch: pytest.MonkeyPatch, start_method: str
) -> None:
  monkeypatch.delenv(OPENBLAS_THREADS_VAR, raising=False)
  with one_blas_thread_in_children(start_method):
    assert OPENBLAS_THREADS_VAR not in os.environ
  assert OPENBLAS_THREADS_VAR not in os.environ


def test_removed_after_an_exception_in_the_block(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv(OPENBLAS_THREADS_VAR, raising=False)
  with (
    pytest.raises(RuntimeError, match="start failed"),
    one_blas_thread_in_children("spawn"),
  ):
    raise RuntimeError("start failed")
  assert OPENBLAS_THREADS_VAR not in os.environ


def test_a_value_the_user_set_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv(OPENBLAS_THREADS_VAR, "4")
  with one_blas_thread_in_children("spawn"):
    assert os.environ.get(OPENBLAS_THREADS_VAR) == "4"
  assert os.environ.get(OPENBLAS_THREADS_VAR) == "4"


def test_a_value_the_user_set_during_the_block_is_kept(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv(OPENBLAS_THREADS_VAR, raising=False)
  with one_blas_thread_in_children("spawn"):
    os.environ[OPENBLAS_THREADS_VAR] = "4"
  assert os.environ.get(OPENBLAS_THREADS_VAR) == "4"


def test_overlapping_blocks_keep_it_until_the_last_one_ends(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Two sessions starting children from two threads: the first to finish must
  not take the variable away from the other's start calls."""
  monkeypatch.delenv(OPENBLAS_THREADS_VAR, raising=False)
  entered = threading.Event()
  leave = threading.Event()

  def first() -> None:
    with one_blas_thread_in_children("spawn"):
      entered.set()
      leave.wait(10)

  thread = threading.Thread(target=first, name="first-block", daemon=True)
  thread.start()
  assert entered.wait(10)
  with one_blas_thread_in_children("spawn"):
    leave.set()
    thread.join(10)
    assert not thread.is_alive()
    assert os.environ.get(OPENBLAS_THREADS_VAR) == "1"
  assert OPENBLAS_THREADS_VAR not in os.environ
