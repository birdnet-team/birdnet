"""The largest worker count whose estimated peak fits, within one and the cores."""

from __future__ import annotations

import pytest

from birdnet.acoustic.inference.memory import fit_n_workers

pytestmark = pytest.mark.no_tf

_GIB = 2**30
_MIB = 2**20


def test_shrinks_to_what_fits() -> None:
  # (8 GiB - 256 MiB reserve) * 0.9 = 7142 MiB, six workers of 1 GiB.
  assert (
    fit_n_workers(
      n_cores=32,
      available_bytes=8 * _GIB,
      worker_bytes=_GIB,
      slot_bytes=0,
      prefetch_ratio=1,
    )
    == 6
  )


def test_never_more_than_the_cores() -> None:
  assert (
    fit_n_workers(
      n_cores=4,
      available_bytes=1000 * _GIB,
      worker_bytes=_MIB,
      slot_bytes=0,
      prefetch_ratio=1,
    )
    == 4
  )


def test_never_fewer_than_one() -> None:
  assert (
    fit_n_workers(
      n_cores=8, available_bytes=0, worker_bytes=_GIB, slot_bytes=0, prefetch_ratio=1
    )
    == 1
  )


def test_ring_slots_count_per_worker() -> None:
  # Each worker brings (1 + prefetch_ratio) slots with it: 200 MiB against
  # 100 MiB per worker out of the same 7142 MiB.
  slot = 100 * _MIB
  with_prefetch = fit_n_workers(
    n_cores=64,
    available_bytes=8 * _GIB,
    worker_bytes=0,
    slot_bytes=slot,
    prefetch_ratio=1,
  )
  without = fit_n_workers(
    n_cores=64,
    available_bytes=8 * _GIB,
    worker_bytes=0,
    slot_bytes=slot,
    prefetch_ratio=0,
  )
  assert (with_prefetch, without) == (35, 64)
