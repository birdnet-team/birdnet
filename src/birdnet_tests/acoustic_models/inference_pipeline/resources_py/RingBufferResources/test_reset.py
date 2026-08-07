import numpy as np

from birdnet.acoustic.inference.resources import (
  RingBufferResources,
)
from birdnet.core.base import get_session_id


def _create() -> RingBufferResources:
  return RingBufferResources._create(
    get_session_id(), 4, 1, 48_000 * 3, np.dtype(np.uint32), 1000, "spawn"
  )


def test_reset_drains_leftover_wake_up_permits() -> None:
  """The shutdown hand-off leaves one permit outstanding after every run.

  Every run starts with an empty ring, so carrying a permit into the next one
  lets a worker acquire it, scan an empty ring and hit the "sem_fill was
  available" assertion. That surfaces as an intermittent `Analysis was
  cancelled` a few runs into a reused session, so it is pinned here rather
  than left to a timing-dependent integration test.
  """
  result = _create()
  result.sem_filled_slots.release()
  result.sem_filled_slots.release()
  assert result.sem_filled_slots.get_value() == 2

  result.reset()

  assert result.sem_filled_slots.get_value() == 0


def test_reset_is_a_noop_when_no_permits_are_outstanding() -> None:
  result = _create()
  assert result.sem_filled_slots.get_value() == 0

  result.reset()

  assert result.sem_filled_slots.get_value() == 0
