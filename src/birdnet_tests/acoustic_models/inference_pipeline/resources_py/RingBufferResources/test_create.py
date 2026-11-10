from multiprocessing import shared_memory

import numpy as np
import pytest

from birdnet.acoustic.inference.resources import (
  RingBufferResources,
)
from birdnet.core.base import get_session_id, get_session_id_hash
from birdnet_tests.fake_acoustic_backend import COLLIDING_SESSION_IDS

pytestmark = pytest.mark.no_tf


def _create(session_id: str) -> RingBufferResources:
  return RingBufferResources._create(
    session_id, 4, 1, 48_000 * 3, np.dtype(np.uint32), 1000, "spawn"
  )


def _names(resources: RingBufferResources) -> list[str]:
  return [
    resources.rf_file_indices.name,
    resources.rf_segment_indices.name,
    resources.rf_audio_samples.name,
    resources.rf_batch_sizes.name,
    resources.rf_flags.name,
  ]


def _exists(name: str) -> bool:
  try:
    shm = shared_memory.SharedMemory(name=name, create=False)
  except FileNotFoundError:
    return False
  shm.close()
  return True


def test_ring_names_have_max_30_chars() -> None:
  # on macOS the limit is 31 including "/" at the start
  # e.g., this was too long:
  # /bn_ring_file_indices_53879_8340719616_1763049803980550000
  # /bn_ring_file_indices_53879_834 <- this is maximum
  allowed_max_length = 30

  assert max(len(name) for name in _names(_create(get_session_id()))) <= (
    allowed_max_length
  )


def test_sessions_whose_short_hashes_collide_get_their_own_ring_names() -> None:
  first, second = COLLIDING_SESSION_IDS
  assert get_session_id_hash(first) == get_session_id_hash(second)

  assert not set(_names(_create(first))) & set(_names(_create(second)))


def test_creating_the_resources_leaves_an_existing_segment_alone() -> None:
  # A segment under a session's name is a live session's memory. Unlinking it
  # hands that memory to the newcomer on POSIX, where unlink takes effect.
  session_id = get_session_id()
  name = _create(session_id).rf_flags.name
  owner = shared_memory.SharedMemory(name=name, create=True, size=8)
  try:
    owner.buf[:4] = b"mine"

    _create(session_id)

    attached = shared_memory.SharedMemory(name=name, create=False)
    try:
      assert bytes(attached.buf[:4]) == b"mine"
    finally:
      attached.close()
  finally:
    owner.close()
    owner.unlink()


def test_a_taken_name_fails_the_newcomer_and_not_the_owner() -> None:
  session_id = get_session_id()
  resources = _create(session_id)
  *before_the_taken_one, taken = _names(resources)
  owner = shared_memory.SharedMemory(name=taken, create=True, size=8)
  try:
    owner.buf[:4] = b"mine"

    with pytest.raises(FileExistsError), resources.shared_memory_context(session_id):
      pass

    assert bytes(owner.buf[:4]) == b"mine"
    assert _exists(taken)
    assert not any(_exists(name) for name in before_the_taken_one)
  finally:
    owner.close()
    owner.unlink()
