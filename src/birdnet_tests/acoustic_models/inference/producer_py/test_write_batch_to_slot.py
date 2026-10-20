"""A batch lands in its ring slot as ``np.stack`` builds it; nothing else changes."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from birdnet.acoustic.inference.core.producer import Producer

pytestmark = pytest.mark.no_tf

_N_SLOTS = 3
_BATCH_SIZE = 4
_N_SAMPLES = 5
_UNTOUCHED = -1.0


def _bare_producer(monkeypatch: pytest.MonkeyPatch) -> Producer:
  """A producer with plain NumPy arrays where the shared-memory ring would be."""
  producer = Producer.__new__(Producer)
  producer._n_slots = _N_SLOTS
  producer._batch_size = _BATCH_SIZE
  producer._ring_audio_samples = np.full(
    (_N_SLOTS, _BATCH_SIZE, _N_SAMPLES), _UNTOUCHED, dtype=np.float32
  )
  producer._ring_file_indices = np.zeros((_N_SLOTS, _BATCH_SIZE), dtype=np.uint16)
  producer._ring_segment_indices = np.zeros((_N_SLOTS, _BATCH_SIZE), dtype=np.uint16)
  producer._ring_batch_sizes = np.zeros(_N_SLOTS, dtype=np.uint8)
  monkeypatch.setattr(Producer, "_logger", logging.getLogger(__name__))
  return producer


@pytest.mark.parametrize("batch_size", [_BATCH_SIZE, 2])
@pytest.mark.parametrize("sample_dtype", [np.float32, np.float64])
def test_batch_is_written_to_its_slot_only(
  monkeypatch: pytest.MonkeyPatch, batch_size: int, sample_dtype: type
) -> None:
  producer = _bare_producer(monkeypatch)
  rng = np.random.default_rng(0)
  batch = tuple(
    rng.standard_normal(_N_SAMPLES).astype(sample_dtype) for _ in range(batch_size)
  )
  slot = 1

  producer._write_batch_to_slot(
    slot, tuple(range(batch_size)), tuple(range(10, 10 + batch_size)), batch
  )

  ring = producer._ring_audio_samples
  assert ring is not None
  np.testing.assert_array_equal(
    ring[slot, :batch_size], np.stack(batch).astype(np.float32)
  )
  assert (ring[slot, batch_size:] == _UNTOUCHED).all()
  assert (ring[[0, 2]] == _UNTOUCHED).all()
  assert producer._ring_batch_sizes is not None
  assert producer._ring_batch_sizes[slot] == batch_size
