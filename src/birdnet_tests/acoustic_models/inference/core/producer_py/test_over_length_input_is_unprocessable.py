"""An input with more segments than the ring can index must not pass as valid.

The producer and the input analyzer both check this; the analyzer cancels the
run. The producer's own check is exercised directly here: it must mark the
input unprocessable and cancel, instead of returning as if the input had zero
segments, which the per-file completion marker would report as a valid result.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import pytest

from birdnet.acoustic.inference.core.producer import Producer

pytestmark = pytest.mark.no_tf

_SAMPLE_RATE = 48_000
_SEGMENT_S = 3.0


class _SharedValue:
  value = 0


def _bare_producer(
  monkeypatch: pytest.MonkeyPatch, max_supported_segment_index: int
) -> Producer:
  """A producer with only the state ``get_segments_from_input`` reads.

  The real constructor wants a session's worth of queues, events and ring
  fields; none of them is touched on this path.
  """
  producer = Producer.__new__(Producer)
  producer._segment_duration_s = _SEGMENT_S
  producer._overlap_duration_s = 0.0
  producer._speed = 1.0
  producer._target_sample_rate = _SAMPLE_RATE
  producer.segment_duration_samples = int(_SEGMENT_S * _SAMPLE_RATE)
  producer._use_bandpass = False
  producer._max_segment_idx_ptr = _SharedValue()
  producer._max_supported_segment_index = max_supported_segment_index
  producer._cancel_event = threading.Event()  # type: ignore[assignment]
  producer._unprocessable_inputs = set()
  monkeypatch.setattr(Producer, "_logger", logging.getLogger(__name__))
  return producer


def _audio(n_segments: int) -> tuple[np.ndarray, int]:
  return np.zeros(int(n_segments * _SEGMENT_S * _SAMPLE_RATE), np.float32), _SAMPLE_RATE


def test_over_length_input_is_unprocessable_and_cancels_the_run(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  producer = _bare_producer(monkeypatch, max_supported_segment_index=2)

  segments = list(producer.get_segments_from_input(7, _audio(n_segments=4)))

  assert segments == []
  assert producer._unprocessable_inputs == {7}
  assert producer._cancel_event.is_set()


def test_input_within_the_limit_is_segmented(monkeypatch: pytest.MonkeyPatch) -> None:
  producer = _bare_producer(monkeypatch, max_supported_segment_index=2)

  segments = list(producer.get_segments_from_input(7, _audio(n_segments=3)))

  assert [index for index, _ in segments] == [0, 1, 2]
  assert producer._unprocessable_inputs == set()
  assert not producer._cancel_event.is_set()
