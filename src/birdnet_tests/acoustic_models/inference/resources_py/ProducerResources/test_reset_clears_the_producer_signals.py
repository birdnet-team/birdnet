"""A reused session's next run starts with no producer signal of the last run.

A worker whose wait for a filled slot times out ends its run once the
all-finished flag is set, so a flag left over from the last run lets the next
run's workers stop while the producers are still at work. A finish signal left
set, or its wake, ends the parent's wait for a producer that has not finished.
"""

import time
from types import SimpleNamespace

import pytest

from birdnet.acoustic.inference.core.sync import read_failure, write_failure
from birdnet.acoustic.inference.resources import ProducerResources

pytestmark = pytest.mark.no_tf


def test_reset_clears_every_producer_signal_of_the_last_run() -> None:
  config = SimpleNamespace(
    start_method="spawn", processing_conf=SimpleNamespace(producers=2)
  )
  resources = ProducerResources.create(config)  # type: ignore[arg-type]
  resources.all_finished.set()
  resources.n_finished_pointer.value = 2
  for signal in resources.finish_signals:
    signal.set()
  for start_signal in resources.start_signals:
    start_signal.release()
  for slot in resources.failure_slots:
    write_failure(slot, RuntimeError("left over"))

  resources.reset()

  assert not resources.all_finished.is_set(), "all-finished flag"
  assert resources.n_finished_pointer.value == 0, "finished-producer count"
  for i, signal in enumerate(resources.finish_signals):
    assert not signal.is_set(), f"producer {i} finish signal"
    begin = time.monotonic()
    assert not signal.wait(timeout=0.3)
    assert time.monotonic() - begin >= 0.2, f"producer {i} finish wake"
  for i, start_signal in enumerate(resources.start_signals):
    assert not start_signal.acquire(block=False), f"producer {i} start permits"
  for i, slot in enumerate(resources.failure_slots):
    assert read_failure(slot) is None, f"producer {i} failure slot"
