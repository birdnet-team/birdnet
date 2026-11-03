"""The session's unprocessable-input set is the union of every producer's report.

Which producer reads which input depends on timing, so a session test cannot
pin this: one producer regularly drains every unreadable input while the other
waits on the ring.
"""

from types import SimpleNamespace

import pytest

from birdnet.acoustic.inference.resources import ProducerResources

pytestmark = pytest.mark.no_tf


def _resources(n_producers: int) -> ProducerResources:
  config = SimpleNamespace(
    start_method="spawn", processing_conf=SimpleNamespace(producers=n_producers)
  )
  return ProducerResources.create(config)  # type: ignore[arg-type]


def test_every_producers_report_is_kept() -> None:
  resources = _resources(3)

  resources.store_unprocessed_inputs([{1}, set(), {3, 1}])

  assert resources.unprocessed_inputs == {1, 3}


def test_reset_forgets_the_last_runs_reports() -> None:
  resources = _resources(2)
  resources.store_unprocessed_inputs([{0}, {2}])

  resources.reset()

  with pytest.raises(AssertionError):
    _ = resources.unprocessed_inputs
