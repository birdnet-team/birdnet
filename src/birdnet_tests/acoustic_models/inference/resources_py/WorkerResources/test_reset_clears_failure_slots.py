"""A reused session starts its next run without the last run's failure text."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from birdnet.acoustic.inference.core.sync import read_failure, write_failure
from birdnet.acoustic.inference.resources import WorkerResources
from birdnet_tests.fake_acoustic_backend import FakeAcousticBackend

pytestmark = pytest.mark.no_tf


def test_reset_clears_every_workers_slot() -> None:
  config = SimpleNamespace(
    start_method="spawn",
    processing_conf=SimpleNamespace(workers=2, device="CPU"),
    model_conf=SimpleNamespace(
      path=Path("model"), backend_type=FakeAcousticBackend, backend_kwargs={}
    ),
  )
  resources = WorkerResources.create(config)  # type: ignore[arg-type]
  assert len(resources.failure_slots) == 2
  write_failure(resources.failure_slots[1], RuntimeError("left over"))
  assert read_failure(resources.failure_slots[1]) == "RuntimeError: left over"

  resources.reset()

  assert all(read_failure(slot) is None for slot in resources.failure_slots)
