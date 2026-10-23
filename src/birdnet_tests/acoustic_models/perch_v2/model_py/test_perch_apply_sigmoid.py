from pathlib import Path

import pytest
from ordered_set import OrderedSet

from birdnet.acoustic.models.perch_v2.model import AcousticModelPerchV2
from birdnet_tests.fake_acoustic_backend import FakeAcousticBackend

pytestmark = pytest.mark.no_tf


def _model(tmp_path: Path) -> AcousticModelPerchV2:
  model_path = tmp_path / "model"
  model_path.write_bytes(b"fake")
  return AcousticModelPerchV2.load(
    model_path, OrderedSet(["species_a"]), FakeAcousticBackend, {}
  )


# Not entered: the fields are parent-side config.
def test_apply_sigmoid_runs_at_the_default_sensitivity(tmp_path: Path) -> None:
  """Perch's default sigmoid_sensitivity is None, which means the plain sigmoid."""
  session = _model(tmp_path).predict_session(top_k=None, apply_sigmoid=True)
  assert session._specific_config.apply_sigmoid is True
  assert session._specific_config.sigmoid_sensitivity == 1.0
