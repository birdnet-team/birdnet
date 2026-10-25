from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from birdnet.acoustic.inference.session import (
  AcousticEncodingSession,
  AcousticPredictionSession,
)
from birdnet_tests.fake_acoustic_backend import (
  fake_encode_session,
  fake_predict_session,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf

_FACTORIES = pytest.mark.parametrize(
  "factory", [fake_predict_session, fake_encode_session], ids=["predict", "encode"]
)

_Factory = Callable[..., AcousticPredictionSession | AcousticEncodingSession]


def _two_files(tmp_path: Path) -> list[Path]:
  return [write_marked_audio(tmp_path / f"{i}.wav", 1) for i in range(2)]


@_FACTORIES
def test_more_files_than_max_n_files_are_rejected(
  tmp_path: Path, factory: _Factory
) -> None:
  files = _two_files(tmp_path)
  # Rejected before the pipeline is touched, so the session is never entered.
  session = factory(tmp_path, max_n_files=1)

  with pytest.raises(
    RuntimeError,
    match=r"^Number of input files \(2\) exceeds the maximum allowed \(1\)\.$",
  ):
    session.run(files)


@_FACTORIES
def test_exactly_max_n_files_are_accepted(tmp_path: Path, factory: _Factory) -> None:
  files = _two_files(tmp_path)

  with factory(tmp_path, n_workers=1, max_n_files=2) as session:
    result = session.run(files)

  assert result.n_inputs == 2
