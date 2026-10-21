import csv
from pathlib import Path

import pytest
from ordered_set import OrderedSet

import birdnet_tests.fake_acoustic_backend as fake_backend
from birdnet.acoustic.inference.core.prediction.prediction_result import (
  VAR_SPECIES_NAME,
)
from birdnet_tests.fake_acoustic_backend import fake_predict_session, write_marked_audio

pytestmark = pytest.mark.no_tf


def _quoted_species(n: int) -> OrderedSet[str]:
  return OrderedSet([f'Sp "{i}"_Common, "{i}"' for i in range(n)])


def test_quotes_in_species_names_survive_a_csv_reader(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(fake_backend, "_species", _quoted_species)
  audio = write_marked_audio(tmp_path / "a.wav", 2)
  with fake_predict_session(tmp_path, n_workers=1) as session:
    result = session.run(audio)

  out = tmp_path / "out.csv"
  result.to_csv(out, silent=True)
  with out.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

  assert rows
  names = set(_quoted_species(fake_backend.DEFAULT_N_SPECIES))
  assert {row[VAR_SPECIES_NAME] for row in rows} <= names
  assert all(None not in row for row in rows), "a row has extra fields"
