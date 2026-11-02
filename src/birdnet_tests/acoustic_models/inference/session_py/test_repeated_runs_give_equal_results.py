"""Repeated runs give equal results: twice on one session, and on two sessions alike.

Two workers, so the blocks arrive in whatever order the workers finish them.
The files have 3 and 5 segments, so the shorter one's rows 3 and 4 are padding
the tensor never writes (``core/tensor.py::_allocate``): the masks are compared
whole, the values only over each input's own segments, which is why this does
not go through ``helper.assert_*_result_is_equal``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticEncodingResultBase,
)
from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticPredictionResultBase,
)
from birdnet_tests.fake_acoustic_backend import (
  fake_encode_session,
  fake_predict_session,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf

_SEGMENTS = (3, 5)


def _files(tmp_path: Path) -> list[str]:
  return [str(write_marked_audio(tmp_path / f"{n}.wav", n)) for n in _SEGMENTS]


def _assert_defined_rows_equal(
  values: dict[str, tuple[np.ndarray, np.ndarray]], masks: tuple[np.ndarray, np.ndarray]
) -> None:
  np.testing.assert_array_equal(masks[0], masks[1], err_msg="mask")
  for name, (first, second) in values.items():
    assert first.shape == second.shape, name
    for i, n in enumerate(_SEGMENTS):
      np.testing.assert_array_equal(
        first[i, :n], second[i, :n], err_msg=f"{name}, input {i}"
      )


def _in_species_order(
  res: AcousticPredictionResultBase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """ids, scores and masks with the species axis sorted, as top-k may permute it."""
  ids = np.asarray(res.species_ids)
  order = np.argsort(ids, axis=-1)
  return (
    np.take_along_axis(ids, order, axis=-1),
    np.take_along_axis(np.asarray(res.species_probs), order, axis=-1),
    np.take_along_axis(np.asarray(res.species_masked), order, axis=-1),
  )


def _assert_predictions_equal(
  first: AcousticPredictionResultBase, second: AcousticPredictionResultBase
) -> None:
  assert list(first.inputs) == list(second.inputs)
  ids_a, probs_a, masks_a = _in_species_order(first)
  ids_b, probs_b, masks_b = _in_species_order(second)
  _assert_defined_rows_equal(
    {"species_ids": (ids_a, ids_b), "species_probs": (probs_a, probs_b)},
    (masks_a, masks_b),
  )


def _assert_encodings_equal(
  first: AcousticEncodingResultBase, second: AcousticEncodingResultBase
) -> None:
  assert list(first.inputs) == list(second.inputs)
  _assert_defined_rows_equal(
    {"embeddings": (np.asarray(first.embeddings), np.asarray(second.embeddings))},
    (np.asarray(first.embeddings_masked), np.asarray(second.embeddings_masked)),
  )


def test_predict_twice_on_one_session(tmp_path: Path) -> None:
  files = _files(tmp_path)
  with fake_predict_session(tmp_path, n_workers=2) as session:
    first = session.run(files)
    second = session.run(files)

  _assert_predictions_equal(first, second)


def test_predict_on_two_sessions(tmp_path: Path) -> None:
  files = _files(tmp_path)
  with fake_predict_session(tmp_path, n_workers=2) as session:
    first = session.run(files)
  with fake_predict_session(tmp_path, n_workers=2) as session:
    second = session.run(files)

  _assert_predictions_equal(first, second)


def test_encode_twice_on_one_session(tmp_path: Path) -> None:
  files = _files(tmp_path)
  with fake_encode_session(tmp_path, n_workers=2) as session:
    first = session.run(files)
    second = session.run(files)

  _assert_encodings_equal(first, second)


def test_encode_on_two_sessions(tmp_path: Path) -> None:
  files = _files(tmp_path)
  with fake_encode_session(tmp_path, n_workers=2) as session:
    first = session.run(files)
  with fake_encode_session(tmp_path, n_workers=2) as session:
    second = session.run(files)

  _assert_encodings_equal(first, second)
