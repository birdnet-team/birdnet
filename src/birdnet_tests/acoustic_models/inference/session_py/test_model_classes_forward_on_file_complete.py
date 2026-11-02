"""Every model class forwards ``on_file_complete`` to its sessions, predict and encode.

The callback's behaviour is pinned on the fake backend in
``test_on_file_complete_predict.py`` and ``test_on_file_complete_encode.py``; one
run on the real V2.4 model shows the wiring in
``acoustic_models/v2_4/model_py/test_*/test_on_file_complete*_v2_4.py``. Here
each class builds its sessions on the fake, at its own segment length and
sample rate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ordered_set import OrderedSet

from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticFileEncodingResult,
)
from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticFilePredictionResult,
)
from birdnet.acoustic.models.base import AcousticModelBase
from birdnet_tests.fake_acoustic_backend import (
  FAKE_BACKEND_FOR_MODEL_CLASS,
  FakeAcousticBackend,
  fake_model_path,
  write_marked_audio,
)
from birdnet_tests.helper import FileResultCollector

pytestmark = pytest.mark.no_tf

_SHORT_SEGMENTS = 2
_LONG_SEGMENTS = 4


@pytest.fixture(params=list(FAKE_BACKEND_FOR_MODEL_CLASS), ids=["2.4", "3.0", "perch"])
def model(request: pytest.FixtureRequest, tmp_path: Path) -> AcousticModelBase:
  n = FakeAcousticBackend.n_species_out
  species = OrderedSet([f"Fake species {i}_Fake {i}" for i in range(n)])
  backend = FAKE_BACKEND_FOR_MODEL_CLASS[request.param]
  return request.param.load(fake_model_path(tmp_path), species, backend, {})


def _files(model: AcousticModelBase, tmp_path: Path) -> tuple[Path, Path]:
  sizes = {
    "segment_size_s": type(model).get_segment_size_s(),
    "sample_rate": type(model).get_sample_rate(),
  }
  return (
    write_marked_audio(tmp_path / "short.wav", _SHORT_SEGMENTS, **sizes),
    write_marked_audio(tmp_path / "long.wav", _LONG_SEGMENTS, **sizes),
  )


def _assert_each_file_once_with_its_aggregate_slice(
  collector: FileResultCollector,
  aggregate: AcousticFilePredictionResult | AcousticFileEncodingResult,
  short: Path,
  long: Path,
  arrays: tuple[str, ...],
) -> None:
  assert len(collector.order) == 2
  assert set(collector.results) == {str(short.absolute()), str(long.absolute())}

  aggregate_inputs = list(aggregate.inputs)
  for key, per_file in collector.results.items():
    idx = aggregate_inputs.index(key)
    n_seg = getattr(per_file, arrays[0]).shape[1]
    for name in arrays:
      np.testing.assert_array_equal(
        getattr(per_file, name)[0], getattr(aggregate, name)[idx, :n_seg], err_msg=name
      )

  short_result = collector.results[str(short.absolute())]
  long_result = collector.results[str(long.absolute())]
  assert getattr(short_result, arrays[0]).shape[1] == _SHORT_SEGMENTS
  assert getattr(long_result, arrays[0]).shape[1] == _LONG_SEGMENTS


def test_predict_session_fires_once_per_file(
  model: AcousticModelBase, tmp_path: Path
) -> None:
  short, long = _files(model, tmp_path)
  collector: FileResultCollector[AcousticFilePredictionResult] = FileResultCollector()
  with model.predict_session(n_workers=1, on_file_complete=collector) as session:
    aggregate = session.run([str(short), str(long)])

  _assert_each_file_once_with_its_aggregate_slice(
    collector,
    aggregate,
    short,
    long,
    ("species_probs", "species_ids", "species_masked"),
  )


def test_encode_session_fires_once_per_file(
  model: AcousticModelBase, tmp_path: Path
) -> None:
  short, long = _files(model, tmp_path)
  collector: FileResultCollector[AcousticFileEncodingResult] = FileResultCollector()
  with model.encode_session(n_workers=1, on_file_complete=collector) as session:
    aggregate = session.run([str(short), str(long)])

  _assert_each_file_once_with_its_aggregate_slice(
    collector, aggregate, short, long, ("embeddings", "embeddings_masked")
  )
