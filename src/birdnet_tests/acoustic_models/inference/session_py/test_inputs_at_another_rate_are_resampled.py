"""Audio at another rate than the model's is resampled on the way in, file or array.

The producer cuts segments at the source rate and resamples each one alone
(``core/producer.py::get_segments_with_overlap``), so a constant segment keeps
its marker: a constant has only a DC component. The bandpass stays off because
the fake session's bounds equal the model's. The resampling arithmetic itself
is unit-covered in ``core/producer_py/test_resample_array.py`` and
``test_get_data_segments_with_overlap.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticEncodingResultBase,
)
from birdnet_tests.fake_acoustic_backend import (
  DEFAULT_EMB_DIM,
  fake_encode_session,
  segment_marker,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf

_MODEL_SAMPLE_RATE = 32_000
_INPUT_SAMPLE_RATE = 48_000
_SEGMENTS = 2


def _assert_every_marker_arrived(res: AcousticEncodingResultBase) -> None:
  embeddings = np.asarray(res.embeddings)
  assert embeddings.shape == (1, _SEGMENTS, DEFAULT_EMB_DIM)
  assert not np.asarray(res.embeddings_masked).any()
  for j in range(_SEGMENTS):
    # Measured: the stretch moves a constant by ~2e-10.
    np.testing.assert_allclose(embeddings[0, j, 0], segment_marker(j), atol=1e-6)


def test_file_and_array_at_another_rate_are_resampled(tmp_path: Path) -> None:
  path = write_marked_audio(
    tmp_path / "48k.wav", _SEGMENTS, sample_rate=_INPUT_SAMPLE_RATE
  )
  with fake_encode_session(
    tmp_path, n_workers=1, model_sample_rate=_MODEL_SAMPLE_RATE
  ) as session:
    from_file = session.run(path)
    from_array = session.run_arrays(sf.read(path))

  _assert_every_marker_arrived(from_file)
  _assert_every_marker_arrived(from_array)
