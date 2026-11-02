"""The per-file completion callback (``on_file_complete``) on real V2.4 encoding.

The callback's behaviour is pinned on the fake backend in
``acoustic_models/inference/session_py/test_on_file_complete_encode.py``. This
one run shows the model wires it the same way and reports the model's shapes.
"""

import numpy as np

from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticFileEncodingResult,
)
from birdnet.model_loader import load
from birdnet_tests.helper import FileResultCollector
from birdnet_tests.test_files import TEST_FILE_LONG, TEST_FILE_SHORT


def test_tflite_fp32_fires_once_per_file_with_the_aggregate_slices() -> None:
  collector: FileResultCollector[AcousticFileEncodingResult] = FileResultCollector()
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  with model.encode_session(n_workers=1, on_file_complete=collector) as session:
    aggregate = session.run([str(TEST_FILE_SHORT), str(TEST_FILE_LONG)])

  assert len(collector.order) == 2
  assert set(collector.results) == {
    str(TEST_FILE_SHORT.absolute()),
    str(TEST_FILE_LONG.absolute()),
  }

  aggregate_inputs = list(aggregate.inputs)
  for key, per_file in collector.results.items():
    idx = aggregate_inputs.index(key)
    n_seg = per_file.embeddings.shape[1]
    np.testing.assert_array_equal(
      per_file.embeddings[0], aggregate.embeddings[idx, :n_seg]
    )
    np.testing.assert_array_equal(
      per_file.embeddings_masked[0], aggregate.embeddings_masked[idx, :n_seg]
    )

  short = collector.results[str(TEST_FILE_SHORT.absolute())]
  long = collector.results[str(TEST_FILE_LONG.absolute())]
  assert short.embeddings.shape == (1, 3, 1024)
  assert long.embeddings.shape == (1, 40, 1024)
