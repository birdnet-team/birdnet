"""The model-level ``predict(..)`` convenience over the long test file.

The 36 rows at threshold 0.1 are model content, so this stays on the real
model; the structured array's layout is pinned on fakes in
``inference/predictions/prediction_result_py/test_prediction_to_structured_array.py``.
"""

from birdnet.model_loader import load
from birdnet_tests.test_files import TEST_FILE_LONG


def test_predict_to_structured_array() -> None:
  model = load("acoustic", "2.4", "tf")
  result = model.predict([TEST_FILE_LONG], top_k=5, default_confidence_threshold=0.1)
  array = result.to_structured_array()
  assert array.shape == (36,)
  assert array.dtype.names == (
    "input",
    "start_time",
    "end_time",
    "species_name",
    "confidence",
  )
