"""The geo v3.0 TF backends address model outputs by raw tensor index.

Those indices are hardcoded per precision and change with every model export, so
a release bump can silently point them at the wrong tensor. Running the model to
find out needs TensorFlow 2.18/2.19 (its .tflite files carry select TF ops), which
no CI lane has - but *reading* the output details only needs an interpreter that
constructs, which any TensorFlow new enough to parse the model can do. That is
enough to pin the constants, and it covers every lane except the macOS Intel one
(pinned to TF <2.17, too old to open the INT8 export at all).
"""

import pytest

from birdnet.core.backends import load_tf_model, tf_installed
from birdnet.geo.models.v3_0.tf import (
  GeoTFBackendFP16V3_0,
  GeoTFBackendFP32V3_0,
  GeoTFBackendInt8V3_0,
  GeoTFDownloaderV3_0,
)
from birdnet.globals import (
  MODEL_PRECISION_FP16,
  MODEL_PRECISION_FP32,
  MODEL_PRECISION_INT8,
  MODEL_PRECISIONS,
)
from birdnet_tests.helper import ensure_tf_2_18_or_skip

_BACKENDS = {
  MODEL_PRECISION_INT8: GeoTFBackendInt8V3_0,
  MODEL_PRECISION_FP16: GeoTFBackendFP16V3_0,
  MODEL_PRECISION_FP32: GeoTFBackendFP32V3_0,
}


@pytest.mark.load_model
@pytest.mark.parametrize("precision", list(_BACKENDS))
def test_in_and_out_idx_match_the_model(precision: MODEL_PRECISIONS) -> None:
  if not tf_installed():
    pytest.skip("TensorFlow is not available")
  ensure_tf_2_18_or_skip()

  model_path, species_list = GeoTFDownloaderV3_0.get_model_path_and_labels(
    "en_us", precision
  )
  backend = _BACKENDS[precision]

  interpreter = load_tf_model(model_path, "tflite", allocate_tensors=True)
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  assert len(input_details) == 1
  assert input_details[0]["index"] == backend.in_idx()

  # The geo model has a single head, so the prediction is its only output.
  assert len(output_details) == 1
  assert output_details[0]["index"] == backend.prediction_out_idx()
  assert not backend.supports_encoding()

  # ... and it must have one output value per species in the shipped label file.
  assert output_details[0]["shape"][1] == len(species_list)
