"""TensorFlow is an optional dependency: neither importing ``birdnet`` nor running
the onnx backend may import it. Each check runs in a fresh interpreter, so a
TensorFlow already imported by the test process (or by another test) cannot mask
a regression; with TensorFlow absent the checks are trivially true, which is why
they also run in the TensorFlow-equipped lanes.
"""

import subprocess
import sys
import textwrap

import pytest

from birdnet.core.backends import litert_installed, onnxruntime_installed, tf_installed
from birdnet_tests.helper import ensure_validity_check_or_skip

pytestmark = pytest.mark.no_tf


def _run_isolated(code: str) -> None:
  subprocess.run(
    [sys.executable, "-c", textwrap.dedent(code)],
    check=True,
    timeout=540,
  )


def test_import_birdnet_does_not_import_tensorflow() -> None:
  _run_isolated(
    """
    import sys
    import birdnet
    assert "tensorflow" not in sys.modules, "import birdnet imported tensorflow"
    """
  )


def test_custom_classifier_checks_do_not_import_tensorflow() -> None:
  """The validity check and the inspection open the file in a subprocess; the
  interpreter they need stays out of the caller, so an application can look
  at a file before deciding how to load it."""
  ensure_validity_check_or_skip()
  if tf_installed():
    library = "tflite"
  elif litert_installed():
    library = "litert"
  else:
    pytest.skip("neither TensorFlow nor ai_edge_litert is installed")
  _run_isolated(
    f"""
    import sys
    import birdnet
    model = "src/birdnet_tests/TEST_FILES/custom_models/tf/replace.tflite"
    labels = "src/birdnet_tests/TEST_FILES/custom_models/tf/replace_Labels.txt"
    birdnet.inspect_custom_tflite(model, library="{library}")
    birdnet.load_custom("acoustic", "2.4", "tf", model, labels, library="{library}")
    if {tf_installed()}:
      # The pb path checks validity by loading the SavedModel.
      raven = "src/birdnet_tests/TEST_FILES/custom_models/raven/CustomClassifier"
      birdnet.load_custom(
        "acoustic", "2.4", "pb", raven, raven + "/labels/label_names.csv", is_raven=True
      )
    assert "tensorflow" not in sys.modules, "the inspection imported tensorflow"
    assert "ai_edge_litert" not in sys.modules, "the inspection imported litert"
    """
  )


@pytest.mark.skipif(not onnxruntime_installed(), reason="onnxruntime not installed")
def test_acoustic_v3_onnx_prediction_does_not_import_tensorflow() -> None:
  _run_isolated(
    """
    import sys
    import birdnet
    model = birdnet.load("acoustic", "3.0", "onnx")
    predictions = model.predict("example/soundscape.wav")
    assert predictions.species_probs.shape[0] == 1
    assert "tensorflow" not in sys.modules, "the onnx backend imported tensorflow"
    """
  )
