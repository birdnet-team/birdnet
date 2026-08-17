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

from birdnet.core.backends import onnxruntime_installed

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
