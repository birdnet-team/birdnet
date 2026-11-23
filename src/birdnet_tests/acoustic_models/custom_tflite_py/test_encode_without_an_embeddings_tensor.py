"""Encoding with a custom classifier whose file has no embeddings tensor.

Loaded with ``check_validity=False`` such a file reaches a worker, and the
worker has to say what the file lacks.
"""

from pathlib import Path

import numpy as np
import pytest

from birdnet.acoustic.models.v2_4.tf import AcousticTFBackendCustomV2_4
from birdnet.core import backends
from birdnet_tests.acoustic_models.custom_tflite_py.stub_interpreter import (
  v2_4_export,
)

pytestmark = pytest.mark.no_tf


def test_encode_names_the_missing_embeddings_tensor(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  interp = v2_4_export(3)
  interp.others = []
  monkeypatch.setattr(backends, "load_tf_model", lambda *_args, **_kwargs: interp)
  backend = AcousticTFBackendCustomV2_4(
    Path("classifier.tflite"), "CPU", False, inference_library="litert"
  )
  backend.load()

  with pytest.raises(ValueError, match=r"exposes no embeddings tensor"):
    backend.encode(np.zeros((1, 144_000), dtype=np.float32))
