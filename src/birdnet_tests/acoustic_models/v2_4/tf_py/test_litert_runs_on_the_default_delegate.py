"""The 2.4 model on LiteRT really runs on its default delegate, XNNPACK.

Seen from outside by what the delegate takes away: a tensor inside the graph
cannot be read after a prediction. Every LiteRT release birdnet supports
delegates this file; a session that needs such a tensor, the embeddings, gets
an interpreter without the delegate.
"""

import numpy as np
import pytest

import birdnet
from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet.acoustic.models.v2_4.tf import AcousticTFBackendFP32V2_4
from birdnet_tests.helper import ensure_litert_or_skip

pytestmark = pytest.mark.litert


def test_a_prediction_is_delegated_and_embeddings_still_come_out() -> None:
  ensure_litert_or_skip()
  model = birdnet.load("acoustic", "2.4", "tf", library="litert")
  backend = AcousticTFBackendFP32V2_4(
    model.model_path, "CPU", False, inference_library="litert"
  )
  backend.load()
  segment = np.zeros((1, AcousticModelV2_4.get_segment_size_samples()), np.float32)

  scores = backend.predict(segment)
  with pytest.raises(ValueError, match=r"Tensor data is null"):
    backend._interp.get_tensor(backend.encoding_out_idx())
  embeddings = backend.encode(segment)

  assert scores.shape == (1, AcousticModelV2_4.get_n_classes())
  assert embeddings.shape == (1, AcousticModelV2_4.get_embeddings_dim())
