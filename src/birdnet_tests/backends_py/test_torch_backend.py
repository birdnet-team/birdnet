"""Output handling of ``TorchBackend``.

The exported TorchScript modules differ in shape: the acoustic model returns a
tuple of (predictions, embeddings) with the activation already applied, while
the geo model returns a single tensor of logits. Both must end up as
probabilities in the same layout as the other backends of the same model.
"""

from pathlib import Path
from typing import Any

import pytest

from birdnet.acoustic.models.v3_0.pt import AcousticPTBackendFP32V3_0
from birdnet.geo.models.v3_0.pt import GeoPTBackendFP32V3_0
from birdnet_tests.helper import ensure_torch_or_skip


def _backend(backend_type: Any, model: Any) -> Any:  # noqa: ANN401
  """Build a backend around a stand-in module, skipping load()/the download."""
  import torch

  backend = backend_type(
    model_path=Path("does-not-exist.pt"), device_name="CPU", half_precision=False
  )
  backend._model = model
  backend._device = torch.device("cpu")
  return backend


def test_geo_single_tensor_output_gets_sigmoid_applied() -> None:
  ensure_torch_or_skip()
  import torch

  logits = torch.tensor([[-2.0, 0.0, 8.5955]])
  backend = _backend(GeoPTBackendFP32V3_0, lambda batch: logits)

  result = backend.predict(torch.zeros((1, 3), dtype=torch.float32))

  assert torch.allclose(result, torch.sigmoid(logits))
  # the geo model's other backends return probabilities, not logits
  assert float(result.max()) <= 1.0
  assert float(result.min()) >= 0.0


def test_acoustic_tuple_output_is_indexed_and_left_untouched() -> None:
  ensure_torch_or_skip()
  import torch

  predictions = torch.tensor([[0.25, 0.75]])
  embeddings = torch.tensor([[1.0, 2.0, 3.0]])
  backend = _backend(AcousticPTBackendFP32V3_0, lambda batch: (predictions, embeddings))

  assert torch.equal(
    backend.predict(torch.zeros((1, 4), dtype=torch.float32)), predictions
  )
  assert torch.equal(
    backend.encode(torch.zeros((1, 4), dtype=torch.float32)), embeddings
  )


def test_single_tensor_output_is_rejected_when_encoding_is_supported() -> None:
  ensure_torch_or_skip()
  import torch

  backend = _backend(
    AcousticPTBackendFP32V3_0, lambda batch: torch.tensor([[0.25, 0.75]])
  )

  with pytest.raises(ValueError, match=r"expected to return a tuple"):
    backend.predict(torch.zeros((1, 4), dtype=torch.float32))


def test_n_species_is_probed_from_a_single_tensor_output() -> None:
  ensure_torch_or_skip()
  import torch

  backend = _backend(GeoPTBackendFP32V3_0, lambda batch: torch.zeros((1, 14082)))

  assert backend.n_species == 14082
