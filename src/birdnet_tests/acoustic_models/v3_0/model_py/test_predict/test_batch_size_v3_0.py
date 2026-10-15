"""The 3.0 tf and pb exports have a static batch of 1; onnx and pt do not."""

from pathlib import Path

import pytest
from ordered_set import OrderedSet

from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.model_loader import load

_DOWNLOADERS = {
  "tf": ("AcousticTFDownloaderV3_0", "birdnet_v3.tflite", "tf_installed"),
  "pb": ("AcousticPBDownloaderV3_0", "birdnet_v3_pb", "tf_installed"),
  "onnx": ("AcousticOnnxDownloaderV3_0", "birdnet_v3.onnx", "onnxruntime_installed"),
  "pt": ("AcousticPTDownloaderV3_0", "birdnet_v3.pt", "torch_installed"),
}


def _fake_model(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> AcousticModelV3_0:
  downloader, file_name, installed = _DOWNLOADERS[backend]
  model_path = tmp_path / file_name
  if backend == "pb":
    model_path.mkdir()
  else:
    model_path.write_bytes(b"fake")
  monkeypatch.setattr(f"birdnet.model_loader.{installed}", lambda: True)
  monkeypatch.setattr(
    f"birdnet.model_loader.{downloader}.get_model_path_and_labels",
    lambda *args: (model_path, OrderedSet(["species_a"])),
  )
  model = load("acoustic", "3.0", backend)
  assert isinstance(model, AcousticModelV3_0)
  return model


@pytest.mark.parametrize("backend", ["tf", "pb"])
def test_static_batch_backends_reject_batch_size_above_one(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
  model = _fake_model(tmp_path, monkeypatch, backend)
  with pytest.raises(
    ValueError, match=r"batch_size=2 is not supported .* static batch"
  ):
    model.predict_session(batch_size=2)
  with pytest.raises(ValueError, match=r"batch_size=4 is not supported"):
    model.encode_session(batch_size=4)


@pytest.mark.parametrize("backend", ["tf", "pb"])
def test_static_batch_backends_accept_batch_size_one(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
  model = _fake_model(tmp_path, monkeypatch, backend)
  # Not entered: entering would spawn workers that crash on the fake file.
  model.predict_session(batch_size=1, top_k=None)
  model.encode_session(batch_size=1)


@pytest.mark.no_tf
@pytest.mark.parametrize("backend", ["onnx", "pt"])
def test_dynamic_batch_backends_accept_larger_batches(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
  model = _fake_model(tmp_path, monkeypatch, backend)
  model.predict_session(batch_size=8, top_k=None)
  model.encode_session(batch_size=8)
