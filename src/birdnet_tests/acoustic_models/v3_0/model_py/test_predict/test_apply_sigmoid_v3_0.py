from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from ordered_set import OrderedSet

from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.model_loader import load
from birdnet_tests.helper import (
  ensure_onnxruntime_or_skip,
  ensure_torch_or_skip,
  ensure_v3_0_torch_backend_or_skip,
)
from birdnet_tests.test_files import TEST_FILE_SHORT

_Backend = Literal["pt", "onnx", "tf", "pb"]

# The v3.0 exports apply the sigmoid inside the model graph, so the pipeline
# must not apply a second one: doubly squashed scores all land in [0.5, 0.73].
# Anchoring an absolute value catches that entire failure class - two backends
# that are wrong identically still agree with each other, but not with this.
_EXPECTED_TOP_SPECIES = "Poecile atricapillus_Black-capped Chickadee"
_EXPECTED_TOP_CONFIDENCE = 0.918
_CONFIDENCE_ABS_TOL = 0.01


def _fake_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AcousticModelV3_0:
  model_path = tmp_path / "birdnet_v3.onnx"
  model_path.write_bytes(b"onnx")
  monkeypatch.setattr("birdnet.model_loader.onnxruntime_installed", lambda: True)
  monkeypatch.setattr(
    "birdnet.model_loader.AcousticOnnxDownloaderV3_0.get_model_path_and_labels",
    lambda lang, precision: (model_path, OrderedSet(["species_a"])),
  )
  return load("acoustic", "3.0", "onnx")


def test_v3_0_apply_softmax_raises_error(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model = _fake_model(tmp_path, monkeypatch)
  with pytest.raises(ValueError, match=r"apply_softmax is not supported"):
    model.predict_session(apply_sigmoid=False, apply_softmax=True)


def test_v3_0_non_default_sigmoid_sensitivity_raises_error(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model = _fake_model(tmp_path, monkeypatch)
  with pytest.raises(ValueError, match=r"sigmoid_sensitivity is not supported"):
    model.predict_session(sigmoid_sensitivity=0.9)


def test_v3_0_apply_sigmoid_skips_the_pipeline_sigmoid(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model = _fake_model(tmp_path, monkeypatch)
  with model.predict_session(top_k=None) as session:
    assert session._specific_config.apply_sigmoid is False
    assert session._specific_config.sigmoid_sensitivity is None


def _load_model(backend: _Backend) -> AcousticModelV3_0:
  if backend == "pt":
    ensure_torch_or_skip()
    ensure_v3_0_torch_backend_or_skip()
    return load("acoustic", "3.0", "pt", precision="fp32")
  if backend == "onnx":
    ensure_onnxruntime_or_skip()
    return load("acoustic", "3.0", "onnx", precision="fp32")
  if backend == "tf":
    return load("acoustic", "3.0", "tf", precision="fp32", library="tflite")
  return load("acoustic", "3.0", "pb", precision="fp32")


@pytest.mark.parametrize("backend", ["pt", "onnx", "tf", "pb"])
def test_v3_0_predict_default_confidence_is_calibrated(backend: _Backend) -> None:
  model = _load_model(backend)
  with model.predict_session(
    n_workers=1,
    top_k=None,
    default_confidence_threshold=-float("inf"),
  ) as session:
    res = session.run(TEST_FILE_SHORT)

  # The species axis is in top-k selection order, not species order: map the
  # winning slot through species_ids to get the actual species.
  first_segment = np.asarray(res.species_probs)[0, 0]
  top_slot = int(np.argmax(first_segment))
  top_species_id = int(np.asarray(res.species_ids)[0, 0, top_slot])
  assert list(res.species_list)[top_species_id] == _EXPECTED_TOP_SPECIES
  assert first_segment[top_slot] == pytest.approx(
    _EXPECTED_TOP_CONFIDENCE, abs=_CONFIDENCE_ABS_TOL
  )
