from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from ordered_set import OrderedSet

from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.model_loader import load
from birdnet_tests.helper import (
  ensure_onnxruntime_or_skip,
  ensure_tf_2_18_or_skip,
  ensure_torch_or_skip,
  ensure_v3_0_torch_backend_or_skip,
)
from birdnet_tests.test_files import TEST_FILE_SHORT

# Deliberately no `load_model` marker: that phase holds only the download
# tests; these use the models it already fetched.
_Backend = Literal["pt", "onnx", "tf", "pb"]

# The exports bake in the sigmoid; a second one squashes every score into
# [0.5, 0.73]. An absolute anchor catches that class - backends that are wrong
# identically still agree with each other, but not with this.
_EXPECTED_TOP_SPECIES = "Poecile atricapillus_Black-capped Chickadee"
_EXPECTED_TOP_CONFIDENCE = 0.918
_CONFIDENCE_ABS_TOL = 0.01
# fp16 deviates up to ~0.003 from fp32 here; leave headroom for kernel spread.
_CONFIDENCE_ABS_TOL_FP16 = 0.02


def _fake_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AcousticModelV3_0:
  model_path = tmp_path / "birdnet_v3.onnx"
  model_path.write_bytes(b"onnx")
  monkeypatch.setattr("birdnet.model_loader.onnxruntime_installed", lambda: True)
  monkeypatch.setattr(
    "birdnet.model_loader.AcousticOnnxDownloaderV3_0.get_model_path_and_labels",
    lambda lang, precision: (model_path, OrderedSet(["species_a"])),
  )
  return load("acoustic", "3.0", "onnx")


@pytest.mark.no_tf
def test_v3_0_apply_softmax_raises_error(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model = _fake_model(tmp_path, monkeypatch)
  with pytest.raises(ValueError, match=r"apply_softmax is not supported"):
    model.predict_session(apply_sigmoid=False, apply_softmax=True)


@pytest.mark.no_tf
def test_v3_0_non_default_sigmoid_sensitivity_raises_error(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model = _fake_model(tmp_path, monkeypatch)
  with pytest.raises(ValueError, match=r"sigmoid_sensitivity is not supported"):
    model.predict_session(sigmoid_sensitivity=0.9)


@pytest.mark.no_tf
def test_v3_0_non_default_sigmoid_sensitivity_raises_even_without_apply_sigmoid(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model = _fake_model(tmp_path, monkeypatch)
  with pytest.raises(ValueError, match=r"sigmoid_sensitivity is not supported"):
    model.predict_session(apply_sigmoid=False, sigmoid_sensitivity=0.9)


@pytest.mark.no_tf
def test_v3_0_apply_sigmoid_skips_the_pipeline_sigmoid(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model = _fake_model(tmp_path, monkeypatch)
  # Not entered: the fields are parent-side config; entering would spawn
  # workers that crash on the fake model file.
  session = model.predict_session(top_k=None)
  assert session._specific_config.apply_sigmoid is False
  assert session._specific_config.sigmoid_sensitivity is None


def _load_model(backend: _Backend, precision: str) -> AcousticModelV3_0:
  if backend == "pt":
    ensure_torch_or_skip()
    ensure_v3_0_torch_backend_or_skip()
    return load("acoustic", "3.0", "pt", precision=precision)
  if backend == "onnx":
    ensure_onnxruntime_or_skip()
    return load("acoustic", "3.0", "onnx", precision=precision)
  # The v3.0 exports need a newer TF than the macOS Intel pin (<2.17).
  if backend == "tf":
    ensure_tf_2_18_or_skip()
    return load("acoustic", "3.0", "tf", precision=precision, library="tflite")
  ensure_tf_2_18_or_skip()
  return load("acoustic", "3.0", "pb", precision=precision)


# fp16 variants are separate export files with their own output indices.
@pytest.mark.parametrize(
  ("backend", "precision"),
  [
    pytest.param("pt", "fp32", marks=pytest.mark.no_tf),
    pytest.param("onnx", "fp32", marks=pytest.mark.no_tf),
    pytest.param("onnx", "fp16", marks=pytest.mark.no_tf),
    ("tf", "fp32"),
    ("tf", "fp16"),
    ("pb", "fp32"),
  ],
)
def test_v3_0_predict_default_confidence_is_calibrated(
  backend: _Backend, precision: str
) -> None:
  model = _load_model(backend, precision)
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
  tol = _CONFIDENCE_ABS_TOL_FP16 if precision == "fp16" else _CONFIDENCE_ABS_TOL
  assert list(res.species_list)[top_species_id] == _EXPECTED_TOP_SPECIES
  assert first_segment[top_slot] == pytest.approx(_EXPECTED_TOP_CONFIDENCE, abs=tol)
