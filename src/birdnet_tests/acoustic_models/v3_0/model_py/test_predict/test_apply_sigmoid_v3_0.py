import inspect
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from ordered_set import OrderedSet

from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticPredictionResultBase,
)
from birdnet.acoustic.models.v3_0 import onnx, pb, pt, tf
from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.model_loader import load
from birdnet_tests.acoustic_models.v3_0.model_py.conftest import (
  V3_0Backend,
  load_v3_0,
  onnx_fp32_prediction,
)
from birdnet_tests.helper import scores_at_sigmoid_sensitivity
from birdnet_tests.test_files import TEST_FILE_SHORT

# Deliberately no `load_model` marker: that phase holds only the download
# tests; these use the models it already fetched.

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


# Sessions below are not entered: the fields are parent-side config; entering
# would spawn workers that crash on the fake model file.
@pytest.mark.no_tf
def test_v3_0_apply_sigmoid_skips_the_pipeline_sigmoid(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model = _fake_model(tmp_path, monkeypatch)
  session = model.predict_session(top_k=None)
  assert session._specific_config.apply_sigmoid is False
  assert session._specific_config.sigmoid_sensitivity is None


@pytest.mark.no_tf
def test_v3_0_non_default_sigmoid_sensitivity_turns_the_pipeline_sigmoid_on(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model = _fake_model(tmp_path, monkeypatch)
  session = model.predict_session(top_k=None, sigmoid_sensitivity=0.9)
  assert session._specific_config.apply_sigmoid is True
  assert session._specific_config.sigmoid_sensitivity == 0.9


@pytest.mark.no_tf
def test_v3_0_sigmoid_sensitivity_none_is_the_default_sigmoid(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model = _fake_model(tmp_path, monkeypatch)
  session = model.predict_session(top_k=None, sigmoid_sensitivity=None)
  assert session._specific_config.apply_sigmoid is False
  assert session._specific_config.sigmoid_sensitivity is None


@pytest.mark.no_tf
def test_v3_0_sigmoid_sensitivity_is_ignored_without_apply_sigmoid(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model = _fake_model(tmp_path, monkeypatch)
  session = model.predict_session(
    top_k=None, apply_sigmoid=False, sigmoid_sensitivity=0.9
  )
  assert session._specific_config.apply_sigmoid is False
  assert session._specific_config.sigmoid_sensitivity is None


@pytest.mark.no_tf
def test_v3_0_sigmoid_sensitivity_is_validated(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model = _fake_model(tmp_path, monkeypatch)
  with pytest.raises(ValueError, match=r"range \[0.5, 1.5\]"):
    model.predict_session(sigmoid_sensitivity=2.0)
  with pytest.raises(TypeError, match=r"must be a number"):
    model.predict_session(sigmoid_sensitivity="high")  # type: ignore[arg-type]


def _backend_classes(module: ModuleType) -> list[type]:
  return [
    cls
    for name, cls in inspect.getmembers(module, inspect.isclass)
    if cls.__module__ == module.__name__ and "Backend" in name
  ]


@pytest.mark.no_tf
@pytest.mark.parametrize("module", [onnx, pb, pt, tf], ids=lambda m: m.__name__)
def test_every_v3_0_backend_declares_the_sigmoid_in_its_graph(
  module: ModuleType,
) -> None:
  # The prediction worker recovers the logits only for a backend that says so;
  # one that stays silent gets the pipeline sigmoid on top of its scores.
  classes = _backend_classes(module)
  assert classes
  for cls in classes:
    assert cls.applies_sigmoid() is True, cls.__name__


def _run(model: AcousticModelV3_0, **kwargs: object) -> AcousticPredictionResultBase:
  with model.predict_session(
    n_workers=1,
    top_k=None,
    default_confidence_threshold=-float("inf"),
    **kwargs,  # type: ignore[arg-type]
  ) as session:
    return session.run(TEST_FILE_SHORT)


# fp16 variants are separate export files with their own output indices. pb is
# bounded to onnx within 1e-4 by test_v3_0_predict_pb_and_onnx_are_close.
@pytest.mark.parametrize(
  ("backend", "precision"),
  [
    pytest.param("pt", "fp32", marks=pytest.mark.no_tf),
    pytest.param("onnx", "fp32", marks=pytest.mark.no_tf),
    pytest.param("onnx", "fp16", marks=pytest.mark.no_tf),
    ("tf", "fp32"),
    ("tf", "fp16"),
  ],
)
def test_v3_0_predict_default_confidence_is_calibrated(
  backend: V3_0Backend, precision: str
) -> None:
  if (backend, precision) == ("onnx", "fp32"):
    res = onnx_fp32_prediction()
  else:
    res = _run(load_v3_0(backend, precision))

  # The species axis is in top-k selection order, not species order: map the
  # winning slot through species_ids to get the actual species.
  first_segment = np.asarray(res.species_probs)[0, 0]
  top_slot = int(np.argmax(first_segment))
  top_species_id = int(np.asarray(res.species_ids)[0, 0, top_slot])
  tol = _CONFIDENCE_ABS_TOL_FP16 if precision == "fp16" else _CONFIDENCE_ABS_TOL
  assert list(res.species_list)[top_species_id] == _EXPECTED_TOP_SPECIES
  assert first_segment[top_slot] == pytest.approx(_EXPECTED_TOP_CONFIDENCE, abs=tol)


def _in_model_order(res: AcousticPredictionResultBase) -> np.ndarray:
  """Every score of every segment, species in model order; shape (segments, n)."""
  order = np.argsort(np.asarray(res.species_ids), axis=-1)
  return np.take_along_axis(np.asarray(res.species_probs), order, axis=-1)[0]


def _default_scores() -> np.ndarray:
  return _in_model_order(onnx_fp32_prediction())


# The recovery runs in the prediction worker and never reads the backend:
# prediction_worker_py/test_in_graph_sigmoid_sensitivity.py pins it on the
# fake for both model classes, and the flag the worker branches on is pinned
# per class above. One real backend shows it on real scores.
@pytest.mark.no_tf
@pytest.mark.parametrize("sensitivity", [0.5, 1.5])
def test_v3_0_sigmoid_sensitivity_transforms_the_recovered_logits(
  sensitivity: float,
) -> None:
  default = _default_scores()
  scores = _in_model_order(_run(load_v3_0("onnx"), sigmoid_sensitivity=sensitivity))

  # Two runs of one model differ by ~1e-7 in score, which a score next to 1
  # turns into a visible shift of its logit; the recovery itself is good to
  # 1.3e-5 at 0.5.
  np.testing.assert_allclose(
    scores, scores_at_sigmoid_sensitivity(default, sensitivity), atol=1e-4
  )
  # Outside this band the clip at +-15 holds a score where it is, as it holds
  # a 2.4 model's logits.
  moved = (default > 0.01) & (default < 0.99)
  assert moved.any()
  if sensitivity < 1.0:
    assert np.all(scores[moved] < default[moved])
  else:
    assert np.all(scores[moved] > default[moved])


@pytest.mark.no_tf
def test_v3_0_threshold_applies_to_the_transformed_scores() -> None:
  threshold = 0.1
  # 0.75 shifts every logit by -2.5: the file's top detection (0.918) lands at
  # 0.48 and still clears the threshold, which at 0.5 nothing does.
  with load_v3_0("onnx").predict_session(
    n_workers=1,
    top_k=None,
    sigmoid_sensitivity=0.75,
    default_confidence_threshold=threshold,
  ) as session:
    res = session.run(TEST_FILE_SHORT)

  probs = np.asarray(res.species_probs)
  masked = np.asarray(res.species_masked)
  np.testing.assert_array_equal(masked, probs < threshold)
  # The sensitivity moved scores below the threshold that cleared it before.
  assert 0 < (~masked).sum() < (_default_scores() >= threshold).sum()
