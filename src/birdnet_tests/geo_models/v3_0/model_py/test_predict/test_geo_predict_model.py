import numpy as np
import pytest

from birdnet.geo.models.base import GeoModelBase
from birdnet.model_loader import load
from birdnet_tests.helper import (
  ensure_gpu_or_skip,
  ensure_onnxruntime_or_skip,
  ensure_tf_2_19_or_2_18,
  ensure_torch_or_skip,
  ensure_v3_0_torch_backend_or_skip,
)

# What the result reports about the call (path, version, coordinates, week) is
# pinned on the fake in geo_models/base_py/test_geo_predict_forwards_its_arguments.py,
# the dtype the session returns in geo_models/inference/session_py/
# test_half_precision.py, and each export's model_precision in the calibration
# test below.


def _load(backend: str) -> GeoModelBase:
  kwargs: dict = {"precision": "fp32"}
  if backend == "onnx":
    ensure_onnxruntime_or_skip()
  elif backend == "pt":
    ensure_torch_or_skip()
    ensure_v3_0_torch_backend_or_skip()
  elif backend == "tf":
    ensure_tf_2_19_or_2_18()
    kwargs["library"] = "tflite"
  return load("geo", "3.0", backend, **kwargs)


_SMOKE_CASES = [
  pytest.param("tf", "CPU", True, id="tf-fp32-half"),
  pytest.param("pb", "CPU", True, id="pb-half"),
  pytest.param("onnx", "CPU", True, marks=pytest.mark.no_tf, id="onnx-fp32-half"),
  pytest.param("pt", "CPU", True, marks=pytest.mark.no_tf, id="pt-half"),
  pytest.param("tf", "GPU", False, marks=pytest.mark.gpu, id="tf-gpu"),
  pytest.param("pb", "GPU", False, marks=pytest.mark.gpu, id="pb-gpu"),
  pytest.param("pt", "GPU", False, marks=pytest.mark.gpu, id="pt-gpu"),
  pytest.param("onnx", "GPU", False, marks=pytest.mark.gpu, id="onnx-gpu"),
]


@pytest.mark.parametrize(("backend", "device", "half_precision"), _SMOKE_CASES)
def test_predict_shape_and_dtype(
  backend: str, device: str, half_precision: bool
) -> None:
  if device == "GPU":
    ensure_gpu_or_skip()
  model = _load(backend)

  result = model.predict(
    20, 50, week=1, min_confidence=0.03, half_precision=half_precision, device=device
  )

  assert result.species_probs.shape == (14082,)
  assert result.species_probs.dtype == (np.float16 if half_precision else np.float32)


@pytest.mark.no_tf
@pytest.mark.parametrize("backend", ["pt", "onnx"])
def test_year_round(backend: str) -> None:
  model = _load(backend)
  result = model.predict(20, 50, min_confidence=0.03, half_precision=False)

  assert result.latitude == 20
  assert result.longitude == 50
  assert result.week == -1
  assert result.model_version == "3.0"
  assert result.model_precision == "fp32"


# --- Calibration ---

# Absolute anchors (Ithaca, NY, late January): an abundant winter resident
# near 1, an absent species near 0. Catches squashed outputs and species-list
# drift, which the cross-backend tests above cannot. Keyed by sci-name prefix
# because common names come from the replaceable taxonomy.
_CALIBRATION_LAT = 42.5
_CALIBRATION_LON = -76.45
_CALIBRATION_WEEK = 4
_RESIDENT_PREFIX = "Spinus tristis_"  # American Goldfinch: p=0.9998 (fp32)
_ABSENT_PREFIX = "Dromaius novaehollandiae_"  # Emu: p=0.0000 (fp32)


def _species_prob(result, sci_name_prefix: str) -> float:  # noqa: ANN001
  species = [str(s) for s in result.species_list]
  matches = [i for i, s in enumerate(species) if s.startswith(sci_name_prefix)]
  assert len(matches) == 1, f"expected exactly one match: {sci_name_prefix}"
  slot = int(np.where(np.asarray(result.species_ids) == matches[0])[0][0])
  return float(np.asarray(result.species_probs)[slot])


@pytest.mark.parametrize(
  ("backend", "precision", "resident_min", "absent_max"),
  [
    pytest.param("onnx", "fp32", 0.99, 0.01, marks=pytest.mark.no_tf),
    pytest.param("onnx", "fp16", 0.99, 0.01, marks=pytest.mark.no_tf),
    ("pb", "fp32", 0.99, 0.01),
    pytest.param("pt", "fp32", 0.99, 0.01, marks=pytest.mark.no_tf),
    ("tf", "fp32", 0.99, 0.01),
    ("tf", "fp16", 0.99, 0.01),
    # int8 deviates more; the anchors only reject squashed/misaligned outputs.
    ("tf", "int8", 0.9, 0.05),
  ],
)
def test_calibration_anchor_probabilities(
  backend: str, precision: str, resident_min: float, absent_max: float
) -> None:
  kwargs: dict = {"precision": precision}
  if backend == "onnx":
    ensure_onnxruntime_or_skip()
  elif backend == "pt":
    ensure_torch_or_skip()
    ensure_v3_0_torch_backend_or_skip()
  elif backend == "tf":
    ensure_tf_2_19_or_2_18()
    kwargs["library"] = "tflite"

  model = load("geo", "3.0", backend, **kwargs)
  result = model.predict(
    _CALIBRATION_LAT, _CALIBRATION_LON, week=_CALIBRATION_WEEK, min_confidence=0.0
  )

  assert result.model_precision == precision
  resident = _species_prob(result, _RESIDENT_PREFIX)
  absent = _species_prob(result, _ABSENT_PREFIX)
  # Two-sided: raw logits (missing baked-in sigmoid) pass one-sided bounds.
  assert resident_min < resident <= 1.0, f"resident species scored {resident}"
  assert 0.0 <= absent < absent_max, f"absent species scored {absent}"


@pytest.mark.no_tf
def test_pt_matches_onnx() -> None:
  """The backends must agree - same species order, same probabilities.

  This is the assertion that catches a missing or misplaced sigmoid in the
  TorchScript path as well as a species list that drifted out of sync with the
  model outputs; the per-backend tests above pass in both cases.
  """
  ensure_torch_or_skip()
  ensure_v3_0_torch_backend_or_skip()
  ensure_onnxruntime_or_skip()

  pt_model = load("geo", "3.0", "pt", precision="fp32")
  onnx_model = load("geo", "3.0", "onnx", precision="fp32")

  assert list(pt_model.species_list) == list(onnx_model.species_list)

  for latitude, longitude, week in ((42.5, -76.45, 4), (-33.9, 151.2, None)):
    pt_result = pt_model.predict(latitude, longitude, week=week, min_confidence=0.0)
    onnx_result = onnx_model.predict(latitude, longitude, week=week, min_confidence=0.0)
    np.testing.assert_allclose(
      pt_result.species_probs,
      onnx_result.species_probs,
      rtol=0,
      atol=1e-5,
      err_msg=f"pt and onnx disagree at ({latitude}, {longitude}, week={week})",
    )
