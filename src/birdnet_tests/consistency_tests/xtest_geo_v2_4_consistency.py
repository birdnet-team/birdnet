from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy.testing
import pytest
from tqdm import tqdm

from birdnet.geo_models.inference.prediction_result import GeoPredictionResult
from birdnet.geo_models.v2_4.model import GeoModelV2_4
from birdnet.model_loader import load
from birdnet_tests.helper import ensure_gpu_or_skip, ensure_litert_or_skip


@dataclass()
class GeoTestCase:
  latitude: float = 0.0
  longitude: float = 0.0
  min_confidence: float = 0.1
  week: int | None = None


TEST_CASES = {
  1: GeoTestCase(),
  2: GeoTestCase(min_confidence=0.3),
  3: GeoTestCase(week=12),
  4: GeoTestCase(latitude=45.0, longitude=-93.0),
  5: GeoTestCase(latitude=-33.9, longitude=151.2, week=25),
  6: GeoTestCase(latitude=51.5, longitude=-0.1, min_confidence=0.05, week=40),
}
TEST_CASES_REF_DIR = Path(__file__).with_suffix("")


def predict_test_cases(
  model: GeoModelV2_4,
  device: str = "CPU",
) -> Generator[tuple[int, GeoPredictionResult], None, None]:
  for case_nr, default in tqdm(list(TEST_CASES.items())):
    with model.predict_session(
      half_precision=False,
      device=device,
      min_confidence=default.min_confidence,
    ) as session:
      result = session.run(
        default.latitude,
        default.longitude,
        week=default.week,
      )
      yield case_nr, result


def create_reference_results() -> None:
  TEST_CASES_REF_DIR.mkdir(exist_ok=True, parents=True)
  model = load("geo", "2.4", "tf", precision="fp32", library="tflite")
  for case_nr, result in predict_test_cases(model):
    case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    result.save(case_file)


def assert_prediction_results_are_close(
  result: GeoPredictionResult,
  ref_result: GeoPredictionResult,
  case_nr: int,
  rtol: float,
  atol: float,
) -> None:
  numpy.testing.assert_equal(
    result.species_list,
    ref_result.species_list,
    err_msg=f"Species lists do not match for test case '{case_nr}'",
  )

  numpy.testing.assert_equal(
    result.species_ids,
    ref_result.species_ids,
    err_msg=f"Species IDs do not match for test case '{case_nr}'",
  )

  numpy.testing.assert_equal(
    result.species_masked,
    ref_result.species_masked,
    err_msg=f"Species masked do not match for test case '{case_nr}'",
  )

  numpy.testing.assert_allclose(
    result.species_probs,
    ref_result.species_probs,
    rtol=rtol,
    atol=atol,
    err_msg=f"Species probabilities do not match for test case '{case_nr}'",
  )


def test_pb_cpu_is_very_close() -> None:
  model = load("geo", "2.4", "pb", precision="fp32")
  for case_nr, result in predict_test_cases(model, device="CPU"):
    ref_case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    ref_result = GeoPredictionResult.load(ref_case_file)
    assert_prediction_results_are_close(
      result, ref_result, case_nr, rtol=0.00001, atol=1e-8
    )


@pytest.mark.gpu
def test_pb_gpu_is_very_close() -> None:
  ensure_gpu_or_skip()

  model = load("geo", "2.4", "pb", precision="fp32")
  for case_nr, result in predict_test_cases(model, device="GPU"):
    ref_case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    ref_result = GeoPredictionResult.load(ref_case_file)
    assert_prediction_results_are_close(
      result, ref_result, case_nr, rtol=0.00001, atol=1e-8
    )


def test_tf32_is_same() -> None:
  model = load("geo", "2.4", "tf", precision="fp32", library="tflite")
  for case_nr, result in predict_test_cases(model):
    ref_case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    ref_result = GeoPredictionResult.load(ref_case_file)
    assert_prediction_results_are_close(result, ref_result, case_nr, rtol=0, atol=0)


@pytest.mark.litert
def test_tf32_litert_is_very_close() -> None:
  ensure_litert_or_skip()

  model = load("geo", "2.4", "tf", precision="fp32", library="litert")
  for case_nr, result in predict_test_cases(model):
    ref_case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    ref_result = GeoPredictionResult.load(ref_case_file)
    assert_prediction_results_are_close(
      result, ref_result, case_nr, rtol=0.00001, atol=1e-8
    )


if __name__ == "__main__":
  create_reference_results()
