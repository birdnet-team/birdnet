from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.testing
import pytest
from tqdm import tqdm

from birdnet.acoustic_models.inference.prediction.result import (
  AcousticPredictionResultBase,
)
from birdnet.acoustic_models.v2_4.model import AcousticModelV2_4
from birdnet.model_loader import load
from birdnet_tests.helper import (
  ensure_gpu_or_skip,
  ensure_litert_or_skip,
  get_max_absolute_tolerance,
  get_max_absolute_tolerance_threshold,
)
from birdnet_tests.test_files import TEST_FILE_LONG


@dataclass()
class AudioTestCase:
  chunk_overlap_s: float = 0.0
  bandpass_fmin: int = 0
  bandpass_fmax: int = 15_000


TEST_CASES = [
  AudioTestCase(),
  AudioTestCase(chunk_overlap_s=0.5),
  AudioTestCase(bandpass_fmin=1_000, bandpass_fmax=8_000),
]

TEST_CASES_REF_DIR = Path(__file__).with_suffix("")


def predict_test_cases(
  model: AcousticModelV2_4,
  device: str,
  n_workers: int,
) -> Generator[tuple[int, AcousticPredictionResultBase], None, None]:
  for case_nr, default in enumerate(tqdm(TEST_CASES)):
    with model.predict_session(
      top_k=None,
      default_confidence_threshold=-np.inf,
      n_workers=n_workers,
      n_feeders=1,
      prefetch_ratio=1,
      half_precision=False,
      show_stats=None,
      max_audio_duration_min=None,
      device=device,
      max_n_files=1,
      batch_size=1,
      overlap_duration_s=default.chunk_overlap_s,
      bandpass_fmin=default.bandpass_fmin,
      bandpass_fmax=default.bandpass_fmax,
      apply_sigmoid=True,
      sigmoid_sensitivity=1.0,
      custom_species_list=None,
    ) as session:
      result = session.run(TEST_FILE_LONG)
    yield case_nr, result


def create_reference_results() -> None:
  from shutil import rmtree

  if TEST_CASES_REF_DIR.is_dir():
    rmtree(TEST_CASES_REF_DIR)
  TEST_CASES_REF_DIR.mkdir(exist_ok=False, parents=True)
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  for case_nr, result in predict_test_cases(model, device="CPU", n_workers=4):
    case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    result.save(case_file)


def test_cases_inference_with_model(
  model: AcousticModelV2_4,
  device: str,
  atol: float,
  mean_atol: float,
  mean_atol_threshold: float = 0.1,
  n_workers: int = 4,
) -> None:
  max_abs_vals = []
  mean_abs_vals = []
  for case_nr, result in predict_test_cases(model, device, n_workers):
    ref_case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    ref_result = AcousticPredictionResultBase.load(ref_case_file)
    max_abs, mean_abs_thres = get_prediction_result_tolerances(
      result, ref_result, case_nr, mean_atol_threshold
    )
    max_abs_vals.append(max_abs)
    mean_abs_vals.append(mean_abs_thres)

  assert max(max_abs_vals) <= atol and max(mean_abs_vals) <= mean_atol, (
    f"Atol {atol} or mean atol {mean_atol} exceeded: "
    f"max atol {max(max_abs_vals)}, "
    f"max mean atol threshold {max(mean_abs_vals)}"
  )


def get_prediction_result_tolerances(
  result: AcousticPredictionResultBase,
  ref_result: AcousticPredictionResultBase,
  case_nr: int,
  mean_atol_threshold: float = 0.1,
) -> tuple[float, float]:
  # filepaths differ on different systems
  # numpy.testing.assert_equal(
  #   result.files,
  #   ref_result.files,
  #   err_msg=f"Files do not match for test case '{case_nr}'",
  # )

  numpy.testing.assert_equal(
    result.input_durations,
    ref_result.input_durations,
    err_msg=f"File durations do not match for test case '{case_nr}'",
  )

  numpy.testing.assert_equal(
    result.species_list,
    ref_result.species_list,
    err_msg=f"Species lists do not match for test case '{case_nr}'",
  )

  # all should be unmasked
  numpy.testing.assert_equal(
    result.species_masked,
    ref_result.species_masked,
    err_msg=f"Species masked do not match for test case '{case_nr}'",
  )

  assert result.segment_duration_s == ref_result.segment_duration_s
  assert result.overlap_duration_s == ref_result.overlap_duration_s

  # Sort species probabilities by species IDs before comparison
  result_sort_idx = numpy.argsort(result.species_ids, axis=-1)
  ref_sort_idx = numpy.argsort(ref_result.species_ids, axis=-1)

  sorted_result_ids = numpy.take_along_axis(
    result.species_ids, result_sort_idx, axis=-1
  )
  sorted_ref_ids = numpy.take_along_axis(ref_result.species_ids, ref_sort_idx, axis=-1)

  # order may differ due to different top-k selection, but ids must be the same
  np.testing.assert_array_equal(
    sorted_result_ids,
    sorted_ref_ids,
  )

  sorted_result_probs = numpy.take_along_axis(
    result.species_probs, result_sort_idx, axis=-1
  )
  sorted_ref_probs = numpy.take_along_axis(
    ref_result.species_probs, ref_sort_idx, axis=-1
  )

  max_abs = get_max_absolute_tolerance(
    sorted_result_probs,
    sorted_ref_probs,
  )

  mean_abs_thres = get_max_absolute_tolerance_threshold(
    sorted_result_probs,
    sorted_ref_probs,
    threshold=mean_atol_threshold,
  )

  # max_rel, max_abs = estimate_best_rtol_atol(
  #   sorted_result_probs,
  #   sorted_ref_probs,
  # )

  return max_abs, mean_abs_thres

  # numpy.testing.assert_allclose(
  #   result.species_probs,
  #   ref_result.species_probs,
  #   atol=max_abs,
  #   err_msg=f"Species probabilities do not match for test case '{case_nr}'",
  # )

  # precision = worst_decimal_precision(
  #   result.species_probs,
  #   ref_result.species_probs,
  # )
  # assert precision == decimal, (
  #   f"Precision {precision} does not match expected {decimal} for test case '{case_nr}'"
  # )
  # numpy.testing.assert_almost_equal(
  #   result.species_probs,
  #   ref_result.species_probs,
  #   decimal=decimal,
  #   err_msg=f"Species probabilities do not match for test case '{case_nr}'",
  # )


def test_pb_cpu_is_close() -> None:
  model = load("acoustic", "2.4", "pb", precision="fp32")
  test_cases_inference_with_model(model, "CPU", atol=0.003, mean_atol=0.0001)


@pytest.mark.gpu
def test_pb_gpu_is_close() -> None:
  ensure_gpu_or_skip()

  model = load("acoustic", "2.4", "pb", precision="fp32")
  test_cases_inference_with_model(
    model, "GPU", atol=0.002, mean_atol=0.0001, n_workers=1
  )


def test_tf32_is_same() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  test_cases_inference_with_model(model, "CPU", atol=0, mean_atol=0)


@pytest.mark.litert
def test_tf32_litert_is_very_close() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  test_cases_inference_with_model(model, "CPU", atol=0.00001, mean_atol=0.000001)


def test_tf16_is_close() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp16")
  test_cases_inference_with_model(model, "CPU", atol=0.03, mean_atol=0.003)


def test_int8_is_somewhat_close() -> None:
  model = load("acoustic", "2.4", "tf", precision="int8")
  test_cases_inference_with_model(model, "CPU", atol=0.35, mean_atol=0.09)


if __name__ == "__main__":
  create_reference_results()
