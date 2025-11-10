from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import numpy.testing
import pytest
from tqdm import tqdm

from birdnet.acoustic_models.inference.scores.prediction_result import PredictionResult
from birdnet.acoustic_models.v2_4.model import AcousticModelV2_4
from birdnet.model_loader import load
from birdnet_tests.helper import (
  ensure_gpu_or_skip,
  ensure_litert_or_skip,
  estimate_best_rtol_atol,
)
from birdnet_tests.test_files import TEST_FILE_WAV


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
) -> Generator[tuple[int, PredictionResult], None, None]:
  for case_nr, default in enumerate(tqdm(TEST_CASES)):
    with model.predict_session(
      top_k=None,
      n_workers=4,
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
      apply_sigmoid=False,
      sigmoid_sensitivity=None,
      custom_species_list=None,
    ) as session:
      result = session.run(TEST_FILE_WAV)
      yield case_nr, result


def create_reference_results() -> None:
  from shutil import rmtree

  if TEST_CASES_REF_DIR.is_dir():
    rmtree(TEST_CASES_REF_DIR)
  TEST_CASES_REF_DIR.mkdir(exist_ok=False, parents=True)
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  for case_nr, result in predict_test_cases(model, device="CPU"):
    case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    result.save(case_file)


def test_cases_inference_with_model(
  model: AcousticModelV2_4, device: str, atol: float, rtol: float
) -> None:
  max_abs_vals = []
  max_rel_vals = []
  for case_nr, result in predict_test_cases(model, device):
    ref_case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    ref_result = PredictionResult.load(ref_case_file)
    max_rel, max_abs = get_prediction_result_tolerances(result, ref_result, case_nr)
    max_abs_vals.append(max_abs)
    max_rel_vals.append(max_rel)
  
  print(
    f"Max absolute tolerance: {max(max_abs_vals)}, max relative tolerance: {max(max_rel_vals)}"
  )
  assert max(max_abs_vals) <= atol
  assert max(max_rel_vals) <= rtol


def get_prediction_result_tolerances(
  result: PredictionResult,
  ref_result: PredictionResult,
  case_nr: int,
) -> tuple[float, float]:
  # filepaths differ on different systems
  # numpy.testing.assert_equal(
  #   result.files,
  #   ref_result.files,
  #   err_msg=f"Files do not match for test case '{case_nr}'",
  # )

  numpy.testing.assert_equal(
    result.file_durations,
    ref_result.file_durations,
    err_msg=f"File durations do not match for test case '{case_nr}'",
  )

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

  assert result.segment_duration_s == ref_result.segment_duration_s
  assert result.overlap_duration_s == ref_result.overlap_duration_s

  max_rel, max_abs = estimate_best_rtol_atol(
    result.species_probs,
    ref_result.species_probs,
  )

  return max_rel, max_abs

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
  test_cases_inference_with_model(model, "CPU", atol=0.016, rtol=0.06)


@pytest.mark.gpu
def test_pb_gpu_is_close() -> None:
  ensure_gpu_or_skip()

  model = load("acoustic", "2.4", "pb", precision="fp32")
  test_cases_inference_with_model(model, "GPU", atol=0, rtol=0)


def test_tf32_is_same() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  test_cases_inference_with_model(model, "CPU", atol=0, rtol=0)


@pytest.mark.litert
def test_tf32_litert_is_very_close() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  test_cases_inference_with_model(model, "CPU", atol=3e-5, rtol=6e-5)


def test_tf16_is_not_so_close() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp16")
  test_cases_inference_with_model(model, "CPU", atol=0.36, rtol=0.29)


def test_int8_is_somewhat_close() -> None:
  model = load("acoustic", "2.4", "tf", precision="int8")
  test_cases_inference_with_model(model, "CPU", atol=-1, rtol=-1)


if __name__ == "__main__":
  create_reference_results()
