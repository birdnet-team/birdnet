from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy.testing
from tqdm import tqdm

from birdnet.acoustic_models.inference.scores.prediction_result import PredictionResult
from birdnet.acoustic_models.v2_4.model import AcousticModelV2_4
from birdnet.model_loader import load
from birdnet_tests.test_files import TEST_FILE_WAV


@dataclass()
class AudioTestCase:
  min_confidence: float = 0.1
  top_k: int | None = None
  chunk_overlap_s: float = 0.0
  bandpass_fmin: int = 0
  bandpass_fmax: int = 15_000
  apply_sigmoid: bool = True
  sigmoid_sensitivity: float | None = 1.0
  filter_species: set[str] | None = None


TEST_CASES = {
  1: AudioTestCase(),
  2: AudioTestCase(min_confidence=0.3),
  3: AudioTestCase(top_k=3),
  4: AudioTestCase(chunk_overlap_s=0.5),
  5: AudioTestCase(bandpass_fmin=1_000, bandpass_fmax=8_000),
  6: AudioTestCase(apply_sigmoid=False),
  7: AudioTestCase(sigmoid_sensitivity=1.5),
}
TEST_CASES_REF_DIR = Path(__file__).with_suffix("")


def predict_test_cases(
  model: AcousticModelV2_4,
) -> Generator[tuple[int, PredictionResult], None, None]:
  for case_nr, default in tqdm(list(TEST_CASES.items())):
    with model.predict_session(
      top_k=default.top_k,
      n_workers=1,
      n_feeders=1,
      prefetch_ratio=1,
      half_precision=False,
      show_stats=None,
      max_audio_duration_min=None,
      device="CPU",
      max_n_files=1,
      batch_size=1,
      overlap_duration_s=default.chunk_overlap_s,
      bandpass_fmin=default.bandpass_fmin,
      bandpass_fmax=default.bandpass_fmax,
      apply_sigmoid=default.apply_sigmoid,
      sigmoid_sensitivity=default.sigmoid_sensitivity,
      custom_species_list=default.filter_species,
    ) as session:
      result = session.run(TEST_FILE_WAV)
      yield case_nr, result


def create_reference_results() -> None:
  TEST_CASES_REF_DIR.mkdir(exist_ok=True, parents=True)
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  for case_nr, result in predict_test_cases(model):
    case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    result.save(case_file)


def assert_prediction_results_are_close(
  result: PredictionResult,
  ref_result: PredictionResult,
  case_nr: int,
  rtol: float,
  atol: float,
) -> None:
  numpy.testing.assert_equal(
    result.files,
    ref_result.files,
    err_msg=f"Files do not match for test case '{case_nr}'",
  )

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

  numpy.testing.assert_allclose(
    result.species_probs,
    ref_result.species_probs,
    rtol=rtol,
    atol=atol,
    err_msg=f"Species probabilities do not match for test case '{case_nr}'",
  )


def test_pb_is_close() -> None:
  model = load("acoustic", "2.4", "pb", precision="fp32")
  for case_nr, result in predict_test_cases(model):
    ref_case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    ref_result = PredictionResult.load(ref_case_file)
    assert_prediction_results_are_close(
      result, ref_result, case_nr, rtol=0.02, atol=1e-8
    )


def test_tf32_is_same() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  for case_nr, result in predict_test_cases(model):
    ref_case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    ref_result = PredictionResult.load(ref_case_file)
    assert_prediction_results_are_close(result, ref_result, case_nr, rtol=0, atol=0)


def test_tf32_litert_is_very_close() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  for case_nr, result in predict_test_cases(model):
    ref_case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    ref_result = PredictionResult.load(ref_case_file)
    assert_prediction_results_are_close(
      result, ref_result, case_nr, rtol=0.0001, atol=1e-8
    )


def test_tf16_is_somewhat_close() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp16")
  for case_nr, result in predict_test_cases(model):
    ref_case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    ref_result = PredictionResult.load(ref_case_file)
    assert_prediction_results_are_close(
      result, ref_result, case_nr, rtol=0.3, atol=1e-8
    )


def test_int8_is_somewhat_close() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp16")
  for case_nr, result in predict_test_cases(model):
    ref_case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    ref_result = PredictionResult.load(ref_case_file)
    assert_prediction_results_are_close(
      result, ref_result, case_nr, rtol=0.3, atol=1e-8
    )


if __name__ == "__main__":
  create_reference_results()
