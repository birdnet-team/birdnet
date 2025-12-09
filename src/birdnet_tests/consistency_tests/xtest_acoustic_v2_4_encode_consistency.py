from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import numpy.testing
import pytest
from tqdm import tqdm

from birdnet.acoustic_models.inference.encoding.result import (
  AcousticEncodingResultBase,
)
from birdnet.acoustic_models.v2_4.model import AcousticModelV2_4
from birdnet.model_loader import load
from birdnet_tests.helper import ensure_gpu_or_skip, ensure_litert_or_skip
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
  device: str = "CPU",
) -> Generator[tuple[int, AcousticEncodingResultBase], None, None]:
  for case_nr, default in enumerate(tqdm(TEST_CASES)):
    with model.encode_session(
      n_workers=1,
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
    ) as session:
      result = session.run(TEST_FILE_LONG)
      yield case_nr, result


def create_reference_results() -> None:
  TEST_CASES_REF_DIR.mkdir(exist_ok=True, parents=True)
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  for case_nr, result in predict_test_cases(model):
    case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    result.save(case_file)


def assert_encoding_results_are_close(
  result: AcousticEncodingResultBase,
  ref_result: AcousticEncodingResultBase,
  case_nr: int,
  rtol: float,
  atol: float,
) -> None:
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
    result.embeddings_masked,
    ref_result.embeddings_masked,
    err_msg=f"Embeddings masked do not match for test case '{case_nr}'",
  )

  assert result.segment_duration_s == ref_result.segment_duration_s
  assert result.overlap_duration_s == ref_result.overlap_duration_s

  numpy.testing.assert_allclose(
    result.embeddings,
    ref_result.embeddings,
    rtol=rtol,
    atol=atol,
    err_msg=f"Embeddings do not match for test case '{case_nr}'",
  )


def test_pb_cpu_is_close() -> None:
  model = load("acoustic", "2.4", "pb", precision="fp32")
  for case_nr, result in predict_test_cases(model, device="CPU"):
    ref_case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    ref_result = AcousticEncodingResultBase.load(ref_case_file)
    assert_encoding_results_are_close(result, ref_result, case_nr, rtol=0.01, atol=0.01)


@pytest.mark.gpu
def test_pb_gpu_is_close() -> None:
  ensure_gpu_or_skip()

  model = load("acoustic", "2.4", "pb", precision="fp32")
  for case_nr, result in predict_test_cases(model, device="GPU"):
    ref_case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    ref_result = AcousticEncodingResultBase.load(ref_case_file)
    assert_encoding_results_are_close(result, ref_result, case_nr, rtol=0.01, atol=0.01)


def test_tf32_is_same() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")
  for case_nr, result in predict_test_cases(model):
    ref_case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    ref_result = AcousticEncodingResultBase.load(ref_case_file)
    assert_encoding_results_are_close(result, ref_result, case_nr, rtol=0, atol=0)


@pytest.mark.litert
def test_tf32_litert_is_very_close() -> None:
  ensure_litert_or_skip()

  model = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  for case_nr, result in predict_test_cases(model):
    ref_case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    ref_result = AcousticEncodingResultBase.load(ref_case_file)
    assert_encoding_results_are_close(
      result, ref_result, case_nr, rtol=0.001, atol=0.0001
    )


def test_tf16_is_somewhat_close() -> None:
  model = load("acoustic", "2.4", "tf", precision="fp16")
  for case_nr, result in predict_test_cases(model):
    ref_case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    ref_result = AcousticEncodingResultBase.load(ref_case_file)
    assert_encoding_results_are_close(result, ref_result, case_nr, rtol=0.1, atol=0.1)


def test_int8_is_somewhat_close() -> None:
  model = load("acoustic", "2.4", "tf", precision="int8")
  for case_nr, result in predict_test_cases(model):
    ref_case_file = TEST_CASES_REF_DIR / f"{case_nr}.npz"
    ref_result = AcousticEncodingResultBase.load(ref_case_file)
    assert_encoding_results_are_close(result, ref_result, case_nr, rtol=0.1, atol=0.1)


if __name__ == "__main__":
  create_reference_results()
