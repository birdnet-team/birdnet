import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import numpy.testing as npt
import numpy.typing as npty
import pytest
from ordered_set import OrderedSet
from tqdm import tqdm

from birdnet.models.model_v2m4_base import ModelV2M4Base
from birdnet.models.model_v2m4_protobuf import ModelV2M4Protobuf
from birdnet.types import Language, Species, SpeciesPredictions
from birdnet_tests.helper import (TEST_FILES_DIR, convert_predictions_to_numpy,
                                  species_predictions_are_equal)

# Duration: 120s
TEST_FILE_WAV = TEST_FILES_DIR / "soundscape.wav"
TEST_PATH = Path(f"{TEST_FILE_WAV}.global.pkl")


@dataclass()
class AudioTestCase():
  min_confidence: float = 0.1
  batch_size: int = 1
  chunk_overlap_s: float = 0.0
  use_bandpass: bool = True
  bandpass_fmin: Optional[int] = 0
  bandpass_fmax: Optional[int] = 15_000
  apply_sigmoid: bool = True
  sigmoid_sensitivity: Optional[float] = 1.0
  filter_species: Optional[Set[Species]] = None

  # def predict_species_within_audio_file(self, model: ModelV2M4Base, audio_file: Path) -> SpeciesPredictions:
  #   return model.predict_species_within_audio_file(
  #     audio_file,
  #     min_confidence=self.min_confidence,
  #     batch_size=self.batch_size,
  #     chunk_overlap_s=self.chunk_overlap_s,
  #     use_bandpass=self.use_bandpass,
  #     bandpass_fmin=self.bandpass_fmin,
  #     bandpass_fmax=self.bandpass_fmax,
  #     apply_sigmoid=self.apply_sigmoid,
  #     sigmoid_sensitivity=self.sigmoid_sensitivity,
  #     filter_species=self.filter_species,
  #   )


def predict_species_within_audio_file(test_case: AudioTestCase, model: ModelV2M4Base, audio_file: Path) -> SpeciesPredictions:
  return model.predict_species_within_audio_file(
    audio_file,
    min_confidence=test_case.min_confidence,
    batch_size=test_case.batch_size,
    chunk_overlap_s=test_case.chunk_overlap_s,
    use_bandpass=test_case.use_bandpass,
    bandpass_fmin=test_case.bandpass_fmin,
    bandpass_fmax=test_case.bandpass_fmax,
    apply_sigmoid=test_case.apply_sigmoid,
    sigmoid_sensitivity=test_case.sigmoid_sensitivity,
    filter_species=test_case.filter_species,
  )


class DummyModel(ModelV2M4Base):
  def __init__(self, /, *, language: Language = "en_us") -> None:
    super().__init__(language=language)
    self._species_list = OrderedSet(("species1", "species2", "species3", "species0"))

  def _predict_species(self, batch: npty.NDArray[np.float32]) -> npty.NDArray[np.float32]:
    assert batch.dtype == np.float32
    prediction_np = np.zeros((len(batch), len(self._species_list)))
    for i in range(len(batch)):
      prediction_np[i, 0] = 0.1
      prediction_np[i, 1] = 0.3
      prediction_np[i, 2] = 0.2
      prediction_np[i, 3] = 0.1
    return prediction_np


@pytest.fixture(name="model")
def get_model():
  model = DummyModel(language="en_us")
  return model


def test_invalid_audio_file_path_raises_value_error(model: ModelV2M4Base):
  with pytest.raises(ValueError, match=r"Value for 'audio_file' is invalid! It needs to be a path to an existing audio file."):
    model.predict_species_within_audio_file(
        TEST_FILES_DIR / "dummy.wav"
    )


def test_invalid_batch_size_raises_value_error(model: ModelV2M4Base):
  with pytest.raises(ValueError, match=r"Value for 'batch_size' is invalid! It needs to be larger than zero."):
    model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        batch_size=0
    )


def test_invalid_min_confidence_raises_value_error(model: ModelV2M4Base):
  with pytest.raises(ValueError, match=r"Value for 'min_confidence' is invalid! It needs to be in interval \[0.0, 1.0\)."):
    model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        min_confidence=-0.1
    )

  with pytest.raises(ValueError, match=r"Value for 'min_confidence' is invalid! It needs to be in interval \[0.0, 1.0\)."):
    model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        min_confidence=1.1
    )


def test_invalid_chunk_overlap_s_raises_value_error(model: ModelV2M4Base):
  with pytest.raises(ValueError, match=r"Value for 'chunk_overlap_s' is invalid! It needs to be in interval \[0.0, 3.0\)"):
    model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        chunk_overlap_s=-1,
    )

  with pytest.raises(ValueError, match=r"Value for 'chunk_overlap_s' is invalid! It needs to be in interval \[0.0, 3.0\)"):
    model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        chunk_overlap_s=4.0,
    )


def test_invalid_bandpass_fmin_raises_value_error(model: ModelV2M4Base):
  with pytest.raises(ValueError, match=r"Value for 'bandpass_fmin' is invalid! It needs to be larger than zero."):
    model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        bandpass_fmin=-1
    )


def test_invalid_bandpass_fmax_raises_value_error(model: ModelV2M4Base):
  with pytest.raises(ValueError, match=r"Value for 'bandpass_fmax' is invalid! It needs to be larger than 'bandpass_fmin'."):
    model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        bandpass_fmin=500,
        bandpass_fmax=400
    )


def test_invalid_sigmoid_sensitivity_raises_value_error(model: ModelV2M4Base):
  with pytest.raises(ValueError, match=r"Value for 'sigmoid_sensitivity' is required if 'apply_sigmoid==True'!"):
    model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        apply_sigmoid=True,
        sigmoid_sensitivity=None
    )

  with pytest.raises(ValueError, match=r"Value for 'sigmoid_sensitivity' is invalid! It needs to be in interval \[0.5, 1.5\]."):
    model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        apply_sigmoid=True,
        sigmoid_sensitivity=0.4
    )

  with pytest.raises(ValueError, match=r"Value for 'sigmoid_sensitivity' is invalid! It needs to be in interval \[0.5, 1.5\]."):
    model.predict_species_within_audio_file(
        TEST_FILE_WAV,
        apply_sigmoid=True,
        sigmoid_sensitivity=1.6
    )


def test_example_interval_count_is_40_on_0_overlap(model: ModelV2M4Base):
  res = model.predict_species_within_audio_file(
    TEST_FILE_WAV, min_confidence=0, apply_sigmoid=False, chunk_overlap_s=0)
  file_dur_s = 120
  chunk_size = 3
  assert len(res) == file_dur_s / chunk_size == 40
  assert (0, 3) in res
  assert (66, 69) in res
  assert (117, 120) in res
  assert list(res.keys())[0] == (0, 3)
  assert list(res.keys())[-1] == (117, 120)


def test_example_interval_count_is_60_on_1_overlap(model: ModelV2M4Base):
  res = model.predict_species_within_audio_file(
    TEST_FILE_WAV, min_confidence=0, apply_sigmoid=False, chunk_overlap_s=1)
  assert len(res) == 60
  assert (0, 3) in res
  assert (2, 5) in res
  assert (66, 69) in res
  assert (116, 119) in res
  assert (118, 120) in res
  assert (119, 120) not in res
  assert list(res.keys())[0] == (0, 3)
  assert list(res.keys())[-1] == (118, 120)


def test_example_interval_count_is_118_on_2_overlap(model: ModelV2M4Base):
  res = model.predict_species_within_audio_file(
    TEST_FILE_WAV, min_confidence=0, apply_sigmoid=False, chunk_overlap_s=2)
  assert len(res) == 118
  assert (0, 3) in res
  assert (2, 5) in res
  assert (66, 69) in res
  assert (67, 70) in res
  assert (116, 119) in res
  assert (117, 120) in res
  assert (118, 120) not in res
  assert list(res.keys())[0] == (0, 3)
  assert list(res.keys())[-1] == (117, 120)


def test_example_batch_size_4_does_not_change_results(model: ModelV2M4Base):
  res = model.predict_species_within_audio_file(
    TEST_FILE_WAV, min_confidence=0, apply_sigmoid=False, chunk_overlap_s=0, batch_size=4)
  assert len(res) == 40

  assert list(res[(0, 3)].keys()) == ['species2', 'species3', 'species0', 'species1']
  assert list(res[(0, 3)].values()) == [0.3, 0.2, 0.1, 0.1]

  assert list(res[(66, 69)].keys()) == ['species2', 'species3', 'species0', 'species1']
  assert list(res[(66, 69)].values()) == [0.3, 0.2, 0.1, 0.1]

  assert list(res[(117, 120)].keys()) == ['species2', 'species3', 'species0', 'species1']
  assert list(res[(117, 120)].values()) == [0.3, 0.2, 0.1, 0.1]


def test_predictions_are_sorted_correctly(model: ModelV2M4Base):
  res = model.predict_species_within_audio_file(
    TEST_FILE_WAV, min_confidence=0, apply_sigmoid=False)

  # score: desc
  # name: asc

  assert list(res[(0, 3)].keys()) == ['species2', 'species3', 'species0', 'species1']
  assert list(res[(0, 3)].values()) == [0.3, 0.2, 0.1, 0.1]

  assert list(res[(66, 69)].keys()) == ['species2', 'species3', 'species0', 'species1']
  assert list(res[(66, 69)].values()) == [0.3, 0.2, 0.1, 0.1]

  assert list(res[(117, 120)].keys()) == ['species2', 'species3', 'species0', 'species1']
  assert list(res[(117, 120)].values()) == [0.3, 0.2, 0.1, 0.1]


def test_intervals_are_float(model: ModelV2M4Base):
  res = model.predict_species_within_audio_file(
    TEST_FILE_WAV, min_confidence=0, apply_sigmoid=False)

  actual_keys = list(res.keys())
  assert (0, 3) == actual_keys[0]
  assert isinstance(actual_keys[0][0], float)
  assert isinstance(actual_keys[0][1], float)
  assert isinstance(actual_keys[20][0], float)
  assert isinstance(actual_keys[20][1], float)
  assert isinstance(actual_keys[39][0], float)
  assert isinstance(actual_keys[39][1], float)
  assert len(actual_keys) == 40


def test_filter_species_filters_species(model: ModelV2M4Base):
  res = model.predict_species_within_audio_file(
    TEST_FILE_WAV, min_confidence=0, apply_sigmoid=False, filter_species={'species1', 'species2'})

  assert list(res[(0, 3)].keys()) == ['species2', 'species1']
  assert list(res[(0, 3)].values()) == [0.3, 0.1]

  assert list(res[(66, 69)].keys()) == ['species2', 'species1']
  assert list(res[(66, 69)].values()) == [0.3, 0.1]

  assert list(res[(117, 120)].keys()) == ['species2', 'species1']
  assert list(res[(117, 120)].values()) == [0.3, 0.1]


def test_apply_sigmoid_0p5(model: ModelV2M4Base):
  res = model.predict_species_within_audio_file(
    TEST_FILE_WAV, min_confidence=0, apply_sigmoid=True, sigmoid_sensitivity=0.5)

  npt.assert_almost_equal(
    list(res[(0, 3)].values()),
    [0.5374298453437496, 0.52497918747894, 0.5124973964842103, 0.5124973964842103],
    decimal=7,
  )


def test_apply_sigmoid_1p5(model: ModelV2M4Base):
  res = model.predict_species_within_audio_file(
    TEST_FILE_WAV, min_confidence=0, apply_sigmoid=True, sigmoid_sensitivity=1.5)

  assert list(res[(0, 3)].keys()) == ['species2', 'species3', 'species0', 'species1']
  npt.assert_almost_equal(
    list(res[(0, 3)].values()),
    [0.610639233949222, 0.574442516811659, 0.5374298453437496, 0.5374298453437496],
    decimal=7,
  )


def create_ground_truth_test_file(model: ModelV2M4Base, path: Path):
  test_cases = [
    AudioTestCase(),
    AudioTestCase(min_confidence=0.3),
    AudioTestCase(min_confidence=0.3, chunk_overlap_s=1),
    AudioTestCase(chunk_overlap_s=0.5),
    AudioTestCase(chunk_overlap_s=1),
    AudioTestCase(chunk_overlap_s=2),
    AudioTestCase(batch_size=8),
    AudioTestCase(bandpass_fmin=5_000),
    AudioTestCase(bandpass_fmax=10_000),
    AudioTestCase(use_bandpass=False),
    AudioTestCase(apply_sigmoid=False),
    AudioTestCase(sigmoid_sensitivity=0.5),
    AudioTestCase(filter_species={'Poecile atricapillus_Black-capped Chickadee', 'Engine_Engine'}),
    AudioTestCase(min_confidence=0.3, chunk_overlap_s=0.5, batch_size=8, bandpass_fmin=5_000, bandpass_fmax=10_000,
                  sigmoid_sensitivity=0.5),
  ]

  results = []
  for test_case in tqdm(test_cases):
    gt = predict_species_within_audio_file(test_case, model, TEST_FILE_WAV)
    gt_np, _, _ = convert_predictions_to_numpy(gt)
    n_predictions = np.count_nonzero(gt_np)
    assert n_predictions >= 5
    test_case_dict = asdict(test_case)
    results.append((test_case_dict, gt))

  with path.open("wb") as f:
    pickle.dump(results, f)


def model_test_soundscape_predictions_are_globally_correct(model: ModelV2M4Base, /, *, precision: int):
  with TEST_PATH.open("rb") as f:
    test_cases: List[Tuple[Dict, SpeciesPredictions]] = pickle.load(f)

  for test_case_dict, gt in tqdm(test_cases):
    test_case = AudioTestCase(**test_case_dict)
    res = predict_species_within_audio_file(test_case,
                                            model, TEST_FILE_WAV)
    assert species_predictions_are_equal(res, gt, precision=precision)


def model_minimum_test_soundscape_predictions_are_correct(model: ModelV2M4Base, /, *, precision: int):
  res = model.predict_species_within_audio_file(
    TEST_FILE_WAV, min_confidence=0)

  assert list(res[(0, 3)].keys())[0] == 'Poecile atricapillus_Black-capped Chickadee'
  npt.assert_almost_equal(
    res[(0, 3)]['Poecile atricapillus_Black-capped Chickadee'],
    0.81405735,  # 0.8140561 TFLite
    decimal=precision
  )

  assert list(res[(66, 69)].keys())[0] == 'Engine_Engine'
  npt.assert_almost_equal(
    res[(66, 69)]['Engine_Engine'],
    0.08609824,  # 0.0861028 TFLite
    decimal=precision
  )

  assert list(res[(117, 120)].keys())[0] == 'Spinus tristis_American Goldfinch'
  npt.assert_almost_equal(
    res[(117, 120)]['Spinus tristis_American Goldfinch'],
    0.39240596,  # 0.39239582 TFLite
    decimal=precision
  )
  assert len(res) == 40


def model_test_identical_predictions_return_same_result(model: ModelV2M4Base):
  res1 = model.predict_species_within_audio_file(
    TEST_FILE_WAV, min_confidence=0)

  res2 = model.predict_species_within_audio_file(
    TEST_FILE_WAV, min_confidence=0)

  assert species_predictions_are_equal(res1, res2, precision=7)


if __name__ == "__main__":
  # global ground truth is created using protobuf CPU model
  m = ModelV2M4Protobuf(language="en_us", custom_device="/device:CPU:0")
  create_ground_truth_test_file(m, TEST_PATH)
