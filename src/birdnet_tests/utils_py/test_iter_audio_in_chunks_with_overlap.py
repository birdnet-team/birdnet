import numpy as np
import numpy.testing as npt

from birdnet.utils import iter_audio_in_chunks_with_overlap
from birdnet_tests.helper import TEST_FILE_WAV, TEST_FILES_DIR


def test_wav():
  result = list(iter_audio_in_chunks_with_overlap(
    TEST_FILE_WAV,
    chunk_duration_s=3,
    overlap_duration_s=0,
    target_sample_rate=48000,
  ))
  assert len(result) == 120 / 3
  first_result = result[0]
  assert first_result[0] == (0, 3)
  assert first_result[1].shape == (3 * 48000,)
  assert first_result[1].dtype == np.float32
  npt.assert_almost_equal(result[0][1][0], -0.0022888184, decimal=10)
  npt.assert_almost_equal(result[-1][1][-1], -0.00012207031, decimal=10)


def test_flac():
  result = list(iter_audio_in_chunks_with_overlap(
    TEST_FILES_DIR / "soundscape.flac",
    chunk_duration_s=3,
    overlap_duration_s=0,
    target_sample_rate=48000,
  ))
  assert len(result) == 120 / 3
  first_result = result[0]
  assert first_result[0] == (0, 3)
  assert first_result[1].shape == (3 * 48000,)
  assert first_result[1].dtype == np.float32
  npt.assert_almost_equal(result[0][1][0], -0.0022888184, decimal=10)
  npt.assert_almost_equal(result[-1][1][-1], -0.00012207031, decimal=10)


def test_mp3():
  result = list(iter_audio_in_chunks_with_overlap(
    TEST_FILES_DIR / "soundscape.mp3",
    chunk_duration_s=3,
    overlap_duration_s=0,
    target_sample_rate=48000,
  ))
  assert len(result) == 120 / 3
  first_result = result[0]
  assert first_result[0] == (0, 3)
  assert first_result[1].shape == (3 * 48000,)
  assert first_result[1].dtype == np.float32

  # through compression it changed compared to wav or flac
  npt.assert_almost_equal(result[0][1][0], -1.5780089e-03, decimal=10)
  npt.assert_almost_equal(result[-1][1][-1], -5.227531e-05, decimal=10)


def test_ogg():
  result = list(iter_audio_in_chunks_with_overlap(
    TEST_FILES_DIR / "soundscape.ogg",
    chunk_duration_s=3,
    overlap_duration_s=0,
    target_sample_rate=48000,
  ))
  assert len(result) == 120 / 3
  first_result = result[0]
  assert first_result[0] == (0, 3)
  assert first_result[1].shape == (3 * 48000,)
  assert first_result[1].dtype == np.float32

  npt.assert_almost_equal(result[0][1][0], -0.0022254132, decimal=10)
  npt.assert_almost_equal(result[-1][1][-1], -0.00018806655, decimal=10)



def test_overlap_0():
  result = list(iter_audio_in_chunks_with_overlap(
    TEST_FILE_WAV,
    chunk_duration_s=3,
    overlap_duration_s=0,
    target_sample_rate=48000,
  ))
  assert len(result) == 40
  first_result = result[0]
  assert first_result[0] == (0, 3)
  assert first_result[1].shape == (3 * 48000,)
  assert first_result[1].dtype == np.float32
  npt.assert_almost_equal(result[0][1][0], -0.0022888184, decimal=10)
  npt.assert_almost_equal(result[-1][1][-1], -0.00012207031, decimal=10)
  assert [res[0] for res in result[-3:]] == [
    (111, 114),
    (114, 117),
    (117, 120),
  ]


def test_overlap_1():
  result = list(iter_audio_in_chunks_with_overlap(
    TEST_FILE_WAV,
    chunk_duration_s=3,
    overlap_duration_s=1,
    target_sample_rate=48000,
  ))
  assert len(result) == 60
  first_result = result[0]
  assert first_result[0] == (0, 3)
  assert first_result[1].shape == (3 * 48000,)
  assert first_result[1].dtype == np.float32
  npt.assert_almost_equal(result[0][1][0], -0.0022888184, decimal=10)
  npt.assert_almost_equal(result[-1][1][-1], -0.00012207031, decimal=10)
  assert [res[0] for res in result[-3:]] == [
    (114, 117),
    (116, 119),
    (118, 120),
  ]


def test_overlap_2():
  result = list(iter_audio_in_chunks_with_overlap(
    TEST_FILE_WAV,
    chunk_duration_s=3,
    overlap_duration_s=2,
    target_sample_rate=48000,
  ))
  assert len(result) == 118
  first_result = result[0]
  assert first_result[0] == (0, 3)
  assert first_result[1].shape == (3 * 48000,)
  assert first_result[1].dtype == np.float32
  npt.assert_almost_equal(result[0][1][0], -0.0022888184, decimal=10)
  npt.assert_almost_equal(result[-1][1][-1], -0.00012207031, decimal=10)
  assert [res[0] for res in result[-3:]] == [
    (115, 118),
    (116, 119),
    (117, 120),
  ]


def test_overlap_1p5():
  result = list(iter_audio_in_chunks_with_overlap(
    TEST_FILE_WAV,
    chunk_duration_s=3,
    overlap_duration_s=1.5,
    target_sample_rate=48000,
  ))
  assert len(result) == 79
  first_result = result[0]
  assert first_result[0] == (0, 3)
  assert first_result[1].shape == (3 * 48000,)
  assert first_result[1].dtype == np.float32
  npt.assert_almost_equal(result[0][1][0], -0.0022888184, decimal=10)
  npt.assert_almost_equal(result[-1][1][-1], -0.00012207031, decimal=10)
  assert [res[0] for res in result[-3:]] == [
    (114, 117),
    (115.5, 118.5),
    (117, 120.0),
  ]
