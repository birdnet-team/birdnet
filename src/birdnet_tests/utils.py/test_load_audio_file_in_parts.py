from pathlib import Path

import numpy as np
import numpy.testing as npt

from birdnet.utils import load_audio_file_in_parts


def test_wav():
  test_files_dir = Path("src/birdnet_tests/test_files")
  result = list(load_audio_file_in_parts(test_files_dir / "soundscape.wav", 48000, 600))
  assert len(result) == 1
  assert len(result[0]) == 5760000
  assert result[0].dtype == np.float32
  npt.assert_almost_equal(result[0][0], -0.0022888184, decimal=10)
  npt.assert_almost_equal(result[0][-1], -0.00012207031, decimal=10)


def test_flac():
  test_files_dir = Path("src/birdnet_tests/test_files")
  result = list(load_audio_file_in_parts(test_files_dir / "soundscape.flac", 48000, 600))
  assert len(result) == 1
  assert len(result[0]) == 5760000
  assert result[0].dtype == np.float32
  npt.assert_almost_equal(result[0][0], -0.0022888184, decimal=10)
  npt.assert_almost_equal(result[0][-1], -0.00012207031, decimal=10)


def test_mp3():
  test_files_dir = Path("src/birdnet_tests/test_files")
  result = list(load_audio_file_in_parts(test_files_dir / "soundscape.mp3", 48000, 600))
  assert len(result) == 1
  assert len(result[0]) == 5760000
  assert result[0].dtype == np.float32
  # through compression it changed compared to wav or flac
  npt.assert_almost_equal(result[0][0], -1.5780089e-03, decimal=10)
  npt.assert_almost_equal(result[0][-1], -5.227531e-05, decimal=10)


def test_wav_two_splits():
  test_files_dir = Path("src/birdnet_tests/test_files")
  result = list(load_audio_file_in_parts(test_files_dir / "soundscape.wav", 48000, 60))
  assert len(result) == 2
  assert len(result[0]) == 5760000 / 2
  assert len(result[1]) == 5760000 / 2
  assert result[0].dtype == np.float32
  assert result[1].dtype == np.float32
  npt.assert_almost_equal(result[0][0], -0.0022888184, decimal=10)
  npt.assert_almost_equal(result[0][-1], 0.0014648438, decimal=10)
  npt.assert_almost_equal(result[1][0], 0.0014648438, decimal=10)
  npt.assert_almost_equal(result[1][-1], -0.00012207031, decimal=10)


def test_wav_one_second_splits():
  test_files_dir = Path("src/birdnet_tests/test_files")
  result = list(load_audio_file_in_parts(test_files_dir / "soundscape.wav", 48000, 1))
  assert len(result) == 120
  assert len(result[0]) == 5760000 / 120
  assert len(result[-1]) == 5760000 / 120
  assert result[0].dtype == np.float32
  assert result[-1].dtype == np.float32
  npt.assert_almost_equal(result[0][0], -0.0022888184, decimal=10)
  npt.assert_almost_equal(result[-1][-1], -0.00012207031, decimal=10)
