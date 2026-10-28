"""`SequentialFileReader` yields what reading each window separately yields."""

import numpy as np
import pytest
import soundfile

from birdnet.acoustic.inference.core.producer import (
  SequentialFileReader,
  get_segments_with_overlap_samples,
  read_file_in_mono,
)
from birdnet.utils.segments import duration_as_samples
from birdnet_tests.test_files import AUDIO_FORMATS_DIR

pytestmark = pytest.mark.no_tf

LOSSLESS = ["soundscape.wav", "soundscape_stereo.wav", "soundscape.flac"]


def _windows(n_frames: int, sr: int, overlap_s: float, speed: float) -> list:
  return list(
    get_segments_with_overlap_samples(
      n_frames,
      segment_samples=duration_as_samples(3.0, sr),
      overlap_samples=duration_as_samples(overlap_s, sr),
      speed=speed,
    )
  )


@pytest.mark.parametrize("filename", LOSSLESS)
@pytest.mark.parametrize(
  ("overlap_s", "speed"), [(0.0, 1.0), (1.5, 1.0), (2.9, 1.0), (0.0, 0.5), (1.0, 2.0)]
)
def test_windows_match_separate_reads(
  filename: str, overlap_s: float, speed: float
) -> None:
  path = AUDIO_FORMATS_DIR / filename
  info = soundfile.info(path)
  windows = _windows(info.frames, info.samplerate, overlap_s, speed)

  with soundfile.SoundFile(path) as audio_file:
    reader = SequentialFileReader(audio_file)
    for start, end, _ in windows:
      expected = read_file_in_mono(start, end, path)
      np.testing.assert_array_equal(reader(start, end), expected)


def test_backward_jump_seeks() -> None:
  path = AUDIO_FORMATS_DIR / "soundscape.wav"
  with soundfile.SoundFile(path) as audio_file:
    reader = SequentialFileReader(audio_file)
    reader(48_000, 96_000)
    np.testing.assert_array_equal(reader(0, 48_000), read_file_in_mono(0, 48_000, path))


@pytest.mark.parametrize(
  ("filename", "atol"),
  # libsndfile's MP3 decoder rounds differently when read in pieces; Vorbis and
  # Opus decode identically, while seeking to each window misses for them (by
  # up to 9.2e-3 for Vorbis at 2.9 s overlap).
  [("soundscape.ogg", 0.0), ("soundscape.opus", 0.0), ("soundscape.mp3", 1e-7)],
)
def test_lossy_windows_match_a_decode_of_the_whole_file(
  filename: str, atol: float
) -> None:
  path = AUDIO_FORMATS_DIR / filename
  info = soundfile.info(path)
  whole, _ = soundfile.read(path, dtype="float32")
  whole = whole if whole.ndim == 1 else whole.mean(axis=1, dtype=np.float32)

  with soundfile.SoundFile(path) as audio_file:
    reader = SequentialFileReader(audio_file)
    for start, end, _ in _windows(info.frames, info.samplerate, 2.9, 1.0):
      np.testing.assert_allclose(
        reader(start, end), whole[start:end], rtol=0, atol=atol
      )


def test_returned_window_survives_in_place_mutation() -> None:
  path = AUDIO_FORMATS_DIR / "soundscape.wav"
  n = duration_as_samples(3.0, soundfile.info(path).samplerate)
  overlap = n // 2
  with soundfile.SoundFile(path) as audio_file:
    reader = SequentialFileReader(audio_file)
    first = reader(0, n)
    first[:] = 0.0
    second = reader(n - overlap, 2 * n - overlap)
  np.testing.assert_array_equal(
    second, read_file_in_mono(n - overlap, 2 * n - overlap, path)
  )
