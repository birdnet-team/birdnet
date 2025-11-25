
import pytest

from birdnet.acoustic_models.inference.producer import (
  read_file_in_mono,
)
from birdnet_tests.test_files import AUDIO_FORMATS_DIR


def assert_format_can_be_read(filename: str) -> None:
  inp = AUDIO_FORMATS_DIR / filename
  res = read_file_in_mono(
    audio_path=inp,
    start_samples=3 * 48_000,
    end_samples=6 * 48_000,
  )
  assert res.shape == (3 * 48_000,)


def test_mono_can_be_read() -> None:
  assert_format_can_be_read("soundscape.wav")


def test_stereo_can_be_read() -> None:
  assert_format_can_be_read("soundscape_stereo.wav")


def test_three_channels_can_be_read() -> None:
  assert_format_can_be_read("soundscape_3ch.wav")


def test_aac_can_not_be_read() -> None:
  with pytest.raises(
    AssertionError,
  ):
    assert_format_can_be_read("soundscape.aac")


def test_aifc_can_be_read() -> None:
  assert_format_can_be_read("soundscape.aifc")


def test_aiff_can_be_read() -> None:
  assert_format_can_be_read("soundscape.aiff")


def test_au_can_be_read() -> None:
  assert_format_can_be_read("soundscape.au")


def test_flac_can_be_read() -> None:
  assert_format_can_be_read("soundscape.flac")


def test_m4a_can_not_be_read() -> None:
  with pytest.raises(
    AssertionError,
  ):
    assert_format_can_be_read("soundscape.m4a")


def test_mp3_can_be_read() -> None:
  assert_format_can_be_read("soundscape.mp3")


def test_ogg_can_be_read() -> None:
  assert_format_can_be_read("soundscape.ogg")


def test_opus_can_be_read() -> None:
  assert_format_can_be_read("soundscape.opus")


def test_wav_can_be_read() -> None:
  assert_format_can_be_read("soundscape.wav")


def test_wav_ulaw_can_be_read() -> None:
  assert_format_can_be_read("soundscape_ulaw.wav")


def test_wav_alaw_can_be_read() -> None:
  assert_format_can_be_read("soundscape_alaw.wav")


def test_wav_24bit_can_be_read() -> None:
  assert_format_can_be_read("soundscape_24bit.wav")


def test_wma_can_not_be_read() -> None:
  with pytest.raises(
    AssertionError,
  ):
    assert_format_can_be_read("soundscape.wma")
