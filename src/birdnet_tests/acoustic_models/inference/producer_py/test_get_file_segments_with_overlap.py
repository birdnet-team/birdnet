from math import ceil

import numpy as np
import numpy.testing
import pytest
import soundfile

from birdnet.acoustic_models.inference.producer import (
  get_file_segments_with_overlap,
)
from birdnet_tests.test_files import AUDIO_FORMATS_DIR, TEST_FILE_LONG


def format_can_be_read(filename: str) -> bool:
  inp = AUDIO_FORMATS_DIR / filename
  inp_sf = soundfile.info(inp)
  res = list(
    get_file_segments_with_overlap(
      inp,
      inp_sf.frames,
      inp_sf.samplerate,
      segment_duration_s=3,
      overlap_duration_s=0,
      target_sample_rate=48000,
      speed=1,
    )
  )
  result = len(res) == 40
  return result


def test_mono_can_be_read() -> None:
  assert format_can_be_read("soundscape.wav")


def test_stereo_can_be_read() -> None:
  assert format_can_be_read("soundscape_stereo.wav")


def test_three_channels_can_be_read() -> None:
  assert format_can_be_read("soundscape_3ch.wav")


def test_aac_can_not_be_read() -> None:
  with pytest.raises(
    soundfile.LibsndfileError,
  ):
    format_can_be_read("soundscape.aac")


def test_aifc_can_be_read() -> None:
  assert format_can_be_read("soundscape.aifc")


def test_aiff_can_be_read() -> None:
  assert format_can_be_read("soundscape.aiff")


def test_au_can_be_read() -> None:
  assert format_can_be_read("soundscape.au")


def test_flac_can_be_read() -> None:
  assert format_can_be_read("soundscape.flac")


def test_m4a_can_not_be_read() -> None:
  with pytest.raises(
    soundfile.LibsndfileError,
  ):
    format_can_be_read("soundscape.m4a")


def test_mp3_can_be_read() -> None:
  assert format_can_be_read("soundscape.mp3")


def test_ogg_can_be_read() -> None:
  assert format_can_be_read("soundscape.ogg")


def test_opus_can_be_read() -> None:
  assert format_can_be_read("soundscape.opus")


def test_wav_can_be_read() -> None:
  assert format_can_be_read("soundscape.wav")
  assert format_can_be_read("soundscape_ulaw.wav")
  assert format_can_be_read("soundscape_alaw.wav")
  assert format_can_be_read("soundscape_24bit.wav")


def test_wma_can_not_be_read() -> None:
  with pytest.raises(
    soundfile.LibsndfileError,
  ):
    format_can_be_read("soundscape.wma")


def get_segments(
  seg: float = 3, overlap: float = 0, sr: int = 48_000, speed: float = 1.0
) -> list:
  inp_sf = soundfile.info(TEST_FILE_LONG)

  return list(
    get_file_segments_with_overlap(
      TEST_FILE_LONG,
      inp_sf.frames,
      inp_sf.samplerate,
      segment_duration_s=seg,
      overlap_duration_s=overlap,
      target_sample_rate=sr,
      speed=speed,
    )
  )


def test_segments_dont_include_frames_from_previous_segments() -> None:
  result = get_segments()

  assert len(result) == 40

  close_vals = []
  for i in range(1, len(result)):
    end_of_previous_segment = result[i - 1][-1]
    start_of_current_segment = result[i][0]

    close_vals.append(
      numpy.allclose(
        end_of_previous_segment,
        start_of_current_segment,
      )
    )

  assert not np.all(close_vals)


def test_shape_is_three_times_sample_rate() -> None:
  result = get_segments()

  assert len(result) == 40
  for i in range(len(result)):
    assert result[i].shape == (3 * 48000,)


def test_result_dtype_is_float32() -> None:
  result = get_segments()

  assert len(result) == 40
  for i in range(len(result)):
    assert result[i].dtype == np.float32


def test_downsamling_to_22050() -> None:
  result = get_segments(sr=22050)

  assert len(result) == 40
  for i in range(len(result)):
    assert result[i].shape == (3 * 22050,)
    assert result[i].dtype == np.float32


def test_upsampling_to_96000() -> None:
  result = get_segments(sr=96000)

  assert len(result) == 40
  for i in range(len(result)):
    assert result[i].shape == (3 * 96000,)
    assert result[i].dtype == np.float32


# region overlap


def test_overlap_0() -> None:
  result = get_segments(overlap=0)

  assert len(result) == 40

  numpy.testing.assert_allclose(result[0][0], -0.0022888184, rtol=1e-5)  # same
  numpy.testing.assert_allclose(result[0][-1], 0.001159668, rtol=1e-5)  # same
  numpy.testing.assert_allclose(result[1][0], 0.001159668, rtol=1e-5)
  numpy.testing.assert_allclose(result[1][-1], 0.0010681152, rtol=1e-5)
  numpy.testing.assert_allclose(result[20][0], 0.0014648438, rtol=1e-5)  # same
  numpy.testing.assert_allclose(result[20][-1], 0.0034484863, rtol=1e-5)  # same
  numpy.testing.assert_allclose(result[-1][0], 0.0009765625, rtol=1e-5)
  numpy.testing.assert_allclose(result[-1][-1], -0.00012207031, rtol=1e-5)  # same

  assert len(result[-1]) == 48000 * (3 - 0)


def test_overlap_1() -> None:
  result = get_segments(overlap=1)

  assert len(result) == 60

  numpy.testing.assert_allclose(result[0][0], -0.0022888184, rtol=1e-5)  # same
  numpy.testing.assert_allclose(result[0][-1], 0.001159668, rtol=1e-5)  # same
  numpy.testing.assert_allclose(result[1][0], -0.0011901855, rtol=1e-5)
  numpy.testing.assert_allclose(result[1][-1], 0.001159668, rtol=1e-5)
  numpy.testing.assert_allclose(result[30][0], 0.0014648438, rtol=1e-5)  # same
  numpy.testing.assert_allclose(result[30][-1], 0.0034484863, rtol=1e-5)  # same
  numpy.testing.assert_allclose(result[-1][0], -0.0010070801, rtol=1e-5)
  numpy.testing.assert_allclose(result[-1][-1], -0.00012207031, rtol=1e-5)  # same

  assert len(result[-1]) == 48000 * (3 - 1)


def test_overlap_2() -> None:
  result = get_segments(overlap=2)

  assert len(result) == 120

  numpy.testing.assert_allclose(result[0][0], -0.0022888184, rtol=1e-5)  # same
  numpy.testing.assert_allclose(result[0][-1], 0.001159668, rtol=1e-5)  # same
  numpy.testing.assert_allclose(result[1][0], 0.00088500977, rtol=1e-5)
  numpy.testing.assert_allclose(result[1][-1], -0.0024719238, rtol=1e-5)
  numpy.testing.assert_allclose(result[60][0], 0.0014648438, rtol=1e-5)  # same
  numpy.testing.assert_allclose(result[60][-1], 0.0034484863, rtol=1e-5)  # same
  numpy.testing.assert_allclose(
    result[-1][0], -0.00061035156, rtol=1e-5
  )  # only difference is here
  numpy.testing.assert_allclose(result[-1][-1], -0.00012207031, rtol=1e-5)  # same

  assert len(result[-1]) == 48000 * (3 - 2)


def test_overlap_2_5() -> None:
  result = get_segments(overlap=2.5)

  assert len(result) == 240

  numpy.testing.assert_allclose(result[0][0], -0.0022888184, rtol=1e-5)  # same
  numpy.testing.assert_allclose(result[0][-1], 0.001159668, rtol=1e-5)  # same
  numpy.testing.assert_allclose(result[1][0], -0.0022888184, rtol=1e-5)
  numpy.testing.assert_allclose(result[1][-1], 0.0006713867, rtol=1e-5)
  numpy.testing.assert_allclose(result[120][0], 0.0014648438, rtol=1e-5)  # same
  numpy.testing.assert_allclose(result[120][-1], 0.0034484863, rtol=1e-5)  # same
  numpy.testing.assert_allclose(result[-1][0], 0, rtol=1e-5)  # only difference is here
  numpy.testing.assert_allclose(result[-1][-1], -0.00012207031, rtol=1e-5)  # same
  assert len(result[-1]) == 48000 * (3 - 2.5)


# endregion


def test_double_speed() -> None:
  result = get_segments(speed=2)

  assert len(result) == 20
  for i in range(len(result)):
    assert result[i].shape == (3 * 48000,)


def test_half_speed() -> None:
  result = get_segments(speed=1 / 2)

  assert len(result) == 80
  for i in range(len(result)):
    assert result[i].shape == (3 * 48000,)


def test_one_third_speed() -> None:
  result = get_segments(speed=1 / 3)

  assert len(result) == 40 * 3
  for i in range(len(result)):
    assert result[i].shape == (3 * 48000,)


def generic_speed_test(speed: float) -> None:
  result = get_segments(speed=speed)
  n_full_segments, half_segment = divmod(40, speed)
  assert len(result) == n_full_segments + ceil(half_segment)
  for i in range(int(n_full_segments)):
    assert result[i].shape == (3 * 48000,)

  if half_segment > 0:
    assert result[-1].shape < (3 * 48000,)


def test_triple_speed() -> None:
  generic_speed_test(speed=3)


def test_float_1_dec_speed() -> None:
  generic_speed_test(speed=0.9)


def test_float_2_dec_speed() -> None:
  generic_speed_test(speed=0.19)


def test_float_3_dec_speed() -> None:
  generic_speed_test(speed=0.119)


def test_float_4_dec_speed() -> None:
  generic_speed_test(speed=0.1119)


def test_float_5_dec_speed() -> None:
  generic_speed_test(speed=0.11119)
