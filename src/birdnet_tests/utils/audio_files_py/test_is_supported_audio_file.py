from pathlib import Path

import pytest

from birdnet.utils.audio_files import is_supported_audio_file

pytestmark = pytest.mark.no_tf


def test_is_supported_audio_file(tmp_path: Path) -> None:
  wav = tmp_path / "a.wav"
  wav.touch()
  txt = tmp_path / "b.txt"
  txt.touch()
  upper = tmp_path / "c.FLAC"
  upper.touch()

  assert is_supported_audio_file(wav)
  assert is_supported_audio_file(upper)
  assert not is_supported_audio_file(txt)
