from pathlib import Path

import pytest

from birdnet.utils.audio_files import get_file_formats

pytestmark = pytest.mark.no_tf


def test_get_file_formats() -> None:
  paths = {Path("a.WAV"), Path("b.mp3"), Path("c.wav")}
  assert get_file_formats(paths) == "MP3, WAV"
