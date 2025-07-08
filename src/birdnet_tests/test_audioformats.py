from pathlib import Path

from birdnet.acoustic_models.inference.producer import (
  load_audio_in_segments_with_overlap_locked,
)
from birdnet.helper import SF_FORMATS
from birdnet.io_lock import IOLockHandler


def test_all_supported():
  import soundfile as sf

  possible = {f".{suffix}" for suffix in sf.available_formats()} | {
    ".OPUS",
    ".AIFC",
  }
  assert possible == SF_FORMATS


def test_stereo():
  inp = Path("src/birdnet_debug/audio_formats/soundscape_stereo.wav")
  res = list(
    load_audio_in_segments_with_overlap_locked(inp, IOLockHandler(False, None))
  )
  assert len(res) == 40


test_stereo()
