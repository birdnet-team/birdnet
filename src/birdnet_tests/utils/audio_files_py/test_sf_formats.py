import pytest

from birdnet.utils.audio_files import SF_FORMATS

pytestmark = pytest.mark.no_tf


def test_all_formats_from_sf_are_in_global_var() -> None:
  import soundfile as sf

  possible = {f".{suffix}" for suffix in sf.available_formats()} | {
    ".OPUS",
    ".AIFC",
  }

  assert possible == SF_FORMATS
