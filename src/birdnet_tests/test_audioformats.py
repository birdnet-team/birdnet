from birdnet.helper import SF_FORMATS


def test_all_supported():
  import soundfile as sf

  possible = {f".{suffix}" for suffix in sf.available_formats()} | {
    ".OPUS",
    ".AIFC",
  }
  assert possible == SF_FORMATS
