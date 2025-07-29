from birdnet.helper import SF_FORMATS


def test_all_formats_from_sf_are_in_global_var() -> None:
  import soundfile as sf

  possible = {f".{suffix}" for suffix in sf.available_formats()} | {
    ".OPUS",
    ".AIFC",
  }

  assert possible == SF_FORMATS
