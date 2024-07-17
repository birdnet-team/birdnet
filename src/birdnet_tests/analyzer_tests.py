
from pathlib import Path

from birdnet.analyzer import analyze_file


def test_analyse_sample():
  path = Path("example/soundscape.wav")
  res = analyze_file(path)
  assert res == {}


test_analyse_sample()
