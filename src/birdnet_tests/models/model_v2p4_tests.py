from pathlib import Path

from birdnet.models.model_v2p4 import Downloader, ModelV2p4


def test_analyse_sample():
  path = Path("example/soundscape.wav")
  model = ModelV2p4()
  res = model.analyze_file(path)
  assert res == {}

def test_get_species():
  model = ModelV2p4()
  species = model.get_species_from_location(42.5, -76.45, 4)
  assert species == {}
  

def test_downloader():
  downloader = Downloader()
  downloader.ensure_model_is_available()
test_analyse_sample()
# test_downloader()
# test_get_species()