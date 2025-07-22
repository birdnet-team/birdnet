from birdnet.acoustic_models.v2_4.pb import AcousticPBModelV2_4
from birdnet.acoustic_models.v2_4.tf import AcousticTFModelV2_4
from birdnet.geo_models.v2_4.pb import GeoPBModelV2_4
from birdnet.geo_models.v2_4.tf import GeoTFModelV2_4
from birdnet.model_loader import load


def test_types_are_correct():
  assert load("acoustic", "2.4", "pb") is AcousticPBModelV2_4
  assert load("acoustic", "2.4", "tf") is AcousticTFModelV2_4
  assert load("geo", "2.4", "pb") is GeoPBModelV2_4
  assert load("geo", "2.4", "tf") is GeoTFModelV2_4
  