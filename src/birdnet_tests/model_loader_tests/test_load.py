from birdnet.acoustic_models.v2_4.pb import AcousticPBModelV2_4
from birdnet.acoustic_models.v2_4.tf import AcousticTFModelV2_4
from birdnet.acoustic_models.v3_0.tf import AcousticTFModelV3_0
from birdnet.geo_models.v2_4.pb import GeoPBModelV2_4
from birdnet.geo_models.v2_4.tf import GeoTFModelV2_4
from birdnet.model_loader import load2


def test_types_are_correct():
  assert load2("acoustic", "2.4", "pb") is AcousticPBModelV2_4
  assert load2("acoustic", "2.4", "tf") is AcousticTFModelV2_4
  assert load2("acoustic", "3.0", "tf") is AcousticTFModelV3_0
  assert load2("geo", "2.4", "pb") is GeoPBModelV2_4
  assert load2("geo", "2.4", "tf") is GeoTFModelV2_4
