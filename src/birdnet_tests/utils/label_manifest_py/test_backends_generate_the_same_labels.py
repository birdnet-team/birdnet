import hashlib
from pathlib import Path

import pytest

from birdnet.acoustic.models.v3_0.model import AcousticDownloaderBaseV3_0
from birdnet.acoustic.models.v3_0.onnx import AcousticOnnxDownloaderV3_0
from birdnet.acoustic.models.v3_0.pb import AcousticPBDownloaderV3_0
from birdnet.acoustic.models.v3_0.pt import AcousticPTDownloaderV3_0
from birdnet.acoustic.models.v3_0.tf import AcousticTFDownloaderV3_0
from birdnet.geo.models.v3_0.model import GeoDownloaderBaseV3_0
from birdnet.geo.models.v3_0.onnx import GeoOnnxDownloaderV3_0
from birdnet.geo.models.v3_0.pb import GeoPBDownloaderV3_0
from birdnet.geo.models.v3_0.pt import GeoPTDownloaderV3_0
from birdnet.geo.models.v3_0.tf import GeoTFDownloaderV3_0

_ACOUSTIC = [
  AcousticTFDownloaderV3_0,
  AcousticPBDownloaderV3_0,
  AcousticPTDownloaderV3_0,
  AcousticOnnxDownloaderV3_0,
]
_GEO = [
  GeoTFDownloaderV3_0,
  GeoPBDownloaderV3_0,
  GeoPTDownloaderV3_0,
  GeoOnnxDownloaderV3_0,
]


def _digest(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_backends_agree(
  downloaders: list[type[AcousticDownloaderBaseV3_0] | type[GeoDownloaderBaseV3_0]],
) -> None:
  digests: dict[str, str] = {}
  for downloader in downloaders:
    downloader.ensure_labels_available()
    lang_file = downloader.get_lang_file("en_us")
    digests[downloader.__name__] = _digest(lang_file)

  assert len(set(digests.values())) == 1, (
    f"backends of one model disagree about the species names they serve: {digests}"
  )


@pytest.mark.no_tf
@pytest.mark.load_model
def test_all_acoustic_backends_generate_identical_labels() -> None:
  """The observed symptom: one backend served 1,725 different common names.

  Only catches backends that disagree, so it is paired with the manifest tests,
  which catch every backend being stale together.
  """
  _assert_backends_agree(_ACOUSTIC)


@pytest.mark.no_tf
@pytest.mark.load_model
def test_all_geo_backends_generate_identical_labels() -> None:
  _assert_backends_agree(_GEO)
