"""Every official downloader refuses a label file that does not match its model.

Run against the real, cached files: the model's class count is patched to an
impossible value, so the file the downloader just accepted must be refused.
"""

from __future__ import annotations

import pytest

from birdnet.acoustic.models.perch_v2.model import AcousticModelPerchV2
from birdnet.acoustic.models.perch_v2.pb import AcousticPBDownloaderPerchV2
from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet.acoustic.models.v2_4.pb import AcousticPBDownloaderV2_4
from birdnet.acoustic.models.v2_4.tf import AcousticTFDownloaderV2_4
from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.acoustic.models.v3_0.onnx import AcousticOnnxDownloaderV3_0
from birdnet.acoustic.models.v3_0.pb import AcousticPBDownloaderV3_0
from birdnet.acoustic.models.v3_0.pt import AcousticPTDownloaderV3_0
from birdnet.acoustic.models.v3_0.tf import AcousticTFDownloaderV3_0
from birdnet.core.base import ModelBase
from birdnet.geo.models.v2_4.model import GeoModelV2_4
from birdnet.geo.models.v2_4.pb import GeoPBDownloaderV2_4
from birdnet.geo.models.v2_4.tf import GeoTFDownloaderV2_4
from birdnet.geo.models.v3_0.model import GeoModelV3_0
from birdnet.geo.models.v3_0.onnx import GeoOnnxDownloaderV3_0
from birdnet.geo.models.v3_0.pb import GeoPBDownloaderV3_0
from birdnet.geo.models.v3_0.pt import GeoPTDownloaderV3_0
from birdnet.geo.models.v3_0.tf import GeoTFDownloaderV3_0
from birdnet_tests.helper import (
  ensure_not_intel_macos_or_skip,
  ensure_tf_2_20_or_skip,
  ensure_v3_0_torch_backend_or_skip,
)

_CASES: list[tuple[type[ModelBase], type, tuple[str, ...]]] = [
  (AcousticModelV2_4, AcousticTFDownloaderV2_4, ("en_us", "fp32")),
  (AcousticModelV2_4, AcousticPBDownloaderV2_4, ("en_us",)),
  (AcousticModelV3_0, AcousticTFDownloaderV3_0, ("en_us", "fp32")),
  (AcousticModelV3_0, AcousticOnnxDownloaderV3_0, ("en_us", "fp32")),
  (AcousticModelV3_0, AcousticPTDownloaderV3_0, ("en_us", "fp32")),
  (AcousticModelV3_0, AcousticPBDownloaderV3_0, ("en_us",)),
  (AcousticModelPerchV2, AcousticPBDownloaderPerchV2, ("CPU",)),
  (GeoModelV2_4, GeoTFDownloaderV2_4, ("en_us",)),
  (GeoModelV2_4, GeoPBDownloaderV2_4, ("en_us",)),
  (GeoModelV3_0, GeoTFDownloaderV3_0, ("en_us", "fp32")),
  (GeoModelV3_0, GeoOnnxDownloaderV3_0, ("en_us", "fp32")),
  (GeoModelV3_0, GeoPTDownloaderV3_0, ("en_us", "fp32")),
  (GeoModelV3_0, GeoPBDownloaderV3_0, ("en_us",)),
]


@pytest.mark.load_model
@pytest.mark.parametrize(
  ("model", "downloader", "args"),
  _CASES,
  ids=[downloader.__name__ for _, downloader, _ in _CASES],
)
def test_a_label_file_of_another_length_is_refused(
  model: type[ModelBase],
  downloader: type,
  args: tuple[str, ...],
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  if downloader is AcousticPBDownloaderPerchV2:
    # The guards of load_perch_v2's tests: the download is only made where Perch runs.
    ensure_not_intel_macos_or_skip()
    ensure_tf_2_20_or_skip()
  if downloader is GeoPTDownloaderV3_0:
    # The geo 3.0 pt load test skips on Intel macOS; keep that lane's download set.
    ensure_v3_0_torch_backend_or_skip()
  downloader.get_model_path_and_labels(*args)
  monkeypatch.setattr(model, "get_n_classes", classmethod(lambda cls: 1))

  with pytest.raises(
    RuntimeError, match=r"has \d+ lines, but the model outputs 1 classes\."
  ):
    downloader.get_model_path_and_labels(*args)
