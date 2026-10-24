"""Label setup serializes on its domain's setup lock, whatever the backend.

Other installed versions of birdnet take the same lock, so a backend that set up
its labels under a lock of its own could write a label directory while another
version is writing it too.
"""

from pathlib import Path
from types import ModuleType

import pytest

import birdnet.acoustic.models.v3_0.model as acoustic_model
import birdnet.geo.models.v3_0.model as geo_model
from birdnet.acoustic.models.v3_0.onnx import AcousticOnnxDownloaderV3_0
from birdnet.acoustic.models.v3_0.pb import AcousticPBDownloaderV3_0
from birdnet.acoustic.models.v3_0.pt import AcousticPTDownloaderV3_0
from birdnet.acoustic.models.v3_0.tf import AcousticTFDownloaderV3_0
from birdnet.geo.models.v3_0.onnx import GeoOnnxDownloaderV3_0
from birdnet.geo.models.v3_0.pb import GeoPBDownloaderV3_0
from birdnet.geo.models.v3_0.pt import GeoPTDownloaderV3_0
from birdnet.geo.models.v3_0.tf import GeoTFDownloaderV3_0
from birdnet.globals import ENV_VAR_APP_DATA
from birdnet.utils import v3_labels
from birdnet.utils.v3_labels import V3LabelDownloaderBase

_DOWNLOADERS = [
  pytest.param(downloader, module, id=downloader.__name__)
  for downloader, module in (
    (AcousticTFDownloaderV3_0, acoustic_model),
    (AcousticOnnxDownloaderV3_0, acoustic_model),
    (AcousticPTDownloaderV3_0, acoustic_model),
    (AcousticPBDownloaderV3_0, acoustic_model),
    (GeoTFDownloaderV3_0, geo_model),
    (GeoOnnxDownloaderV3_0, geo_model),
    (GeoPTDownloaderV3_0, geo_model),
    (GeoPBDownloaderV3_0, geo_model),
  )
]


class _LockTaken(Exception):
  pass


@pytest.mark.no_tf
@pytest.mark.parametrize(("downloader", "module"), _DOWNLOADERS)
def test_label_setup_takes_the_domain_setup_lock(
  downloader: type[V3LabelDownloaderBase],
  module: ModuleType,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv(ENV_VAR_APP_DATA, str(tmp_path))
  monkeypatch.setattr(
    downloader, "_check_labels_available", classmethod(lambda cls: False)
  )
  taken: list[Path] = []

  def _record(lock_dir: Path, description: str) -> None:
    taken.append(lock_dir)
    raise _LockTaken

  monkeypatch.setattr(v3_labels, "directory_lock", _record)

  with pytest.raises(_LockTaken):
    downloader.ensure_labels_available()

  assert taken == [module._setup_lock_dir()]
