"""A precision missing from a downloader's table is rejected before any download
or label work starts."""

import re
import sys

import pytest

from birdnet.geo.models.v3_0.model import _GeoSingleFileDownloaderV3_0
from birdnet.geo.models.v3_0.onnx import GeoOnnxDownloaderV3_0
from birdnet.geo.models.v3_0.onnx import models as onnx_models
from birdnet.geo.models.v3_0.pt import GeoPTDownloaderV3_0
from birdnet.geo.models.v3_0.pt import models as pt_models
from birdnet.geo.models.v3_0.tf import GeoTFDownloaderV3_0
from birdnet.geo.models.v3_0.tf import models as tf_models
from birdnet.globals import VALID_MODEL_PRECISIONS

_TABLES = [
  (GeoTFDownloaderV3_0, tf_models),
  (GeoOnnxDownloaderV3_0, onnx_models),
  (GeoPTDownloaderV3_0, pt_models),
]

# Every valid precision a table lacks, so a check against the global set fails,
# plus one that is no precision at all.
_CASES = [
  pytest.param(downloader, precision, id=f"{downloader.__name__}-{precision}")
  for downloader, table in _TABLES
  for precision in [*(p for p in VALID_MODEL_PRECISIONS if p not in table), "fp8"]
]


def _must_not_be_called(*args: object, **kwargs: object) -> None:
  raise AssertionError("reached a download or label step for an unsupported precision")


@pytest.mark.no_tf
@pytest.mark.parametrize(("downloader", "precision"), _CASES)
def test_unsupported_precision_is_rejected(
  downloader: type[_GeoSingleFileDownloaderV3_0],
  precision: str,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    downloader, "ensure_labels_available", classmethod(_must_not_be_called)
  )
  # Patched wherever a class in the hierarchy binds the name, so the test holds
  # whichever module the download step lives in.
  for klass in downloader.__mro__:
    module = sys.modules[klass.__module__]
    if hasattr(module, "ensure_single_file_model"):
      monkeypatch.setattr(module, "ensure_single_file_model", _must_not_be_called)
  monkeypatch.setattr(
    "birdnet.utils.helper.ensure_single_file_model", _must_not_be_called
  )

  with pytest.raises(
    ValueError,
    match=rf"Unsupported model precision for .* model: {re.escape(precision)}\.",
  ):
    downloader.get_model_path_and_labels("en_us", precision)  # type: ignore[arg-type]
