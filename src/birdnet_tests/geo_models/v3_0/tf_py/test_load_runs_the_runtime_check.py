"""A geo v3.0 TF backend checks the runtime in its own load().

The loader runs the same check before the download, so only a backend built
directly reaches this one.
"""

from pathlib import Path

import pytest

import birdnet.core.backends as backends
import birdnet.geo.models.v3_0.tf as geo_tf
from birdnet.core.backends import TFBackend
from birdnet.geo.models.v3_0.tf import (
  GeoTFBackendFP16V3_0,
  GeoTFBackendFP32V3_0,
  GeoTFBackendInt8V3_0,
)

pytestmark = pytest.mark.no_tf

BACKENDS = pytest.mark.parametrize(
  "backend_type", [GeoTFBackendFP32V3_0, GeoTFBackendFP16V3_0, GeoTFBackendInt8V3_0]
)


def _open_model(*args: object, **kwargs: object) -> None:
  raise AssertionError("the model file was opened")


@BACKENDS
def test_load_refuses_litert_before_opening_the_model(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend_type: type[TFBackend]
) -> None:
  monkeypatch.setattr(backends, "load_tf_model", _open_model)
  backend = backend_type(
    tmp_path / "model.tflite", "CPU", False, inference_library="litert"
  )

  with pytest.raises(RuntimeError, match="not supported with ai_edge_litert"):
    backend.load()


@BACKENDS
def test_load_refuses_an_unsupported_tf_before_opening_the_model(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend_type: type[TFBackend]
) -> None:
  monkeypatch.setattr(backends, "load_tf_model", _open_model)
  monkeypatch.setattr(geo_tf, "_installed_tf_version", lambda: "2.21.0")
  backend = backend_type(
    tmp_path / "model.tflite", "CPU", False, inference_library="tflite"
  )

  with pytest.raises(RuntimeError, match=r"requires TensorFlow 2\.18 or 2\.19"):
    backend.load()
