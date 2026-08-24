from pathlib import Path
from typing import Literal, cast

import pytest
from ordered_set import OrderedSet
from requests import ReadTimeout

from birdnet.geo.models.v2_4.model import GeoModelV2_4
from birdnet.model_loader import load
from birdnet_tests.helper import ensure_litert_or_skip


def _stub_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
  """Return a model path without downloading; the `load_model` tests cover that."""
  monkeypatch.setattr(
    "birdnet.model_loader.GeoPBDownloaderV2_4.get_model_path_and_labels",
    lambda lang: (Path("birdnet_geo_v2.4_pb"), OrderedSet(["species_a"])),
  )
  monkeypatch.setattr(
    "birdnet.model_loader.GeoTFDownloaderV2_4.get_model_path_and_labels",
    lambda lang: (Path("birdnet_geo_v2.4.tflite"), OrderedSet(["species_a"])),
  )


@pytest.mark.litert
@pytest.mark.tf  # the pb guard fires before the kwarg check
def test_pb_v2_4_with_library_raises_error() -> None:
  ensure_litert_or_skip()

  with pytest.raises(
    ValueError,
    match=r"Unexpected keyword arguments: library.",
  ):
    load("geo", "2.4", "pb", precision="fp32", library="litert")  # type: ignore


@pytest.mark.load_model
def test_v2_4_pb() -> None:
  try:
    model = load("geo", "2.4", "pb", precision="fp32")
  except ReadTimeout as e:
    pytest.fail(f"Model download timed out: {e}. Try again later.")
  assert isinstance(model, GeoModelV2_4)


@pytest.mark.load_model
def test_v2_4_tf_fp32() -> None:
  try:
    model = load("geo", "2.4", "tf", precision="fp32", library="tflite")
  except ReadTimeout as e:
    pytest.fail(f"Model download timed out: {e}. Try again later.")
  assert isinstance(model, GeoModelV2_4)


@pytest.mark.load_model
@pytest.mark.litert
def test_v2_4_litert_fp32() -> None:
  ensure_litert_or_skip()

  model = load("geo", "2.4", "tf", precision="fp32", library="litert")
  assert isinstance(model, GeoModelV2_4)


def test_pb_type_is_correct(monkeypatch: pytest.MonkeyPatch) -> None:
  _stub_downloads(monkeypatch)

  assert type(load("geo", "2.4", "pb")) is GeoModelV2_4


def test_tf_tflite_type_is_correct(monkeypatch: pytest.MonkeyPatch) -> None:
  _stub_downloads(monkeypatch)

  assert type(load("geo", "2.4", "tf", library="tflite")) is GeoModelV2_4


@pytest.mark.litert
def test_tf_litert_type_is_correct() -> None:
  ensure_litert_or_skip()

  assert type(load("geo", "2.4", "tf", library="litert")) is GeoModelV2_4


def test_types_with_precisions_are_correct(monkeypatch: pytest.MonkeyPatch) -> None:
  _stub_downloads(monkeypatch)

  assert (
    type(load("geo", "2.4", "pb", precision=cast(Literal["fp32"], f"fp{32}")))
    is GeoModelV2_4
  )
  assert (
    type(load("geo", "2.4", "tf", precision=cast(Literal["fp32"], f"fp{32}")))
    is GeoModelV2_4
  )
