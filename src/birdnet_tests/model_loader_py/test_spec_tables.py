"""The loader's spec tables serve every precision they do not reject by name."""

import pytest

from birdnet.globals import (
  VALID_ACOUSTIC_MODEL_VERSIONS,
  VALID_GEO_MODEL_VERSIONS,
  VALID_MODEL_PRECISIONS,
)
from birdnet.model_loader import (
  _CUSTOM_MODELS,
  _CUSTOM_TFLITE_BACKENDS,
  _LANGUAGE_VALIDATORS,
  _OFFICIAL_MODELS,
  _CustomSpec,
  _OfficialSpec,
)

pytestmark = pytest.mark.no_tf

_SPECS: dict[str, _OfficialSpec | _CustomSpec] = {
  f"{table}-{'-'.join(key)}": spec
  for table, specs in (("official", _OFFICIAL_MODELS), ("custom", _CUSTOM_MODELS))
  for key, spec in specs.items()
}


@pytest.mark.parametrize("name", list(_SPECS))
def test_backends_are_keyed_by_valid_precisions(name: str) -> None:
  assert set(_SPECS[name].backends) <= set(VALID_MODEL_PRECISIONS)


@pytest.mark.parametrize(
  "name",
  [
    name
    for name, spec in _SPECS.items()
    if spec.precision_subject is None and getattr(spec, "resolve", None) is None
  ],
)
def test_an_ungated_spec_serves_every_precision(name: str) -> None:
  # Without a precision gate a missing precision fails as a bare KeyError.
  assert set(_SPECS[name].backends) == set(VALID_MODEL_PRECISIONS)


@pytest.mark.parametrize(
  ("model_type", "version"),
  [("acoustic", v) for v in VALID_ACOUSTIC_MODEL_VERSIONS]
  + [("geo", v) for v in VALID_GEO_MODEL_VERSIONS],
)
def test_every_valid_version_is_served(model_type: str, version: str) -> None:
  # Otherwise `load` fails with a KeyError or lists no supported backends.
  assert version in _LANGUAGE_VALIDATORS
  for table in (_OFFICIAL_MODELS, _CUSTOM_MODELS):
    assert any(key[:2] == (model_type, version) for key in table)


@pytest.mark.parametrize("version", VALID_ACOUSTIC_MODEL_VERSIONS)
def test_every_acoustic_version_has_custom_tflite_backends(version: str) -> None:
  # The custom 'tf' precision error lists the precisions found here.
  assert any(key[0] == version for key in _CUSTOM_TFLITE_BACKENDS)
