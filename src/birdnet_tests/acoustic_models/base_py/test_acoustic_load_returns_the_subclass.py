from pathlib import Path
from typing import Any

import pytest
from ordered_set import OrderedSet

from birdnet.acoustic.models.base import AcousticModelBase
from birdnet.acoustic.models.perch_v2.model import AcousticModelPerchV2
from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.core.backends import BackendLoader
from birdnet_tests.fake_acoustic_backend import FakeAcousticBackend

pytestmark = pytest.mark.no_tf

MODEL_CLASSES = [AcousticModelV2_4, AcousticModelPerchV2, AcousticModelV3_0]
SPECIES = [
  "Turdus merula_Common Blackbird",
  "Parus major_Great Tit",
  "Pica pica_Magpie",
]


def _species_file(tmp_path: Path) -> Path:
  species_file = tmp_path / "species.txt"
  species_file.write_text("\n".join(SPECIES), encoding="utf-8")
  return species_file


def _model_file(tmp_path: Path) -> Path:
  model_path = tmp_path / "model.tflite"
  model_path.write_bytes(b"not a model")
  return model_path


def _check_returning(
  monkeypatch: pytest.MonkeyPatch, n_outputs: int
) -> list[tuple[object, ...]]:
  calls: list[tuple[object, ...]] = []

  def check_model_can_be_loaded(*args: object) -> int:
    calls.append(args)
    return n_outputs

  monkeypatch.setattr(
    BackendLoader, "check_model_can_be_loaded", check_model_can_be_loaded
  )
  return calls


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=lambda c: c.__name__)
def test_load_returns_the_class_it_is_called_on(
  tmp_path: Path, model_class: type[AcousticModelBase]
) -> None:
  model_path = tmp_path / "model.tflite"
  species_list = OrderedSet(SPECIES)
  backend_kwargs: dict[str, Any] = {"option": "value"}

  model = model_class.load(
    model_path, species_list, FakeAcousticBackend, backend_kwargs
  )

  assert type(model) is model_class
  assert model.is_custom_model is False
  assert model.species_list is species_list
  assert model.model_path is model_path
  assert model.backend_type is FakeAcousticBackend
  assert model.backend_kwargs is backend_kwargs


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=lambda c: c.__name__)
def test_load_custom_returns_the_class_it_is_called_on(
  tmp_path: Path, model_class: type[AcousticModelBase]
) -> None:
  model_path = _model_file(tmp_path)
  backend_kwargs: dict[str, Any] = {"option": "value"}

  model = model_class.load_custom(
    model_path,
    _species_file(tmp_path),
    FakeAcousticBackend,
    backend_kwargs,
    check_validity=False,
  )

  assert type(model) is model_class
  assert model.is_custom_model is True
  assert list(model.species_list) == SPECIES
  assert model.model_path is model_path
  assert model.backend_type is FakeAcousticBackend
  assert model.backend_kwargs is backend_kwargs


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=lambda c: c.__name__)
def test_load_custom_reads_a_raven_label_file(
  tmp_path: Path, model_class: type[AcousticModelBase]
) -> None:
  labels_file = tmp_path / "label_names.csv"
  labels_file.write_text(
    "\n".join(f"{i},{name}" for i, name in enumerate(SPECIES)), encoding="utf-8"
  )

  model = model_class.load_custom(
    _model_file(tmp_path), labels_file, FakeAcousticBackend, {}, check_validity=False
  )

  assert list(model.species_list) == SPECIES


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=lambda c: c.__name__)
def test_load_custom_accepts_a_model_with_one_output_per_species(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  model_class: type[AcousticModelBase],
) -> None:
  model_path = _model_file(tmp_path)
  backend_kwargs: dict[str, Any] = {"option": "value"}
  calls = _check_returning(monkeypatch, len(SPECIES))

  model = model_class.load_custom(
    model_path,
    _species_file(tmp_path),
    FakeAcousticBackend,
    backend_kwargs,
    check_validity=True,
  )

  assert type(model) is model_class
  assert list(model.species_list) == SPECIES
  assert len(calls) == 1
  assert calls[0][0] is model_path
  assert calls[0][1] is FakeAcousticBackend
  assert calls[0][2] is backend_kwargs


@pytest.mark.parametrize("n_outputs", [len(SPECIES) - 1, len(SPECIES) + 1])
@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=lambda c: c.__name__)
def test_load_custom_rejects_a_species_count_the_model_does_not_have(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  model_class: type[AcousticModelBase],
  n_outputs: int,
) -> None:
  model_path = _model_file(tmp_path)
  species_file = _species_file(tmp_path)
  backend_kwargs: dict[str, Any] = {"option": "value"}
  calls = _check_returning(monkeypatch, n_outputs)

  with pytest.raises(ValueError) as exc_info:
    model_class.load_custom(
      model_path,
      species_file,
      FakeAcousticBackend,
      backend_kwargs,
      check_validity=True,
    )

  assert str(exc_info.value) == (
    f"Model '{model_path.absolute()}' has {n_outputs} outputs, but "
    f"species list '{species_file.absolute()}' has {len(SPECIES)} species!"
  )
  assert len(calls) == 1
  assert calls[0][0] is model_path
  assert calls[0][1] is FakeAcousticBackend
  assert calls[0][2] is backend_kwargs
