import json
from pathlib import Path

import pytest

from birdnet.utils.label_manifest import (
  LEGACY_MARKER_NAME,
  LabelInput,
  get_manifest_path,
  labels_up_to_date,
  record_generation,
  sha256_bytes,
)

from .conftest import GENERATION_VERSION, GENERATOR, LANGUAGES


def _generate_via_record(
  lang_dir: Path, inputs: dict[str, LabelInput], read_bytes: dict[str, bytes]
) -> None:
  """Generate the way the models do: through `record_generation`."""
  written = []
  for lang in LANGUAGES:
    path = lang_dir / f"{lang}.txt"
    path.write_text(f"Parus major_{lang}", encoding="utf-8")
    written.append(path)
  record_generation(
    lang_dir,
    generator=GENERATOR,
    generator_version=GENERATION_VERSION,
    declared_inputs=inputs,
    read_bytes=read_bytes,
    languages=LANGUAGES,
    lang_files=written,
  )


@pytest.mark.no_tf
def test_records_the_bytes_that_were_read_not_the_pinned_constant(
  lang_dir: Path, inputs: dict[str, LabelInput]
) -> None:
  """The whole point of the manifest.

  If the file on a shared path was a different release than the constants
  declare, recording the declared digest would vouch for content nobody read -
  which is the failure this exists to catch.
  """
  actually_read = {
    "labels": inputs["labels"].path.read_bytes(),
    "taxonomy": b"a different release, same declared constants",
  }

  _generate_via_record(lang_dir, inputs, actually_read)

  manifest = json.loads(get_manifest_path(lang_dir).read_text(encoding="utf-8"))
  assert manifest["inputs"]["taxonomy"]["sha256"] == sha256_bytes(
    actually_read["taxonomy"]
  )
  assert manifest["inputs"]["taxonomy"]["sha256"] != inputs["taxonomy"].sha256
  # and it must therefore not verify against the declared constants
  assert not labels_up_to_date(
    lang_dir,
    generator=GENERATOR,
    generator_version=GENERATION_VERSION,
    inputs=inputs,
    languages=LANGUAGES,
  )


@pytest.mark.no_tf
def test_a_directory_generated_from_the_declared_inputs_verifies(
  lang_dir: Path, inputs: dict[str, LabelInput]
) -> None:
  read = {name: inp.path.read_bytes() for name, inp in inputs.items()}

  _generate_via_record(lang_dir, inputs, read)

  assert labels_up_to_date(
    lang_dir,
    generator=GENERATOR,
    generator_version=GENERATION_VERSION,
    inputs=inputs,
    languages=LANGUAGES,
  )


@pytest.mark.no_tf
def test_drops_a_language_this_version_no_longer_produces(
  lang_dir: Path, inputs: dict[str, LabelInput]
) -> None:
  """Estonian survived the taxonomy losing its column; nothing may again."""
  (lang_dir / "et.txt").write_text("Parus major_et", encoding="utf-8")
  read = {name: inp.path.read_bytes() for name, inp in inputs.items()}

  _generate_via_record(lang_dir, inputs, read)

  assert not (lang_dir / "et.txt").exists()


@pytest.mark.no_tf
def test_removes_the_marker_left_by_earlier_releases(
  lang_dir: Path, inputs: dict[str, LabelInput]
) -> None:
  (lang_dir / LEGACY_MARKER_NAME).write_text("https://example.org/old", "utf-8")
  read = {name: inp.path.read_bytes() for name, inp in inputs.items()}

  _generate_via_record(lang_dir, inputs, read)

  assert not (lang_dir / LEGACY_MARKER_NAME).exists()
