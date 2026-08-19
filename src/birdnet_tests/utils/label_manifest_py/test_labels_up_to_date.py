import json
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from birdnet.utils.label_manifest import (
  LabelInput,
  get_manifest_path,
  labels_up_to_date,
  sha256_bytes,
)

from .conftest import GENERATION_VERSION, GENERATOR, LANGUAGES


def _check(
  lang_dir: Path,
  inputs: dict[str, LabelInput],
  *,
  languages: dict[str, str] | None = None,
  generator_version: int = GENERATION_VERSION,
  generator: str = GENERATOR,
) -> bool:
  return labels_up_to_date(
    lang_dir,
    generator=generator,
    generator_version=generator_version,
    inputs=inputs,
    languages=LANGUAGES if languages is None else languages,
  )


@pytest.mark.no_tf
def test_a_freshly_generated_directory_verifies(
  lang_dir: Path, inputs: dict[str, LabelInput], generate: Callable[[], None]
) -> None:
  generate()

  assert _check(lang_dir, inputs)


@pytest.mark.no_tf
def test_a_directory_without_a_manifest_is_stale(
  lang_dir: Path, inputs: dict[str, LabelInput], generate: Callable[[], None]
) -> None:
  """Every installation predating the manifest looks like this, which is what
  makes upgrading the whole remediation."""
  generate()
  get_manifest_path(lang_dir).unlink()

  assert not _check(lang_dir, inputs)


@pytest.mark.no_tf
def test_an_input_swapped_for_same_size_different_content_is_stale(
  lang_dir: Path, inputs: dict[str, LabelInput], generate: Callable[[], None]
) -> None:
  """The failure the previous design could not see.

  Another installed version replaced the shared taxonomy at its generic path
  with a different release. Byte size and the recorded URL both still matched,
  so the label files were served as current while holding the other release's
  names.
  """
  generate()
  taxonomy = inputs["taxonomy"]
  swapped = bytearray(taxonomy.path.read_bytes())
  swapped[-10:] = b"Kohlmeis3\n"
  assert len(swapped) == taxonomy.size, "the swap must not change the byte size"
  taxonomy.path.write_bytes(bytes(swapped))

  assert not _check(lang_dir, inputs)


@pytest.mark.no_tf
def test_a_generated_file_edited_in_place_is_stale(
  lang_dir: Path, inputs: dict[str, LabelInput], generate: Callable[[], None]
) -> None:
  generate()
  target = lang_dir / "de.txt"
  edited = target.read_text(encoding="utf-8").replace("de", "DE")
  target.write_text(edited, encoding="utf-8")

  assert not _check(lang_dir, inputs)


@pytest.mark.no_tf
def test_a_bumped_generation_version_is_stale(
  lang_dir: Path, inputs: dict[str, LabelInput], generate: Callable[[], None]
) -> None:
  generate()

  assert not _check(lang_dir, inputs, generator_version=GENERATION_VERSION + 1)


@pytest.mark.no_tf
def test_a_directory_written_by_another_generator_is_stale(
  lang_dir: Path, inputs: dict[str, LabelInput], generate: Callable[[], None]
) -> None:
  generate()

  assert not _check(lang_dir, inputs, generator="a_different_generator")


@pytest.mark.no_tf
def test_an_added_language_is_stale(
  lang_dir: Path, inputs: dict[str, LabelInput], generate: Callable[[], None]
) -> None:
  generate()

  assert not _check(lang_dir, inputs, languages={**LANGUAGES, "fr": "common_name_fr"})


@pytest.mark.no_tf
def test_a_removed_language_is_stale(
  lang_dir: Path, inputs: dict[str, LabelInput], generate: Callable[[], None]
) -> None:
  """Estonian outlived the taxonomy column being dropped; it must not again."""
  generate()

  assert not _check(lang_dir, inputs, languages={"en_us": "com_name"})


@pytest.mark.no_tf
def test_a_language_remapped_to_another_column_is_stale(
  lang_dir: Path, inputs: dict[str, LabelInput], generate: Callable[[], None]
) -> None:
  """The taxonomy carries both common_name_zh and common_name_zh-CN, so which
  column a language points at is a real choice that has to be recorded."""
  generate()

  assert not _check(
    lang_dir, inputs, languages={**LANGUAGES, "de": "common_name_de_AT"}
  )


@pytest.mark.no_tf
def test_an_unexpected_extra_language_file_is_stale(
  lang_dir: Path, inputs: dict[str, LabelInput], generate: Callable[[], None]
) -> None:
  generate()
  (lang_dir / "zz.txt").write_text("Parus major_zz", encoding="utf-8")

  assert not _check(lang_dir, inputs)


@pytest.mark.no_tf
def test_a_manifest_that_omits_a_present_file_is_stale(
  lang_dir: Path, inputs: dict[str, LabelInput], generate: Callable[[], None]
) -> None:
  """A file the manifest does not list is a file whose content nothing vouches
  for, even when the language set as a whole still lines up."""
  generate()
  path = get_manifest_path(lang_dir)
  manifest = json.loads(path.read_text(encoding="utf-8"))
  del manifest["files"]["de.txt"]
  path.write_text(json.dumps(manifest), encoding="utf-8")

  assert not _check(lang_dir, inputs)


@pytest.mark.no_tf
def test_a_changed_mtime_alone_does_not_make_it_stale(
  lang_dir: Path, inputs: dict[str, LabelInput], generate: Callable[[], None]
) -> None:
  """A restored CI cache or a copy without -p rewrites timestamps; the content
  is what matters, and re-verifying it must not force a rebuild."""
  generate()
  for path in (
    *(lang_dir / f"{lang}.txt" for lang in LANGUAGES),
    inputs["labels"].path,
  ):
    os.utime(path, ns=(1, 1))

  assert _check(lang_dir, inputs)


@pytest.mark.no_tf
def test_the_stat_fast_path_is_refreshed_after_a_digest_check(
  lang_dir: Path, inputs: dict[str, LabelInput], generate: Callable[[], None]
) -> None:
  """Otherwise every later load would hash every file again."""
  generate()
  touched = lang_dir / "de.txt"
  os.utime(touched, ns=(1, 1))
  assert _check(lang_dir, inputs)

  manifest = json.loads(get_manifest_path(lang_dir).read_text(encoding="utf-8"))

  # Compared against the real stat: filesystems quantise timestamps, so the
  # value written back is whatever the file actually carries now.
  assert manifest["files"]["de.txt"]["mtime_ns"] == touched.stat().st_mtime_ns


@pytest.mark.no_tf
def test_an_input_replaced_by_a_new_release_is_stale(
  lang_dir: Path, inputs: dict[str, LabelInput], generate: Callable[[], None]
) -> None:
  """The constants moved on: the pinned digest no longer describes this file."""
  generate()
  moved_on = {
    **inputs,
    "taxonomy": replace(inputs["taxonomy"], sha256=sha256_bytes(b"a newer release")),
  }

  assert not _check(lang_dir, moved_on)
