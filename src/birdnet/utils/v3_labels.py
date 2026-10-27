"""The localized label generation shared by the acoustic and geo V3.0 models.

Both build one ``<lang>.txt`` per language from the model's own label file and
the shared taxonomy, and record the result in a manifest. A domain supplies its
inputs, its setup lock, how it parses its label file and how it indexes the
taxonomy. The two join on different keys, deliberately; see the module docstring
of ``birdnet/utils/taxonomy_v3.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from birdnet.utils.filesystem import directory_lock, write_text_atomic
from birdnet.utils.label_manifest import (
  LabelInput,
  ensure_artifact,
  get_manifest_path,
  labels_up_to_date,
  record_generation,
)


class V3LabelDownloaderBase:
  _generator_name: ClassVar[str]
  _generation_version: ClassVar[int]

  @classmethod
  def _get_lang_dir(cls) -> Path:
    raise NotImplementedError

  @classmethod
  def _labels_input(cls) -> LabelInput:
    raise NotImplementedError

  @classmethod
  def _taxonomy_input(cls) -> LabelInput:
    raise NotImplementedError

  @classmethod
  def _ensure_taxonomy(cls) -> None:
    raise NotImplementedError

  @classmethod
  def _languages(cls) -> dict[str, str]:
    raise NotImplementedError

  @classmethod
  def _labels_lock_dir(cls) -> Path:
    raise NotImplementedError

  @classmethod
  def _legacy_labels_path(cls) -> Path | None:
    raise NotImplementedError

  @classmethod
  def _display_name(cls) -> str:
    raise NotImplementedError

  @classmethod
  def _parse_labels(cls, labels_raw: bytes) -> list[tuple[str, str, str]]:
    """``(join_key, sci_name, en_us_name)`` per species, in label-file order."""
    raise NotImplementedError

  @classmethod
  def _index_taxonomy(
    cls, taxonomy_raw: bytes, species: list[tuple[str, str, str]]
  ) -> dict[str, dict[str, str]]:
    """Taxonomy rows by the join key ``_parse_labels`` puts first."""
    raise NotImplementedError

  @classmethod
  def _manifest_inputs(cls) -> dict[str, LabelInput]:
    return {"labels": cls._labels_input(), "taxonomy": cls._taxonomy_input()}

  @classmethod
  def _check_labels_available(cls) -> bool:
    return labels_up_to_date(
      cls._get_lang_dir(),
      generator=cls._generator_name,
      generator_version=cls._generation_version,
      inputs=cls._manifest_inputs(),
      languages=cls._languages(),
    )

  @classmethod
  def ensure_labels_available(cls) -> None:
    # Checked before the lock: a current cache then needs no write at all, which
    # keeps read-only and pre-populated app data directories usable.
    if cls._check_labels_available():
      return

    name = cls._display_name()
    with directory_lock(cls._labels_lock_dir(), f"{name} label setup"):
      if cls._check_labels_available():
        return

      ensure_artifact(
        cls._labels_input(),
        f"Downloading {name} labels",
        legacy_path=cls._legacy_labels_path(),
      )
      cls._ensure_taxonomy()

      cls._generate_lang_files()

      # Both inputs were just verified by digest, so this can only fail if
      # generation itself is wrong. Raising beats silently regenerating on every
      # later load while serving names nobody checked.
      if not cls._check_labels_available():
        raise RuntimeError(
          f"The {name} label files could not be generated from verified "
          f"inputs. Remove {cls._get_lang_dir()} and try again; if this "
          "persists it is a bug in birdnet."
        )

  @classmethod
  def _generate_lang_files(cls) -> None:
    lang_dir = cls._get_lang_dir()
    lang_dir.mkdir(parents=True, exist_ok=True)
    # Dropped first: an interrupted run then leaves a directory that visibly
    # fails verification rather than one still vouched for by the old record.
    get_manifest_path(lang_dir).unlink(missing_ok=True)

    # Read once and hash *these* bytes, so the manifest records what was really
    # used rather than what the constants say should have been there.
    inputs = cls._manifest_inputs()
    # Read via the same LabelInput the manifest records, so the digest can
    # never describe a file other than the one that was parsed.
    taxonomy_raw = inputs["taxonomy"].path.read_bytes()
    labels_raw = inputs["labels"].path.read_bytes()

    species = cls._parse_labels(labels_raw)
    taxonomy = cls._index_taxonomy(taxonomy_raw, species)

    # Shared by both models: changing the fallback or the line format means
    # bumping both models' _GENERATION_VERSION.
    languages = cls._languages()
    written: list[Path] = []
    for lang, col in languages.items():
      lang_file = lang_dir / f"{lang}.txt"
      lines: list[str] = []
      for key, sci_name, en_us_name in species:
        localized_name = taxonomy.get(key, {}).get(col, "").strip() or en_us_name
        lines.append(f"{sci_name}_{localized_name}")
      write_text_atomic(lang_file, "\n".join(lines), encoding="utf-8")
      written.append(lang_file)

    record_generation(
      lang_dir,
      generator=cls._generator_name,
      generator_version=cls._generation_version,
      declared_inputs=inputs,
      read_bytes={"labels": labels_raw, "taxonomy": taxonomy_raw},
      languages=languages,
      lang_files=written,
      stats={"n_entries": len(species)},
    )

  @classmethod
  def get_lang_file(cls, lang: str) -> Path:
    return cls._get_lang_dir() / f"{lang}.txt"
