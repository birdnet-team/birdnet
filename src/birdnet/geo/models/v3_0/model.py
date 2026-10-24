from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import ClassVar, final

from ordered_set import OrderedSet

from birdnet.geo.models.base import GeoModelBase
from birdnet.globals import (
  GEO_MODEL_VERSION_V3_0,
  GEO_MODEL_VERSIONS,
  MODEL_BACKENDS,
  MODEL_PRECISIONS,
  MODEL_TYPE_GEO,
  MODEL_TYPES,
)
from birdnet.utils.helper import (
  ModelInfo,
  directory_lock,
  ensure_single_file_model,
  get_species_from_file,
  write_text_atomic,
)
from birdnet.utils.label_manifest import (
  LabelInput,
  ensure_artifact,
  get_manifest_path,
  labels_up_to_date,
  record_generation,
)
from birdnet.utils.local_data import (
  get_birdnet_app_data_folder,
  get_lang_dir,
  get_model_path,
)
from birdnet.utils.taxonomy_v3 import (
  ensure_taxonomy_v3_available,
  get_taxonomy_v3_input,
)

_LABELS_DL_URL = "https://github.com/birdnet-team/geomodel/releases/download/v3.0.4/BirdNET+_Geomodel_V3.0.4_Global_14K_Labels.txt"
_LABELS_DL_SIZE = 671823
_LABELS_DL_SHA256 = "8250b457e45d43fc3e77b5cbd06a1d311baf585ab9c51ed8d42e011d98534835"

_GENERATOR_NAME = "geo_v3_0"
# Bump whenever a change here would produce different <lang>.txt bytes from the
# same inputs - the join key, the tie-break, the fallback, the line format. The
# golden-digest test fails until this and the expected digests agree.
_GENERATION_VERSION = 2

# No Estonian ("et"): the v0.2-Jun2026 taxonomy has no common_name_et column,
# so every Estonian name would silently be the English one.
_LANGUAGE_TO_COLUMN: dict[str, str] = {
  "en_us": "com_name",
  "de": "common_name_de",
  "es": "common_name_es",
  "pl": "common_name_pl",
  "fr": "common_name_fr",
  "nl": "common_name_nl",
  "ru": "common_name_ru",
  "ja": "common_name_ja",
  "cs": "common_name_cs",
  "ca": "common_name_ca",
  "pt": "common_name_pt",
  "no": "common_name_no",
  "bg": "common_name_bg",
  "sv": "common_name_sv",
  "da": "common_name_da",
  "tr": "common_name_tr",
  "sk": "common_name_sk",
  "sr": "common_name_sr",
  "uk": "common_name_uk",
  "zh": "common_name_zh-CN",
  "fi": "common_name_fi",
  "es_es": "common_name_es_ES",
  "es_mx": "common_name_es_MX",
  "es_ec": "common_name_es_EC",
  "pt_pt": "common_name_pt_PT",
  "hr": "common_name_hr",
  "lt": "common_name_lt",
  "fa": "common_name_fa",
  "cy": "common_name_cy",
}


def _geo_v3_0_base_dir() -> Path:
  return get_birdnet_app_data_folder() / "geo-models" / "v3.0"


# Named after the content, so an install of another version cannot overwrite
# this release's copy with its own and leave both re-downloading forever.
def _labels_raw_path() -> Path:
  return _geo_v3_0_base_dir() / f"labels_raw-{_LABELS_DL_SHA256[:12]}.txt"


def _legacy_labels_raw_path() -> Path:
  return _geo_v3_0_base_dir() / "labels_raw.txt"


def _setup_lock_dir() -> Path:
  return get_birdnet_app_data_folder() / ".geo_model_v3_0_setup.lock"


class GeoDownloaderBaseV3_0:
  AVAILABLE_LANGUAGES: OrderedSet[str] = OrderedSet(_LANGUAGE_TO_COLUMN.keys())

  @classmethod
  def _get_lang_dir(cls) -> Path:
    raise NotImplementedError

  @classmethod
  def _labels_input(cls) -> LabelInput:
    return LabelInput(
      path=_labels_raw_path(),
      url=_LABELS_DL_URL,
      size=_LABELS_DL_SIZE,
      sha256=_LABELS_DL_SHA256,
    )

  @classmethod
  def _manifest_inputs(cls) -> dict[str, LabelInput]:
    return {"labels": cls._labels_input(), "taxonomy": get_taxonomy_v3_input()}

  @classmethod
  def _check_labels_available(cls) -> bool:
    return labels_up_to_date(
      cls._get_lang_dir(),
      generator=_GENERATOR_NAME,
      generator_version=_GENERATION_VERSION,
      inputs=cls._manifest_inputs(),
      languages=_LANGUAGE_TO_COLUMN,
    )

  @classmethod
  def ensure_labels_available(cls) -> None:
    # Checked before the lock: a current cache then needs no write at all, which
    # keeps read-only and pre-populated app data directories usable.
    if cls._check_labels_available():
      return

    with directory_lock(_setup_lock_dir(), "geo model v3.0 shared asset setup"):
      if cls._check_labels_available():
        return

      ensure_artifact(
        cls._labels_input(),
        "Downloading geo model v3.0 labels",
        legacy_path=_legacy_labels_raw_path(),
      )
      ensure_taxonomy_v3_available()

      cls._generate_lang_files()

      # Both inputs were just verified by digest, so this can only fail if
      # generation itself is wrong. Raising beats silently regenerating on every
      # later load while serving names nobody checked.
      if not cls._check_labels_available():
        raise RuntimeError(
          "The geo model v3.0 label files could not be generated from verified "
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

    species_order: list[tuple[str, str, str]] = []
    # newline=None reproduces exactly what iterating the file in text mode did.
    # str.splitlines() would additionally break on U+2028 and friends, silently
    # changing the species count if one ever appeared in a label.
    with io.StringIO(labels_raw.decode("utf-8"), newline=None) as f:
      for line in f:
        parts = line.rstrip("\n").split("\t")
        species_order.append((parts[0], parts[1], parts[2]))

    # The labels join to the taxonomy on the species code, which is the stable
    # key - scientific names change between taxonomy versions while codes do not.
    # (The acoustic model has to join on sci_name; see the module docstring of
    # birdnet/utils/taxonomy_v3.py for why the two differ.)
    # Upstream does occasionally reuse one code for two species though (in
    # v0.2-Jun2026: y01249 and grsbop1), and then the row that happens to come
    # last would hand this species the other one's localized names. Prefer the row
    # whose scientific name is the one the labels use for that code; with no such
    # row the last one still wins.
    label_sci_name_by_code = {code: sci_name for code, sci_name, _ in species_order}
    taxonomy: dict[str, dict[str, str]] = {}
    with io.StringIO(taxonomy_raw.decode("utf-8"), newline="") as f:
      reader = csv.DictReader(f)
      for row in reader:
        code = row.get("species_code", "").strip()
        if not code:
          continue
        kept = taxonomy.get(code)
        if kept is not None and kept.get("sci_name", "").strip() == (
          label_sci_name_by_code.get(code)
        ):
          continue
        taxonomy[code] = dict(row)

    written: list[Path] = []
    for lang, col in _LANGUAGE_TO_COLUMN.items():
      lang_file = lang_dir / f"{lang}.txt"
      lines: list[str] = []
      for code, sci_name, en_us_name in species_order:
        tax_row = taxonomy.get(code, {})
        localized_name = tax_row.get(col, "").strip()
        if not localized_name:
          localized_name = en_us_name
        lines.append(f"{sci_name}_{localized_name}")
      write_text_atomic(lang_file, "\n".join(lines), encoding="utf-8")
      written.append(lang_file)

    record_generation(
      lang_dir,
      generator=_GENERATOR_NAME,
      generator_version=_GENERATION_VERSION,
      declared_inputs=inputs,
      read_bytes={"labels": labels_raw, "taxonomy": taxonomy_raw},
      languages=_LANGUAGE_TO_COLUMN,
      lang_files=written,
      stats={"n_entries": len(species_order)},
    )

  @classmethod
  def get_lang_file(cls, lang: str) -> Path:
    return cls._get_lang_dir() / f"{lang}.txt"


class _GeoSingleFileDownloaderV3_0(GeoDownloaderBaseV3_0):
  """Shared logic of the single-file geo V3.0 downloads (tf, onnx, pt).

  A subclass names its backend and its precision table; the paths, the
  availability check and the download follow from those two.
  """

  _backend: ClassVar[MODEL_BACKENDS]
  _models: ClassVar[dict[MODEL_PRECISIONS, ModelInfo]]

  @classmethod
  def _get_lang_dir(cls) -> Path:
    return get_lang_dir("geo", "3.0", cls._backend)

  @classmethod
  def _get_model_path(cls, precision: MODEL_PRECISIONS) -> Path:
    return get_model_path(
      "geo",
      "3.0",
      cls._backend,
      precision,
      content_tag=cls._models[precision].content_tag,
    )

  @classmethod
  def _check_geo_model_available(cls, precision: MODEL_PRECISIONS) -> bool:
    model_path = cls._get_model_path(precision)
    if not model_path.is_file():
      return False
    if model_path.stat().st_size != cls._models[precision].file_size:
      return False
    return cls._check_labels_available()

  @classmethod
  def _download_model(cls, precision: MODEL_PRECISIONS) -> None:
    ensure_single_file_model(
      cls._models[precision],
      cls._get_model_path(precision),
      legacy_path=get_model_path("geo", "3.0", cls._backend, precision),
      description=f"Downloading geo model v3.0 ({cls._backend}, {precision})",
    )

  @classmethod
  def get_model_path_and_labels(
    cls, lang: str, precision: MODEL_PRECISIONS
  ) -> tuple[Path, OrderedSet[str]]:
    if precision not in cls._models:
      raise ValueError(
        f"Unsupported model precision for geo {cls._backend} model: {precision}. "
        f"Currently supported precisions are: {', '.join(cls._models)}."
      )
    assert lang in cls.AVAILABLE_LANGUAGES

    cls.ensure_labels_available()
    if not cls._check_geo_model_available(precision):
      cls._download_model(precision)
    assert cls._check_geo_model_available(precision)

    lang_file = cls.get_lang_file(lang)
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return cls._get_model_path(precision), labels


class GeoModelV3_0(GeoModelBase):
  @classmethod
  @final
  def get_version(cls) -> GEO_MODEL_VERSIONS:
    return GEO_MODEL_VERSION_V3_0

  @classmethod
  @final
  def get_model_type(cls) -> MODEL_TYPES:
    return MODEL_TYPE_GEO
