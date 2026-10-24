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
  ensure_single_file_model,
  get_species_from_file,
)
from birdnet.utils.label_manifest import LabelInput
from birdnet.utils.local_data import (
  get_birdnet_app_data_folder,
  get_lang_dir,
  get_model_path,
)
from birdnet.utils.taxonomy_v3 import (
  ensure_taxonomy_v3_available,
  get_taxonomy_v3_input,
)
from birdnet.utils.v3_labels import V3LabelDownloaderBase

_LABELS_DL_URL = "https://github.com/birdnet-team/geomodel/releases/download/v3.0.4/BirdNET+_Geomodel_V3.0.4_Global_14K_Labels.txt"
_LABELS_DL_SIZE = 671823
_LABELS_DL_SHA256 = "8250b457e45d43fc3e77b5cbd06a1d311baf585ab9c51ed8d42e011d98534835"

_GENERATOR_NAME = "geo_v3_0"
# Bump whenever a change would produce different <lang>.txt bytes from the same
# inputs - the join key or tie-break here, or the fallback or line format in
# birdnet/utils/v3_labels.py, which is shared and so changes both models. The
# golden-digest test fails when the bytes generated from its fixture move;
# nothing forces this bump when its digests are updated.
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


class GeoDownloaderBaseV3_0(V3LabelDownloaderBase):
  AVAILABLE_LANGUAGES: OrderedSet[str] = OrderedSet(_LANGUAGE_TO_COLUMN.keys())
  _generator_name = _GENERATOR_NAME
  _generation_version = _GENERATION_VERSION

  @classmethod
  def _labels_input(cls) -> LabelInput:
    return LabelInput(
      path=_labels_raw_path(),
      url=_LABELS_DL_URL,
      size=_LABELS_DL_SIZE,
      sha256=_LABELS_DL_SHA256,
    )

  @classmethod
  def _taxonomy_input(cls) -> LabelInput:
    return get_taxonomy_v3_input()

  @classmethod
  def _ensure_taxonomy(cls) -> None:
    ensure_taxonomy_v3_available()

  @classmethod
  def _languages(cls) -> dict[str, str]:
    return _LANGUAGE_TO_COLUMN

  @classmethod
  def _labels_lock_dir(cls) -> Path:
    return _setup_lock_dir()

  @classmethod
  def _legacy_labels_path(cls) -> Path:
    return _legacy_labels_raw_path()

  @classmethod
  def _display_name(cls) -> str:
    return "geo model v3.0"

  @classmethod
  def _parse_labels(cls, labels_raw: bytes) -> list[tuple[str, str, str]]:
    species_order: list[tuple[str, str, str]] = []
    # newline=None reproduces exactly what iterating the file in text mode did.
    # str.splitlines() would additionally break on U+2028 and friends, silently
    # changing the species count if one ever appeared in a label.
    with io.StringIO(labels_raw.decode("utf-8"), newline=None) as f:
      for line in f:
        parts = line.rstrip("\n").split("\t")
        species_order.append((parts[0], parts[1], parts[2]))
    return species_order

  @classmethod
  def _index_taxonomy(
    cls, taxonomy_raw: bytes, species: list[tuple[str, str, str]]
  ) -> dict[str, dict[str, str]]:
    # The labels join to the taxonomy on the species code, which is the stable
    # key - scientific names change between taxonomy versions while codes do not.
    # (The acoustic model has to join on sci_name; see the module docstring of
    # birdnet/utils/taxonomy_v3.py for why the two differ.)
    # Upstream does occasionally reuse one code for two species though (in
    # v0.2-Jun2026: y01249 and grsbop1), and then the row that happens to come
    # last would hand this species the other one's localized names. Prefer the row
    # whose scientific name is the one the labels use for that code; with no such
    # row the last one still wins.
    label_sci_name_by_code = {code: sci_name for code, sci_name, _ in species}
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
    return taxonomy


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
