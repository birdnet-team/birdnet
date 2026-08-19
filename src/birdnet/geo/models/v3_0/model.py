from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, final

from ordered_set import OrderedSet

from birdnet.core.backends import (
  BackendLoader,
  VersionedGeoBackendProtocol,
)
from birdnet.geo.inference.prediction_result import GeoPredictionResult
from birdnet.geo.inference.session import GeoPredictionSession
from birdnet.geo.models.base import GeoModelBase
from birdnet.globals import (
  GEO_MODEL_VERSION_V3_0,
  GEO_MODEL_VERSIONS,
  GEO_YEAR_ROUND_AGGREGATION_MAX,
  GEO_YEAR_ROUND_AGGREGATIONS,
  MODEL_TYPE_GEO,
  MODEL_TYPES,
)
from birdnet.utils.helper import (
  directory_lock,
  download_file_tqdm,
  validate_species_list,
  write_text_atomic,
)
from birdnet.utils.label_manifest import (
  LabelInput,
  get_manifest_path,
  labels_up_to_date,
  record_generation,
  verify_download,
)
from birdnet.utils.local_data import APP_DIR
from birdnet.utils.taxonomy_v3 import (
  ensure_taxonomy_v3_available,
  get_taxonomy_v3_input,
  get_taxonomy_v3_path,
  taxonomy_v3_available,
)

_LABELS_DL_URL = "https://github.com/birdnet-team/geomodel/releases/download/v3.0.4/BirdNET+_Geomodel_V3.0.4_Global_14K_Labels.txt"
_LABELS_DL_SIZE = 671823
_LABELS_DL_SHA256 = "8250b457e45d43fc3e77b5cbd06a1d311baf585ab9c51ed8d42e011d98534835"

_GENERATOR_NAME = "geo_v3_0"
# Bump whenever a change here would produce different <lang>.txt bytes from the
# same inputs - the join key, the tie-break, the fallback, the line format. The
# golden-digest test fails until this and the expected digests agree.
_GENERATION_VERSION = 1

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

_GEO_V3_0_BASE_DIR = APP_DIR / "geo-models" / "v3.0"
_LABELS_RAW_PATH = _GEO_V3_0_BASE_DIR / "labels_raw.txt"
_SETUP_LOCK_DIR = APP_DIR / ".geo_model_v3_0_setup.lock"


class GeoDownloaderBaseV3_0:
  AVAILABLE_LANGUAGES: OrderedSet[str] = OrderedSet(_LANGUAGE_TO_COLUMN.keys())

  @classmethod
  def _get_lang_dir(cls) -> Path:
    raise NotImplementedError

  @classmethod
  def _labels_input(cls) -> LabelInput:
    return LabelInput(
      path=_LABELS_RAW_PATH,
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
    with directory_lock(_SETUP_LOCK_DIR, "geo model v3.0 shared asset setup"):
      if cls._check_labels_available():
        return

      # We only get here when the labels are missing or stale. Ensure the raw
      # labels and taxonomy are present and current, then regenerate the
      # per-language files unconditionally: they are either missing or left over
      # from an older labels version (see _check_labels_available). Regeneration
      # is idempotent, so this is safe even if only one of them was outdated.
      labels_stale = not _LABELS_RAW_PATH.is_file() or (
        _LABELS_RAW_PATH.stat().st_size != _LABELS_DL_SIZE
      )
      if labels_stale:
        _LABELS_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        download_file_tqdm(
          _LABELS_DL_URL,
          _LABELS_RAW_PATH,
          download_size=_LABELS_DL_SIZE,
          description="Downloading geo model v3.0 labels",
        )
        verify_download(_LABELS_RAW_PATH, cls._labels_input())

      if not taxonomy_v3_available():
        ensure_taxonomy_v3_available()

      cls._generate_lang_files()

  @classmethod
  def _generate_lang_files(cls) -> None:
    lang_dir = cls._get_lang_dir()
    lang_dir.mkdir(parents=True, exist_ok=True)
    # Dropped first: an interrupted run then leaves a directory that visibly
    # fails verification rather than one still vouched for by the old record.
    get_manifest_path(lang_dir).unlink(missing_ok=True)

    # Read once and hash *these* bytes, so the manifest records what was really
    # used rather than what the constants say should have been there.
    labels_raw = _LABELS_RAW_PATH.read_bytes()
    taxonomy_raw = get_taxonomy_v3_path().read_bytes()

    species_order: list[tuple[str, str, str]] = []
    for line in labels_raw.decode("utf-8").splitlines():
      parts = line.split("\t")
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

    n_unresolved = 0
    written: list[Path] = []
    for lang, col in _LANGUAGE_TO_COLUMN.items():
      lang_file = lang_dir / f"{lang}.txt"
      lines: list[str] = []
      unresolved = 0
      for code, sci_name, en_us_name in species_order:
        tax_row = taxonomy.get(code, {})
        localized_name = tax_row.get(col, "").strip()
        if not localized_name:
          localized_name = en_us_name
          unresolved += 1
        lines.append(f"{sci_name}_{localized_name}")
      write_text_atomic(lang_file, "\n".join(lines), encoding="utf-8")
      written.append(lang_file)
      n_unresolved = max(n_unresolved, unresolved)

    record_generation(
      lang_dir,
      generator=_GENERATOR_NAME,
      generator_version=_GENERATION_VERSION,
      declared_inputs=cls._manifest_inputs(),
      read_bytes={"labels": labels_raw, "taxonomy": taxonomy_raw},
      languages=_LANGUAGE_TO_COLUMN,
      lang_files=written,
      stats={"n_entries": len(species_order), "n_unresolved": n_unresolved},
    )

  @classmethod
  def get_lang_file(cls, lang: str) -> Path:
    return cls._get_lang_dir() / f"{lang}.txt"


class GeoModelV3_0(GeoModelBase):
  def __init__(
    self,
    model_path: Path,
    species_list: OrderedSet[str],
    is_custom_model: bool,
    backend_type: type[VersionedGeoBackendProtocol],
    backend_kwargs: dict[str, Any],
  ) -> None:
    super().__init__(
      model_path, species_list, is_custom_model, backend_type, backend_kwargs
    )

  @classmethod
  def load(
    cls,
    model_path: Path,
    species_list: OrderedSet[str],
    backend_type: type[VersionedGeoBackendProtocol],
    backend_kwargs: dict[str, Any],
  ) -> GeoModelV3_0:
    result = GeoModelV3_0(
      model_path,
      species_list,
      is_custom_model=False,
      backend_type=backend_type,
      backend_kwargs=backend_kwargs,
    )
    return result

  @classmethod
  def load_custom(
    cls,
    model_path: Path,
    species_list: Path,
    backend_type: type[VersionedGeoBackendProtocol],
    backend_kwargs: dict[str, Any],
    check_validity: bool,
  ) -> GeoModelV3_0:
    assert model_path.exists()
    assert species_list.is_file()

    loaded_species_list = validate_species_list(species_list)

    if check_validity:
      n_species_in_model = BackendLoader.check_model_can_be_loaded(
        model_path, backend_type, backend_kwargs
      )

      if n_species_in_model != len(loaded_species_list):
        raise ValueError(
          f"Model '{model_path.absolute()}' has {n_species_in_model} outputs, "
          f"but species list '{species_list.absolute()}' "
          f"has {len(loaded_species_list)} species!"
        )

    result = GeoModelV3_0(
      model_path,
      loaded_species_list,
      is_custom_model=True,
      backend_type=backend_type,
      backend_kwargs=backend_kwargs,
    )
    return result

  @classmethod
  @final
  def get_version(cls) -> GEO_MODEL_VERSIONS:
    return GEO_MODEL_VERSION_V3_0

  @classmethod
  @final
  def get_model_type(cls) -> MODEL_TYPES:
    return MODEL_TYPE_GEO

  def predict_session(
    self,
    /,
    *,
    min_confidence: float = 0.03,
    half_precision: bool = False,
    device: str = "CPU",
  ) -> GeoPredictionSession:
    return GeoPredictionSession(
      species_list=self.species_list,
      model_path=self.model_path,
      model_is_custom=self.is_custom_model,
      model_version=self.get_version(),
      model_backend_type=self.backend_type,
      model_backend_custom_kwargs=self.backend_kwargs,
      min_confidence=min_confidence,
      half_precision=half_precision,
      device=device,
    )

  def predict(
    self,
    latitude: float,
    longitude: float,
    /,
    *,
    week: int | None = None,
    year_round_aggregation: GEO_YEAR_ROUND_AGGREGATIONS = GEO_YEAR_ROUND_AGGREGATION_MAX,  # noqa: E501
    min_confidence: float = 0.03,
    half_precision: bool = False,
    device: str = "CPU",
  ) -> GeoPredictionResult:
    with self.predict_session(
      min_confidence=min_confidence,
      half_precision=half_precision,
      device=device,
    ) as session:
      return session.run(
        latitude,
        longitude,
        week=week,
        year_round_aggregation=year_round_aggregation,
      )
