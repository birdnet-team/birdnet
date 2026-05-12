from __future__ import annotations

import csv
import os
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
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
from birdnet.utils.helper import download_file_tqdm, validate_species_list
from birdnet.utils.local_data import APP_DIR

_LABELS_DL_URL = "https://github.com/birdnet-team/geomodel/releases/download/v3.0.2/BirdNET+_Geomodel_V3.0.2_Global_12K_Labels.txt"
_LABELS_DL_SIZE = 571793
_TAXONOMY_DL_URL = (
  "https://github.com/birdnet-team/geomodel/raw/refs/tags/v3.0.2/taxonomy.csv"
)
_TAXONOMY_DL_SIZE = 9162669

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
  "et": "common_name_et",
}

_GEO_V3_0_BASE_DIR = APP_DIR / "geo-models" / "v3.0"
_LABELS_RAW_PATH = _GEO_V3_0_BASE_DIR / "labels_raw.txt"
_TAXONOMY_PATH = APP_DIR / "taxonomy_v3_0.csv"
_SETUP_LOCK_DIR = APP_DIR / ".geo_model_v3_0_setup.lock"


def _write_text_atomic(path: Path, content: str, encoding: str = "utf-8") -> None:
  fd, temp_name = tempfile.mkstemp(
    dir=path.parent,
    prefix=f"{path.name}.",
    suffix=".tmp",
  )
  os.close(fd)
  temp_path = Path(temp_name)

  try:
    temp_path.write_text(content, encoding=encoding)
    temp_path.replace(path)
  except Exception:
    temp_path.unlink(missing_ok=True)
    raise


@contextmanager
def _setup_lock(timeout_s: float = 300.0) -> Generator[None, None, None]:
  deadline = time.monotonic() + timeout_s
  while True:
    try:
      _SETUP_LOCK_DIR.mkdir(parents=True, exist_ok=False)
      break
    except FileExistsError as err:
      if time.monotonic() >= deadline:
        raise TimeoutError(
          "Timed out while waiting for geo model v3.0 shared asset setup."
        ) from err
      time.sleep(0.1)

  try:
    yield
  finally:
    _SETUP_LOCK_DIR.rmdir()


class GeoDownloaderBaseV3_0:
  AVAILABLE_LANGUAGES: OrderedSet[str] = OrderedSet(_LANGUAGE_TO_COLUMN.keys())

  @classmethod
  def _get_lang_dir(cls) -> Path:
    raise NotImplementedError

  @classmethod
  def _check_labels_available(cls) -> bool:
    if not _LABELS_RAW_PATH.is_file():
      return False
    if _LABELS_RAW_PATH.stat().st_size != _LABELS_DL_SIZE:
      return False
    if not _TAXONOMY_PATH.is_file():
      return False
    if _TAXONOMY_PATH.stat().st_size != _TAXONOMY_DL_SIZE:
      return False
    lang_dir = cls._get_lang_dir()
    if not lang_dir.is_dir():
      return False
    return all((lang_dir / f"{lang}.txt").is_file() for lang in cls.AVAILABLE_LANGUAGES)

  @classmethod
  def ensure_labels_available(cls) -> None:
    with _setup_lock():
      if cls._check_labels_available():
        return

      needs_regen = False

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
        needs_regen = True

      taxonomy_stale = (
        not _TAXONOMY_PATH.is_file()
        or _TAXONOMY_PATH.stat().st_size != _TAXONOMY_DL_SIZE
      )
      if taxonomy_stale:
        _TAXONOMY_PATH.parent.mkdir(parents=True, exist_ok=True)
        download_file_tqdm(
          _TAXONOMY_DL_URL,
          _TAXONOMY_PATH,
          download_size=_TAXONOMY_DL_SIZE,
          description="Downloading geo model v3.0 taxonomy",
        )
        needs_regen = True

      lang_files_missing = not all(
        (cls._get_lang_dir() / f"{lang}.txt").is_file()
        for lang in cls.AVAILABLE_LANGUAGES
      )
      if needs_regen or lang_files_missing:
        cls._generate_lang_files()

  @classmethod
  def _generate_lang_files(cls) -> None:
    species_order: list[tuple[str, str, str]] = []
    with open(_LABELS_RAW_PATH, encoding="utf-8") as f:
      for line in f:
        parts = line.rstrip("\n").split("\t")
        species_order.append((parts[0], parts[1], parts[2]))

    taxonomy: dict[str, dict[str, str]] = {}
    with open(_TAXONOMY_PATH, encoding="utf-8", newline="") as f:
      reader = csv.DictReader(f)
      for row in reader:
        code = row.get("species_code", "").strip()
        if code:
          taxonomy[code] = dict(row)

    lang_dir = cls._get_lang_dir()
    lang_dir.mkdir(parents=True, exist_ok=True)
    for lang, col in _LANGUAGE_TO_COLUMN.items():
      lang_file = lang_dir / f"{lang}.txt"
      lines: list[str] = []
      for code, sci_name, en_us_name in species_order:
        tax_row = taxonomy.get(code, {})
        localized_name = tax_row.get(col, "").strip()
        if not localized_name:
          localized_name = en_us_name
        lines.append(f"{sci_name}_{localized_name}")
      _write_text_atomic(lang_file, "\n".join(lines), encoding="utf-8")

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
