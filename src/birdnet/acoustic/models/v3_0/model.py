from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Callable, Collection, Iterable
from pathlib import Path
from typing import Any, Literal, final

import numpy.typing as npt
from ordered_set import OrderedSet

from birdnet.acoustic.inference.configs import InferenceConfig
from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticEncodingResultBase,
)
from birdnet.acoustic.inference.core.perf_tracker import AcousticProgressStats
from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticPredictionResultBase,
)
from birdnet.acoustic.inference.session import (
  AcousticEncodingSession,
  AcousticPredictionSession,
)
from birdnet.acoustic.models.base import AcousticModelBase
from birdnet.core.backends import BackendLoader, VersionedAcousticBackendProtocol
from birdnet.globals import (
  ACOUSTIC_MODEL_VERSION_V3_0,
  ACOUSTIC_MODEL_VERSIONS,
)
from birdnet.utils.helper import download_file_tqdm, validate_species_list
from birdnet.utils.local_data import APP_DIR
from birdnet.utils.taxonomy_v3 import (
  ensure_taxonomy_v3_available,
  get_taxonomy_v3_path,
  taxonomy_v3_available,
)

_LABELS_DL_URL = "https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_Labels.csv"
_LABELS_DL_SIZE = 809172
_DEFAULT_SEGMENT_SIZE_S = 3.0
_DEFAULT_SEGMENT_SIZE_SAMPLES = 96_000

_LANGUAGE_TO_COLUMN: dict[str, str] = {
  "bg": "common_name_bg",
  "ca": "common_name_ca",
  "cs": "common_name_cs",
  "cy": "common_name_cy",
  "da": "common_name_da",
  "de": "common_name_de",
  "en_us": "com_name",
  "es": "common_name_es",
  "es_ec": "common_name_es_EC",
  "es_es": "common_name_es_ES",
  "es_mx": "common_name_es_MX",
  "et": "common_name_et",
  "fa": "common_name_fa",
  "fi": "common_name_fi",
  "fr": "common_name_fr",
  "hr": "common_name_hr",
  "ja": "common_name_ja",
  "lt": "common_name_lt",
  "nl": "common_name_nl",
  "no": "common_name_no",
  "pl": "common_name_pl",
  "pt": "common_name_pt",
  "pt_pt": "common_name_pt_PT",
  "ru": "common_name_ru",
  "sk": "common_name_sk",
  "sr": "common_name_sr",
  "sv": "common_name_sv",
  "tr": "common_name_tr",
  "uk": "common_name_uk",
  "zh": "common_name_zh-CN",
}

_ACOUSTIC_V3_0_BASE_DIR = APP_DIR / "acoustic-models" / "v3.0"
_LABELS_RAW_PATH = _ACOUSTIC_V3_0_BASE_DIR / "labels_raw.csv"


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


class AcousticDownloaderBaseV3_0:
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
    if not taxonomy_v3_available():
      return False

    lang_dir = cls._get_lang_dir()
    if not lang_dir.is_dir():
      return False
    return all((lang_dir / f"{lang}.txt").is_file() for lang in cls.AVAILABLE_LANGUAGES)

  @classmethod
  def ensure_labels_available(cls) -> None:
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
        description="Downloading acoustic model v3.0 labels",
      )
      needs_regen = True

    taxonomy_stale = not taxonomy_v3_available()
    if taxonomy_stale:
      ensure_taxonomy_v3_available()
      needs_regen = True

    lang_files_missing = not all(
      (cls._get_lang_dir() / f"{lang}.txt").is_file()
      for lang in cls.AVAILABLE_LANGUAGES
    )
    if needs_regen or lang_files_missing:
      cls._generate_lang_files()

  @classmethod
  def _generate_lang_files(cls) -> None:
    species_order: list[tuple[str, str]] = []
    with open(_LABELS_RAW_PATH, encoding="utf-8", newline="") as f:
      reader = csv.DictReader(f, delimiter=";")
      for row in reader:
        sci_name = row.get("sci_name", "").strip()
        en_us_name = row.get("com_name", "").strip()
        if sci_name and en_us_name:
          species_order.append((sci_name, en_us_name))

    taxonomy: dict[str, dict[str, str]] = {}
    with open(get_taxonomy_v3_path(), encoding="utf-8", newline="") as f:
      reader = csv.DictReader(f)
      for row in reader:
        sci_name = row.get("sci_name", "").strip()
        if sci_name:
          taxonomy[sci_name] = dict(row)

    lang_dir = cls._get_lang_dir()
    lang_dir.mkdir(parents=True, exist_ok=True)
    for lang, col in _LANGUAGE_TO_COLUMN.items():
      lang_file = lang_dir / f"{lang}.txt"
      lines: list[str] = []
      for sci_name, en_us_name in species_order:
        tax_row = taxonomy.get(sci_name, {})
        localized_name = tax_row.get(col, "").strip()
        if not localized_name:
          localized_name = en_us_name
        lines.append(f"{sci_name}_{localized_name}")
      _write_text_atomic(lang_file, "\n".join(lines), encoding="utf-8")

  @classmethod
  def get_lang_file(cls, lang: str) -> Path:
    return cls._get_lang_dir() / f"{lang}.txt"


class AcousticModelV3_0(AcousticModelBase):
  def __init__(
    self,
    model_path: Path,
    species_list: OrderedSet[str],
    is_custom_model: bool,
    backend_type: type[VersionedAcousticBackendProtocol],
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
    backend_type: type[VersionedAcousticBackendProtocol],
    backend_kwargs: dict[str, Any],
  ) -> AcousticModelV3_0:
    return AcousticModelV3_0(
      model_path,
      species_list,
      is_custom_model=False,
      backend_type=backend_type,
      backend_kwargs=backend_kwargs,
    )

  @classmethod
  def load_custom(
    cls,
    model_path: Path,
    species_list: Path,
    backend_type: type[VersionedAcousticBackendProtocol],
    backend_kwargs: dict[str, Any],
    check_validity: bool,
  ) -> AcousticModelV3_0:
    assert model_path.exists()
    assert species_list.is_file()

    loaded_species_list = validate_species_list(species_list)

    if check_validity:
      n_species_in_model = BackendLoader.check_model_can_be_loaded(
        model_path, backend_type, backend_kwargs
      )
      if n_species_in_model != len(loaded_species_list):
        raise ValueError(
          f"Model '{model_path.absolute()}' has {n_species_in_model} outputs, but "
          f"species list '{species_list.absolute()}' has "
          f"{len(loaded_species_list)} species!"
        )

    return AcousticModelV3_0(
      model_path,
      loaded_species_list,
      is_custom_model=True,
      backend_type=backend_type,
      backend_kwargs=backend_kwargs,
    )

  @classmethod
  @final
  def get_version(cls) -> ACOUSTIC_MODEL_VERSIONS:
    return ACOUSTIC_MODEL_VERSION_V3_0

  @classmethod
  @final
  def get_sig_fmin(cls) -> int:
    return 0

  @classmethod
  @final
  def get_sig_fmax(cls) -> int:
    return 15_000

  @classmethod
  @final
  def get_sample_rate(cls) -> int:
    return 32_000

  @classmethod
  @final
  def get_segment_size_s(cls) -> float:
    return _DEFAULT_SEGMENT_SIZE_S

  @classmethod
  @final
  def get_segment_size_samples(cls) -> int:
    return _DEFAULT_SEGMENT_SIZE_SAMPLES

  @classmethod
  @final
  def get_embeddings_dim(cls) -> int:
    return 1280

  def encode_session(
    self,
    /,
    *,
    n_producers: int = 1,
    n_workers: int | None = None,
    batch_size: int = 1,
    prefetch_ratio: int = 1,
    overlap_duration_s: float = 0,
    speed: float = 1.0,
    bandpass_fmin: int = 0,
    bandpass_fmax: int = 15_000,
    half_precision: bool = False,
    max_audio_duration_min: float | None = None,
    show_stats: None | Literal["minimal", "progress", "benchmark"] = None,
    progress_callback: Callable[[AcousticProgressStats], None] | None = None,
    device: str | list[str] = "CPU",
    max_n_files: int = 65_536,
    segment_size_s: float = _DEFAULT_SEGMENT_SIZE_S,
  ) -> AcousticEncodingSession:
    return AcousticEncodingSession(
      species_list=self.species_list,
      model_path=self.model_path,
      model_segment_size_s=segment_size_s,
      model_sample_rate=self.get_sample_rate(),
      model_is_custom=self.is_custom_model,
      model_sig_fmin=self.get_sig_fmin(),
      model_sig_fmax=self.get_sig_fmax(),
      model_version=self.get_version(),
      model_backend_type=self.backend_type,
      model_backend_custom_kwargs=self.backend_kwargs,
      model_emb_dim=self.get_embeddings_dim(),
      n_producers=n_producers,
      n_workers=n_workers,
      batch_size=batch_size,
      prefetch_ratio=prefetch_ratio,
      overlap_duration_s=overlap_duration_s,
      speed=speed,
      bandpass_fmin=bandpass_fmin,
      bandpass_fmax=bandpass_fmax,
      half_precision=half_precision,
      max_audio_duration_min=max_audio_duration_min,
      show_stats=show_stats,
      progress_callback=progress_callback,
      device=device,
      max_n_files=max_n_files,
    )

  def predict_session(
    self,
    /,
    *,
    top_k: int | None = 5,
    n_producers: int = 1,
    n_workers: int | None = None,
    batch_size: int = 1,
    prefetch_ratio: int = 1,
    overlap_duration_s: float = 0,
    speed: float = 1.0,
    bandpass_fmin: int = 0,
    bandpass_fmax: int = 15_000,
    apply_sigmoid: bool = True,
    sigmoid_sensitivity: float | None = 1.0,
    default_confidence_threshold: float | None = 0.1,
    custom_confidence_thresholds: dict[str, float] | None = None,
    custom_species_list: str | Path | Collection[str] | None = None,
    half_precision: bool = False,
    max_audio_duration_min: float | None = None,
    show_stats: Literal["minimal", "progress", "benchmark"] | None = None,
    progress_callback: Callable[[AcousticProgressStats], None] | None = None,
    device: str | list[str] = "CPU",
    max_n_files: int = 65_536,
    segment_size_s: float = _DEFAULT_SEGMENT_SIZE_S,
  ) -> AcousticPredictionSession:
    return AcousticPredictionSession(
      species_list=self.species_list,
      model_path=self.model_path,
      model_segment_size_s=segment_size_s,
      model_sample_rate=self.get_sample_rate(),
      model_is_custom=self.is_custom_model,
      model_sig_fmin=self.get_sig_fmin(),
      model_sig_fmax=self.get_sig_fmax(),
      model_version=self.get_version(),
      model_backend_type=self.backend_type,
      model_backend_custom_kwargs=self.backend_kwargs,
      top_k=top_k,
      n_producers=n_producers,
      n_workers=n_workers,
      batch_size=batch_size,
      prefetch_ratio=prefetch_ratio,
      overlap_duration_s=overlap_duration_s,
      speed=speed,
      bandpass_fmin=bandpass_fmin,
      bandpass_fmax=bandpass_fmax,
      apply_sigmoid=apply_sigmoid,
      sigmoid_sensitivity=sigmoid_sensitivity,
      default_confidence_threshold=default_confidence_threshold,
      custom_confidence_thresholds=custom_confidence_thresholds,
      custom_species_list=custom_species_list,
      half_precision=half_precision,
      max_audio_duration_min=max_audio_duration_min,
      show_stats=show_stats,
      progress_callback=progress_callback,
      device=device,
      max_n_files=max_n_files,
    )

  def encode(
    self,
    inp: Path | str | Iterable[Path | str],
    /,
    *,
    n_producers: int = 1,
    n_workers: int | None = None,
    batch_size: int = 1,
    prefetch_ratio: int = 1,
    overlap_duration_s: float = 0,
    speed: float = 1.0,
    bandpass_fmin: int = 0,
    bandpass_fmax: int = 15_000,
    half_precision: bool = False,
    max_audio_duration_min: float | None = None,
    show_stats: None | Literal["minimal", "progress", "benchmark"] = None,
    progress_callback: Callable[[AcousticProgressStats], None] | None = None,
    device: str | list[str] = "CPU",
    segment_size_s: float = _DEFAULT_SEGMENT_SIZE_S,
  ) -> AcousticEncodingResultBase:
    input_files = InferenceConfig.validate_input_files(inp)
    with self.encode_session(
      n_producers=n_producers,
      n_workers=n_workers,
      batch_size=batch_size,
      prefetch_ratio=prefetch_ratio,
      overlap_duration_s=overlap_duration_s,
      speed=speed,
      bandpass_fmin=bandpass_fmin,
      bandpass_fmax=bandpass_fmax,
      half_precision=half_precision,
      max_audio_duration_min=max_audio_duration_min,
      show_stats=show_stats,
      progress_callback=progress_callback,
      device=device,
      max_n_files=len(input_files),
      segment_size_s=segment_size_s,
    ) as session:
      return session.run(input_files)

  def encode_arrays(
    self,
    inp: tuple[npt.NDArray, int] | Iterable[tuple[npt.NDArray, int]],
    /,
    *,
    n_producers: int = 1,
    n_workers: int | None = None,
    batch_size: int = 1,
    prefetch_ratio: int = 1,
    overlap_duration_s: float = 0,
    speed: float = 1.0,
    bandpass_fmin: int = 0,
    bandpass_fmax: int = 15_000,
    half_precision: bool = False,
    max_audio_duration_min: float | None = None,
    show_stats: None | Literal["minimal", "progress", "benchmark"] = None,
    progress_callback: Callable[[AcousticProgressStats], None] | None = None,
    device: str | list[str] = "CPU",
    segment_size_s: float = _DEFAULT_SEGMENT_SIZE_S,
  ) -> AcousticEncodingResultBase:
    input_arrays = InferenceConfig.validate_input_audio(inp)
    with self.encode_session(
      n_producers=n_producers,
      n_workers=n_workers,
      batch_size=batch_size,
      prefetch_ratio=prefetch_ratio,
      overlap_duration_s=overlap_duration_s,
      speed=speed,
      bandpass_fmin=bandpass_fmin,
      bandpass_fmax=bandpass_fmax,
      half_precision=half_precision,
      max_audio_duration_min=max_audio_duration_min,
      show_stats=show_stats,
      progress_callback=progress_callback,
      device=device,
      max_n_files=len(input_arrays),
      segment_size_s=segment_size_s,
    ) as session:
      return session.run_arrays(input_arrays)

  def predict(
    self,
    inp: Path | str | Iterable[Path | str],
    /,
    *,
    top_k: int | None = 5,
    n_producers: int = 1,
    n_workers: int | None = None,
    batch_size: int = 1,
    prefetch_ratio: int = 1,
    overlap_duration_s: float = 0,
    bandpass_fmin: int = 0,
    bandpass_fmax: int = 15_000,
    speed: float = 1.0,
    apply_sigmoid: bool = True,
    sigmoid_sensitivity: float | None = 1.0,
    default_confidence_threshold: float | None = 0.1,
    custom_confidence_thresholds: dict[str, float] | None = None,
    custom_species_list: str | Path | Collection[str] | None = None,
    half_precision: bool = False,
    max_audio_duration_min: float | None = None,
    device: str | list[str] = "CPU",
    show_stats: Literal["minimal", "progress", "benchmark"] | None = None,
    progress_callback: Callable[[AcousticProgressStats], None] | None = None,
    segment_size_s: float = _DEFAULT_SEGMENT_SIZE_S,
  ) -> AcousticPredictionResultBase:
    input_files = InferenceConfig.validate_input_files(inp)
    with self.predict_session(
      top_k=top_k,
      n_producers=n_producers,
      n_workers=n_workers,
      batch_size=batch_size,
      prefetch_ratio=prefetch_ratio,
      overlap_duration_s=overlap_duration_s,
      speed=speed,
      bandpass_fmin=bandpass_fmin,
      bandpass_fmax=bandpass_fmax,
      apply_sigmoid=apply_sigmoid,
      sigmoid_sensitivity=sigmoid_sensitivity,
      default_confidence_threshold=default_confidence_threshold,
      custom_confidence_thresholds=custom_confidence_thresholds,
      custom_species_list=custom_species_list,
      half_precision=half_precision,
      max_audio_duration_min=max_audio_duration_min,
      device=device,
      show_stats=show_stats,
      progress_callback=progress_callback,
      max_n_files=len(input_files),
      segment_size_s=segment_size_s,
    ) as session:
      return session.run(input_files)

  def predict_arrays(
    self,
    inp: tuple[npt.NDArray, int] | Iterable[tuple[npt.NDArray, int]],
    /,
    *,
    top_k: int | None = 5,
    n_producers: int = 1,
    n_workers: int | None = None,
    batch_size: int = 1,
    prefetch_ratio: int = 1,
    overlap_duration_s: float = 0,
    bandpass_fmin: int = 0,
    bandpass_fmax: int = 15_000,
    speed: float = 1.0,
    apply_sigmoid: bool = True,
    sigmoid_sensitivity: float | None = 1.0,
    default_confidence_threshold: float | None = 0.1,
    custom_confidence_thresholds: dict[str, float] | None = None,
    custom_species_list: str | Path | Collection[str] | None = None,
    half_precision: bool = False,
    max_audio_duration_min: float | None = None,
    device: str | list[str] = "CPU",
    show_stats: Literal["minimal", "progress", "benchmark"] | None = None,
    progress_callback: Callable[[AcousticProgressStats], None] | None = None,
    segment_size_s: float = _DEFAULT_SEGMENT_SIZE_S,
  ) -> AcousticPredictionResultBase:
    input_arrays = InferenceConfig.validate_input_audio(inp)
    with self.predict_session(
      top_k=top_k,
      n_producers=n_producers,
      n_workers=n_workers,
      batch_size=batch_size,
      prefetch_ratio=prefetch_ratio,
      overlap_duration_s=overlap_duration_s,
      speed=speed,
      bandpass_fmin=bandpass_fmin,
      bandpass_fmax=bandpass_fmax,
      apply_sigmoid=apply_sigmoid,
      sigmoid_sensitivity=sigmoid_sensitivity,
      default_confidence_threshold=default_confidence_threshold,
      custom_confidence_thresholds=custom_confidence_thresholds,
      custom_species_list=custom_species_list,
      half_precision=half_precision,
      max_audio_duration_min=max_audio_duration_min,
      device=device,
      show_stats=show_stats,
      progress_callback=progress_callback,
      max_n_files=len(input_arrays),
      segment_size_s=segment_size_s,
    ) as session:
      return session.run_arrays(input_arrays)
