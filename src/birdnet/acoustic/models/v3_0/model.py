from __future__ import annotations

import csv
import io
from collections.abc import Callable, Collection, Iterable
from pathlib import Path
from typing import Any, Literal, final

import numpy.typing as npt
from ordered_set import OrderedSet

from birdnet.acoustic.inference.configs import InferenceConfig, PredictionConfig
from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticEncodingResultBase,
  AcousticFileEncodingResult,
)
from birdnet.acoustic.inference.core.perf_tracker import AcousticProgressStats
from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticFilePredictionResult,
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
from birdnet.utils.helper import (
  directory_lock,
  validate_species_list,
  write_text_atomic,
)
from birdnet.utils.label_manifest import (
  LabelInput,
  ensure_artifact,
  get_manifest_path,
  labels_up_to_date,
  record_generation,
)
from birdnet.utils.local_data import APP_DIR
from birdnet.utils.taxonomy_v3 import (
  ensure_taxonomy_v3_available,
  get_taxonomy_v3_input,
  get_taxonomy_v3_path,
)

_LABELS_DL_URL = "https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_Labels.csv"
_LABELS_DL_SIZE = 809172
_LABELS_DL_SHA256 = "8124b0ea2d187104c5e2cd95a0f937165647e20349c8fd34d4d5ef991821f8f0"
_DEFAULT_SEGMENT_SIZE_S = 3.0
_DEFAULT_SEGMENT_SIZE_SAMPLES = 96_000

# No Estonian ("et"): the v0.2-Jun2026 taxonomy has no common_name_et column,
# so every Estonian name would silently be the English one.
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
# Named after the content, so an install of another version cannot overwrite
# this release's copy with its own and leave both re-downloading forever.
_LABELS_RAW_PATH = _ACOUSTIC_V3_0_BASE_DIR / f"labels_raw-{_LABELS_DL_SHA256[:12]}.csv"
_LEGACY_LABELS_RAW_PATH = _ACOUSTIC_V3_0_BASE_DIR / "labels_raw.csv"
_SETUP_LOCK_DIR = _ACOUSTIC_V3_0_BASE_DIR / ".labels_setup.lock"

# Identifies this generator in the manifest, so a directory written by the geo
# model could never be read as an acoustic one.
_GENERATOR_NAME = "acoustic_v3_0"
# Bump whenever a change here would produce different <lang>.txt bytes from the
# same inputs - the join key, the tie-break, the fallback, the line format. The
# golden-digest test fails until this and the expected digests agree.
_GENERATION_VERSION = 1


class AcousticDownloaderBaseV3_0:
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
    # Checked before the lock: a current cache then needs no write at all, which
    # keeps read-only and pre-populated app data directories usable.
    if cls._check_labels_available():
      return

    with directory_lock(_SETUP_LOCK_DIR, "acoustic model v3.0 label setup"):
      if cls._check_labels_available():
        return

      ensure_artifact(
        cls._labels_input(),
        "Downloading acoustic model v3.0 labels",
        legacy_path=_LEGACY_LABELS_RAW_PATH,
      )
      ensure_taxonomy_v3_available()

      cls._generate_lang_files()

      # Both inputs were just verified by digest, so this can only fail if
      # generation itself is wrong. Raising beats silently regenerating on every
      # later load while serving names nobody checked.
      if not cls._check_labels_available():
        raise RuntimeError(
          "The acoustic model v3.0 label files could not be generated from verified "
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
    labels_raw = _LABELS_RAW_PATH.read_bytes()
    taxonomy_raw = get_taxonomy_v3_path().read_bytes()

    species_order: list[tuple[str, str]] = []
    with io.StringIO(labels_raw.decode("utf-8"), newline="") as f:
      reader = csv.DictReader(f, delimiter=";")
      for row in reader:
        sci_name = row.get("sci_name", "").strip()
        en_us_name = row.get("com_name", "").strip()
        if sci_name and en_us_name:
          species_order.append((sci_name, en_us_name))

    # Joined on the scientific name because these labels carry no code the
    # taxonomy shares - unlike the geo model, which joins on species_code. See
    # the module docstring of birdnet/utils/taxonomy_v3.py; the two keys are
    # deliberate and the trade-off is described there.
    taxonomy: dict[str, dict[str, str]] = {}
    with io.StringIO(taxonomy_raw.decode("utf-8"), newline="") as f:
      reader = csv.DictReader(f)
      for row in reader:
        sci_name = row.get("sci_name", "").strip()
        if sci_name:
          taxonomy[sci_name] = dict(row)

    n_unresolved = 0
    written: list[Path] = []
    for lang, col in _LANGUAGE_TO_COLUMN.items():
      lang_file = lang_dir / f"{lang}.txt"
      lines: list[str] = []
      unresolved = 0
      for sci_name, en_us_name in species_order:
        tax_row = taxonomy.get(sci_name, {})
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
    on_file_complete: Callable[[AcousticFileEncodingResult], None] | None = None,
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
      on_file_complete=on_file_complete,
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
    apply_softmax: bool = False,
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
    on_file_complete: Callable[[AcousticFilePredictionResult], None] | None = None,
  ) -> AcousticPredictionSession:
    """Create a prediction session for the BirdNET 3.0 model.

    Scores: every V3.0 export (tf, pb, pt, onnx — official and custom alike)
    applies the sigmoid inside the model graph, so scores leave the model as
    probabilities. ``apply_sigmoid=True`` (the default) returns them unchanged —
    no second sigmoid is applied — and ``apply_sigmoid=False`` returns the
    identical raw model output. Because the model does not expose logits,
    ``sigmoid_sensitivity`` values other than ``1.0`` and ``apply_softmax=True``
    raise a ``ValueError``.
    """
    if apply_softmax:
      raise ValueError(
        "apply_softmax is not supported for acoustic V3.0 models: the exports "
        "apply a sigmoid inside the model graph, so the logits a softmax needs "
        "are not available."
      )
    if apply_sigmoid:
      sigmoid_sensitivity = PredictionConfig.validate_sigmoid_sensitivity(
        sigmoid_sensitivity
      )
    # the sensitivity can never take effect for V3.0, so a non-default value
    # is rejected rather than silently ignored even when apply_sigmoid=False.
    if sigmoid_sensitivity is not None and sigmoid_sensitivity != 1.0:
      raise ValueError(
        "sigmoid_sensitivity is not supported for acoustic V3.0 models: the "
        "exports apply a plain sigmoid inside the model graph, so a scaled "
        "sigmoid cannot be applied. Leave it at its default of 1.0."
      )
    # The model output is already a probability; applying the pipeline
    # sigmoid on top would squash every score into [0.5, 0.73].
    apply_sigmoid = False
    sigmoid_sensitivity = None
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
      apply_softmax=apply_softmax,
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
      on_file_complete=on_file_complete,
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
    on_file_complete: Callable[[AcousticFileEncodingResult], None] | None = None,
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
      on_file_complete=on_file_complete,
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
    apply_softmax: bool = False,
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
    on_file_complete: Callable[[AcousticFilePredictionResult], None] | None = None,
  ) -> AcousticPredictionResultBase:
    """Run prediction with the BirdNET 3.0 model on files or paths.

    Scores are probabilities as the model emits them: the V3.0 exports apply
    the sigmoid inside the model graph. ``apply_sigmoid=True`` (the default)
    returns them unchanged and ``apply_sigmoid=False`` returns the identical
    raw model output, so confidence thresholds are probabilities either way.
    ``sigmoid_sensitivity`` values other than ``1.0`` and ``apply_softmax=True``
    raise a ``ValueError``, because both need the logits the exports do not
    expose. Custom V3.0 models are expected to output probabilities as well.

    ``n_workers`` sets the number of inference worker processes. Its default value,
    ``None``, uses the number of physical CPU cores. Pass a fixed integer to meet a
    scheduler or container process limit. Each worker holds its own copy of the
    model, so a high count raises peak memory use. On Linux and macOS, a worker
    killed by the operating system to reclaim memory *while processing a batch*
    deadlocks the run: the killed process never releases the lock it was
    holding, so the remaining workers wait on it forever and the call never
    returns (see issue #73). Lowering ``n_workers`` or ``batch_size`` reduces
    peak memory and with it how likely such a kill is, but cannot rule it out.

    This method creates one prediction session for the call. That session shuts down
    its producer and worker processes before this method returns, including when
    inference raises an exception.
    """
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
      apply_softmax=apply_softmax,
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
      on_file_complete=on_file_complete,
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
    apply_softmax: bool = False,
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
      apply_softmax=apply_softmax,
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
