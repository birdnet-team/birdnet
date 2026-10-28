from __future__ import annotations

import csv
import io
from collections.abc import Callable, Collection, Iterable
from pathlib import Path
from typing import ClassVar, Literal, final

import numpy.typing as npt
from ordered_set import OrderedSet

from birdnet.acoustic.inference.configs import InferenceConfig
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
from birdnet.acoustic.models.base import (
  AcousticModelBase,
  resolve_in_graph_sigmoid_options,
  validate_batch_size_for_backend,
  validate_segment_size_for_backend,
)
from birdnet.globals import (
  ACOUSTIC_MODEL_VERSION_V3_0,
  ACOUSTIC_MODEL_VERSIONS,
  MODEL_BACKENDS,
  MODEL_PRECISIONS,
)
from birdnet.utils.downloads import ModelInfo, ensure_single_file_model
from birdnet.utils.label_manifest import LabelInput
from birdnet.utils.local_data import (
  get_birdnet_app_data_folder,
  get_lang_dir,
  get_model_path,
)
from birdnet.utils.species_lists import read_official_labels
from birdnet.utils.taxonomy_v3 import (
  ensure_taxonomy_v3_available,
  get_taxonomy_v3_input,
)
from birdnet.utils.v3_labels import V3LabelDownloaderBase

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


def _acoustic_v3_0_base_dir() -> Path:
  return get_birdnet_app_data_folder() / "acoustic-models" / "v3.0"


# Named after the content, so an install of another version cannot overwrite
# this release's copy with its own and leave both re-downloading forever.
def _labels_raw_path() -> Path:
  return _acoustic_v3_0_base_dir() / f"labels_raw-{_LABELS_DL_SHA256[:12]}.csv"


def _legacy_labels_raw_path() -> Path:
  return _acoustic_v3_0_base_dir() / "labels_raw.csv"


def _setup_lock_dir() -> Path:
  return _acoustic_v3_0_base_dir() / ".labels_setup.lock"


# Identifies this generator in the manifest, so a directory written by the geo
# model could never be read as an acoustic one.
_GENERATOR_NAME = "acoustic_v3_0"
# Bump whenever a change would produce different <lang>.txt bytes from the same
# inputs - the join key here, or the fallback or line format in
# birdnet/utils/v3_labels.py, which is shared and so changes both models. The
# golden-digest test fails when the bytes generated from its fixture move;
# nothing forces this bump when its digests are updated.
_GENERATION_VERSION = 2


class AcousticDownloaderBaseV3_0(V3LabelDownloaderBase):
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
    return "acoustic model v3.0"

  @classmethod
  def _parse_labels(cls, labels_raw: bytes) -> list[tuple[str, str, str]]:
    species_order: list[tuple[str, str, str]] = []
    with io.StringIO(labels_raw.decode("utf-8"), newline="") as f:
      reader = csv.DictReader(f, delimiter=";")
      for row in reader:
        sci_name = row.get("sci_name", "").strip()
        en_us_name = row.get("com_name", "").strip()
        if sci_name and en_us_name:
          species_order.append((sci_name, sci_name, en_us_name))
    return species_order

  @classmethod
  def _index_taxonomy(
    cls, taxonomy_raw: bytes, species: list[tuple[str, str, str]]
  ) -> dict[str, dict[str, str]]:
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
    return taxonomy


class _AcousticSingleFileDownloaderV3_0(AcousticDownloaderBaseV3_0):
  """Shared logic of the single-file acoustic V3.0 downloads (tf, onnx, pt).

  A subclass names its backend and its precision table; the paths, the
  availability check and the download follow from those two.
  """

  _backend: ClassVar[MODEL_BACKENDS]
  _models: ClassVar[dict[MODEL_PRECISIONS, ModelInfo]]

  @classmethod
  def _get_lang_dir(cls) -> Path:
    return get_lang_dir("acoustic", "3.0", cls._backend)

  @classmethod
  def _get_paths(cls, precision: MODEL_PRECISIONS) -> tuple[Path, Path]:
    model_path = get_model_path(
      "acoustic",
      "3.0",
      cls._backend,
      precision,
      content_tag=cls._models[precision].content_tag,
    )
    return model_path, cls._get_lang_dir()

  @classmethod
  def _check_acoustic_model_available(cls, precision: MODEL_PRECISIONS) -> bool:
    model_path, lang_dir = cls._get_paths(precision)
    if not model_path.is_file():
      return False
    if model_path.stat().st_size != cls._models[precision].file_size:
      return False
    if not lang_dir.is_dir():
      return False
    return all((lang_dir / f"{lang}.txt").is_file() for lang in cls.AVAILABLE_LANGUAGES)

  @classmethod
  def _download_model(cls, precision: MODEL_PRECISIONS) -> None:
    model_path, _lang_dir = cls._get_paths(precision)
    ensure_single_file_model(
      cls._models[precision],
      model_path,
      legacy_path=get_model_path("acoustic", "3.0", cls._backend, precision),
      description=f"Downloading acoustic model v3.0 ({cls._backend}, {precision})",
    )

  @classmethod
  def get_model_path_and_labels(
    cls, lang: str, precision: MODEL_PRECISIONS
  ) -> tuple[Path, OrderedSet[str]]:
    if precision not in cls._models:
      raise ValueError(
        f"Unsupported model precision for acoustic {cls._backend} model: "
        f"{precision}. Currently supported precisions are: {', '.join(cls._models)}."
      )
    assert lang in cls.AVAILABLE_LANGUAGES

    cls.ensure_labels_available()
    if not cls._check_acoustic_model_available(precision):
      cls._download_model(precision)
    assert cls._check_acoustic_model_available(precision)

    model_path, _ = cls._get_paths(precision)
    lang_file = cls.get_lang_file(lang)
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = read_official_labels(
      lang_file, AcousticModelV3_0.get_n_classes(), "acoustic model v3.0"
    )
    return model_path, labels


class AcousticModelV3_0(AcousticModelBase):
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

  @classmethod
  @final
  def get_n_classes(cls) -> int:
    """Return the 11,560 classes of the official 3.0 output.

    Species of birds and other animals, plus classes above species level such
    as Acrididae (grasshoppers), Bufo (toads) and Cetacea (whales and dolphins).
    """
    return 11_560

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
    """Create an encoding session with explicit resource configuration.

    Args:
      n_producers: Processes tasked with producing audio batches.
      n_workers: Optional worker count for backend processing.
      batch_size: Number of records evaluated per inference call.
      prefetch_ratio: How many batches to decode ahead of processing.
      overlap_duration_s: Seconds of overlap between sliding windows.
      speed: Resampling multiplier to accommodate different recording speeds.
      bandpass_fmin: Lower bound for the bandpass filter in Hz.
      bandpass_fmax: Upper bound for the bandpass filter in Hz.
      half_precision: Cast the model's outputs to float16 after inference;
        inference itself runs at the model's precision.
      max_audio_duration_min: Longest single input, in minutes, the segment
        indices are sized for; ``None`` sizes them for any length. An input with
        more segments than they can hold raises an error.
      show_stats: Level of statistics logging to emit.
      progress_callback: Optional callback to report progress. Invoked from a
        background worker thread, inheriting a copy of the caller's context
        (contextvars) as captured when the call starts.
      device: Target device(s) for running the backend.
      max_n_files: Upper bound on files to limit resource consumption.
      segment_size_s: Length in seconds of the audio segments passed to the model.
        The official ``tf``, ``onnx`` and ``pt`` exports accept lengths other than
        the default 3.0 s. ``pb`` and the custom classifier exports take only 3.0 s
        and raise a ``ValueError`` for any other length; a custom TFLite export
        loaded with ``check_validity=False`` and an explicit version fails during
        the run instead. ``overlap_duration_s`` must be smaller than this value.
      on_file_complete: Optional callback fired once per input file as soon as
        that file is fully processed, receiving a single-file
        AcousticFileEncodingResult (invalid files are reported with their input
        marked unprocessable). Enables streaming per-file persistence. Invoked
        from a background thread with a copy of the caller's context; file
        inputs only (not run_arrays). A callback that raises cancels the run.

    Returns:
      AcousticEncodingSession: Session capable of running encodings.
    """
    validate_batch_size_for_backend(batch_size, self.backend_type)
    validate_segment_size_for_backend(
      segment_size_s, self.get_sample_rate(), self.backend_type
    )
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
    applies the sigmoid inside the model graph, so scores leave the model with
    the sigmoid applied. ``apply_sigmoid=True`` (the default) returns them unchanged —
    no second sigmoid is applied — and ``apply_sigmoid=False`` returns the
    identical raw model output. A ``sigmoid_sensitivity`` other than ``1.0``
    is applied as it is to the 2.4 models: the logits are recovered from the
    model's scores and the pipeline's sigmoid runs on them, so thresholds see
    the transformed scores; with ``apply_sigmoid=False`` the sensitivity is
    ignored, as for the 2.4 models. ``apply_softmax=True`` raises a
    ``ValueError``.

    The ``tf`` and ``pb`` exports (official and custom) have a static batch
    dimension of 1, so ``batch_size`` above 1 raises a ``ValueError`` for them;
    ``onnx`` and ``pt`` take any batch size.

    Args:
      top_k: Number of highest-confidence results to return per segment.
      n_producers: Processes tasked with producing audio batches.
      n_workers: Optional worker count for backend processing.
      batch_size: Number of records evaluated per inference call.
      prefetch_ratio: How many batches to decode ahead of processing.
      overlap_duration_s: Seconds of overlap between sliding windows.
      speed: Resampling multiplier to accommodate different recording speeds.
      bandpass_fmin: Lower bound for the bandpass filter in Hz.
      bandpass_fmax: Upper bound for the bandpass filter in Hz.
      apply_sigmoid: The 3.0 exports apply the sigmoid inside the model graph, so
        False returns the model's scores unchanged, as does True (the default) at
        the default ``sigmoid_sensitivity`` of 1.0. False ignores
        ``sigmoid_sensitivity``.
      apply_softmax: Must stay False; True raises a ``ValueError`` for the 3.0
        models.
      sigmoid_sensitivity: Shifts the logits, recovered from the model's scores,
        by ``10 * (sigmoid_sensitivity - 1)`` before the sigmoid, so values above
        1.0 raise the scores and values below lower them; between 0.5 and 1.5,
        ``None`` means 1.0. Ignored with ``apply_sigmoid=False``.
      default_confidence_threshold: Base threshold to emit a detection, as a
        sigmoid score (range 0 to 1).
      custom_confidence_thresholds: Species-specific override thresholds.
      custom_species_list: Path or iterable defining a subset of species.
      half_precision: Cast the model's outputs to float16 after inference;
        inference itself runs at the model's precision.
      max_audio_duration_min: Longest single input, in minutes, the segment
        indices are sized for; ``None`` sizes them for any length. An input with
        more segments than they can hold raises an error.
      show_stats: Level of statistics logging to emit.
      progress_callback: Optional callback to report progress. Invoked from a
        background worker thread, inheriting a copy of the caller's context
        (contextvars) as captured when the call starts.
      device: Target device(s) for running the backend.
      max_n_files: Upper bound on files to limit resource consumption.
      segment_size_s: Length in seconds of the audio segments passed to the model.
        The official ``tf``, ``onnx`` and ``pt`` exports accept lengths other than
        the default 3.0 s. ``pb`` and the custom classifier exports take only 3.0 s
        and raise a ``ValueError`` for any other length; a custom TFLite export
        loaded with ``check_validity=False`` and an explicit version fails during
        the run instead. ``overlap_duration_s`` must be smaller than this value.
      on_file_complete: Optional callback fired once per input file as soon as
        that file is fully processed, receiving a single-file
        AcousticFilePredictionResult (invalid files are reported with their
        input marked unprocessable). Enables streaming per-file persistence
        (e.g. resumable analysis). Invoked from a background thread with a copy
        of the caller's context; file inputs only (not run_arrays). A callback
        that raises cancels the run.

    Returns:
      AcousticPredictionSession: Session capable of running predictions.
    """
    validate_batch_size_for_backend(batch_size, self.backend_type)
    validate_segment_size_for_backend(
      segment_size_s, self.get_sample_rate(), self.backend_type
    )
    apply_sigmoid, sigmoid_sensitivity = resolve_in_graph_sigmoid_options(
      apply_sigmoid, apply_softmax, sigmoid_sensitivity, "acoustic V3.0 models"
    )
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
    """Run encoding with the BirdNET 3.0 model on files or paths to obtain embeddings.

    Args:
      inp: Path(s) or string(s) pointing to audio files to encode.
      n_producers: Processes tasked with producing audio batches.
      n_workers: Optional worker count for backend processing.
      batch_size: Number of records evaluated per inference call.
      prefetch_ratio: How many batches to decode ahead of processing.
      overlap_duration_s: Seconds of overlap between sliding windows.
      speed: Resampling multiplier to accommodate different recording speeds.
      bandpass_fmin: Lower bound for the bandpass filter in Hz.
      bandpass_fmax: Upper bound for the bandpass filter in Hz.
      half_precision: Cast the model's outputs to float16 after inference;
        inference itself runs at the model's precision.
      max_audio_duration_min: Longest single input, in minutes, the segment
        indices are sized for; ``None`` sizes them for any length. An input with
        more segments than they can hold raises an error.
      show_stats: Level of statistics logging to emit.
      progress_callback: Optional callback to report progress. Invoked from a
        background worker thread, inheriting a copy of the caller's context
        (contextvars) as captured when the call starts.
      device: Target device(s) for running the backend.
      segment_size_s: Length in seconds of the audio segments passed to the model.
        The official ``tf``, ``onnx`` and ``pt`` exports accept lengths other than
        the default 3.0 s. ``pb`` and the custom classifier exports take only 3.0 s
        and raise a ``ValueError`` for any other length; a custom TFLite export
        loaded with ``check_validity=False`` and an explicit version fails during
        the run instead. ``overlap_duration_s`` must be smaller than this value.
      on_file_complete: Optional callback fired once per input file as soon as
        that file is fully processed, receiving a single-file
        AcousticFileEncodingResult (invalid files are reported with their input
        marked unprocessable). Enables streaming per-file persistence. Invoked
        from a background thread with a copy of the caller's context; file
        inputs only (not encode_arrays). A callback that raises cancels the run.

    Returns:
      AcousticEncodingResultBase: Object containing embeddings for each file.
    """
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
    """Run encoding with the BirdNET 3.0 model directly on in-memory audio arrays.

    Args:
      inp: Tuple(s) of (audio ndarray, sampling rate).
      n_producers: Processes generating batches from the arrays.
      n_workers: Optional worker count for backend processing.
      batch_size: Number of records evaluated per inference call.
      prefetch_ratio: How many batches to decode ahead of processing.
      overlap_duration_s: Seconds of overlap between sliding windows.
      speed: Resampling multiplier to accommodate different recording speeds.
      bandpass_fmin: Lower bound for the bandpass filter in Hz.
      bandpass_fmax: Upper bound for the bandpass filter in Hz.
      half_precision: Cast the model's outputs to float16 after inference;
        inference itself runs at the model's precision.
      max_audio_duration_min: Longest single input, in minutes, the segment
        indices are sized for; ``None`` sizes them for any length. An input with
        more segments than they can hold raises an error.
      show_stats: Level of statistics logging to emit.
      progress_callback: Optional callback to report progress. Invoked from a
        background worker thread, inheriting a copy of the caller's context
        (contextvars) as captured when the call starts.
      device: Target device(s) for running the backend.
      segment_size_s: Length in seconds of the audio segments passed to the model.
        The official ``tf``, ``onnx`` and ``pt`` exports accept lengths other than
        the default 3.0 s. ``pb`` and the custom classifier exports take only 3.0 s
        and raise a ``ValueError`` for any other length; a custom TFLite export
        loaded with ``check_validity=False`` and an explicit version fails during
        the run instead. ``overlap_duration_s`` must be smaller than this value.

    Returns:
      AcousticEncodingResultBase: Object containing embeddings for each input array.
    """
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

    Scores are returned as the model emits them: the V3.0 exports apply
    the sigmoid inside the model graph. ``apply_sigmoid=True`` (the default)
    returns them unchanged and ``apply_sigmoid=False`` returns the identical
    raw model output, so confidence thresholds are sigmoid scores either way.
    A ``sigmoid_sensitivity`` other than ``1.0`` is applied as it is to the 2.4
    models, on logits recovered from the scores, and is ignored with
    ``apply_sigmoid=False`` as there. ``apply_softmax=True`` raises a
    ``ValueError``. Custom V3.0 models are expected to apply the sigmoid as
    well.

    This method creates one prediction session for the call. That session shuts down
    its producer and worker processes before this method returns, including when
    inference raises an exception.

    Args:
      inp: Path(s) or string(s) pointing to audio files to analyze.
      top_k: Number of highest-confidence results to return per segment.
      n_producers: Processes tasked with producing audio batches.
      n_workers: Number of inference worker processes. ``None`` uses the number of
        physical CPU cores. Pass a fixed integer to meet a scheduler or container
        process limit. Each worker holds its own copy of the model, so a high
        count raises peak memory use. In the cases the pipeline guards against, a
        worker killed by the operating system to reclaim memory, including while
        it is processing a batch, is reported as an error naming the dead worker
        rather than hanging the call (see issue #73). Lowering ``n_workers`` or
        ``batch_size`` reduces peak memory and with it how likely such a kill is,
        but cannot rule it out.
      batch_size: Number of records evaluated per inference call.
      prefetch_ratio: How many batches to decode ahead of processing.
      overlap_duration_s: Seconds of overlap between sliding windows.
      bandpass_fmin: Lower bound for the bandpass filter in Hz.
      bandpass_fmax: Upper bound for the bandpass filter in Hz.
      speed: Resampling multiplier to accommodate different recording speeds.
      apply_sigmoid: The 3.0 exports apply the sigmoid inside the model graph, so
        False returns the model's scores unchanged, as does True (the default) at
        the default ``sigmoid_sensitivity`` of 1.0. False ignores
        ``sigmoid_sensitivity``.
      apply_softmax: Must stay False; True raises a ``ValueError`` for the 3.0
        models.
      sigmoid_sensitivity: Shifts the logits, recovered from the model's scores,
        by ``10 * (sigmoid_sensitivity - 1)`` before the sigmoid, so values above
        1.0 raise the scores and values below lower them; between 0.5 and 1.5,
        ``None`` means 1.0. Ignored with ``apply_sigmoid=False``.
      default_confidence_threshold: Base threshold to emit a detection, as a
        sigmoid score (range 0 to 1).
      custom_confidence_thresholds: Species-specific override thresholds.
      custom_species_list: Path or iterable defining a subset of species.
      half_precision: Cast the model's outputs to float16 after inference;
        inference itself runs at the model's precision.
      max_audio_duration_min: Longest single input, in minutes, the segment
        indices are sized for; ``None`` sizes them for any length. An input with
        more segments than they can hold raises an error.
      device: Target device(s) for running the backend.
      show_stats: Level of statistics logging to emit.
      progress_callback: Optional callback to report progress. Invoked from a
        background worker thread, inheriting a copy of the caller's context
        (contextvars) as captured when the call starts.
      segment_size_s: Length in seconds of the audio segments passed to the model.
        The official ``tf``, ``onnx`` and ``pt`` exports accept lengths other than
        the default 3.0 s. ``pb`` and the custom classifier exports take only 3.0 s
        and raise a ``ValueError`` for any other length; a custom TFLite export
        loaded with ``check_validity=False`` and an explicit version fails during
        the run instead. ``overlap_duration_s`` must be smaller than this value.
      on_file_complete: Optional callback fired once per input file as soon as
        that file is fully processed, receiving a single-file
        AcousticFilePredictionResult (invalid files are reported with their
        input marked unprocessable). Enables streaming per-file persistence
        (e.g. resumable analysis). Invoked from a background thread with a copy
        of the caller's context; file inputs only (not predict_arrays). A callback
        that raises cancels the run.

    Returns:
      AcousticPredictionResultBase: Object containing detected species and confidence
        scores.
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
    """Run prediction with the BirdNET 3.0 model directly on in-memory audio arrays.

    Args:
      inp: Tuple(s) of (audio ndarray, sampling rate).
      top_k: Number of highest-confidence results to return per segment.
      n_producers: Processes generating batches from the arrays.
      n_workers: Optional worker count for backend processing.
      batch_size: Number of records evaluated per inference call.
      prefetch_ratio: How many batches to decode ahead of processing.
      overlap_duration_s: Seconds of overlap between sliding windows.
      bandpass_fmin: Lower bound for the bandpass filter in Hz.
      bandpass_fmax: Upper bound for the bandpass filter in Hz.
      speed: Resampling multiplier to accommodate different recording speeds.
      apply_sigmoid: The 3.0 exports apply the sigmoid inside the model graph, so
        False returns the model's scores unchanged, as does True (the default) at
        the default ``sigmoid_sensitivity`` of 1.0. False ignores
        ``sigmoid_sensitivity``.
      apply_softmax: Must stay False; True raises a ``ValueError`` for the 3.0
        models.
      sigmoid_sensitivity: Shifts the logits, recovered from the model's scores,
        by ``10 * (sigmoid_sensitivity - 1)`` before the sigmoid, so values above
        1.0 raise the scores and values below lower them; between 0.5 and 1.5,
        ``None`` means 1.0. Ignored with ``apply_sigmoid=False``.
      default_confidence_threshold: Base threshold to emit a detection, as a
        sigmoid score (range 0 to 1).
      custom_confidence_thresholds: Species-specific override thresholds.
      custom_species_list: Path or iterable defining a subset of species.
      half_precision: Cast the model's outputs to float16 after inference;
        inference itself runs at the model's precision.
      max_audio_duration_min: Longest single input, in minutes, the segment
        indices are sized for; ``None`` sizes them for any length. An input with
        more segments than they can hold raises an error.
      device: Target device(s) for running the backend.
      show_stats: Level of statistics logging to emit.
      progress_callback: Optional callback to report progress. Invoked from a
        background worker thread, inheriting a copy of the caller's context
        (contextvars) as captured when the call starts.
      segment_size_s: Length in seconds of the audio segments passed to the model.
        The official ``tf``, ``onnx`` and ``pt`` exports accept lengths other than
        the default 3.0 s. ``pb`` and the custom classifier exports take only 3.0 s
        and raise a ``ValueError`` for any other length; a custom TFLite export
        loaded with ``check_validity=False`` and an explicit version fails during
        the run instead. ``overlap_duration_s`` must be smaller than this value.

    Returns:
      AcousticPredictionResultBase: Object containing detected species and confidence
        scores.
    """
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
