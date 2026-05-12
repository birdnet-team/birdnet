from __future__ import annotations

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
from birdnet.globals import ACOUSTIC_MODEL_VERSIONS
from birdnet.utils.helper import validate_species_list


class AcousticModelPerchV2(AcousticModelBase):
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
  ) -> AcousticModelPerchV2:
    result = AcousticModelPerchV2(
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
    backend_type: type[VersionedAcousticBackendProtocol],
    backend_kwargs: dict[str, Any],
    check_validity: bool,
  ) -> AcousticModelPerchV2:
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

    result = AcousticModelPerchV2(
      model_path,
      loaded_species_list,
      is_custom_model=True,
      backend_type=backend_type,
      backend_kwargs=backend_kwargs,
    )
    return result

  @classmethod
  @final
  def get_version(cls) -> ACOUSTIC_MODEL_VERSIONS:
    return "v2"  # type: ignore

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
    return 5.0

  @classmethod
  @final
  def get_segment_size_samples(cls) -> int:
    return 160_000  # 5.0 * 32_000

  @classmethod
  @final
  def get_embeddings_dim(cls) -> int:
    return 1536

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
    max_n_files: int = 65_536,  # Limit to avoid excessive memory usage
  ) -> AcousticEncodingSession:
    """Create an encoding session with explicit resource configuration.

    Args:
      species_list: Ordered species collection used during the session.
      model_path: Path to the acoustic model binary.
      n_producers: Threads tasked with producing audio batches.
      n_workers: Optional worker count for backend processing.
      batch_size: Number of records evaluated per inference call.
      prefetch_ratio: How many batches to decode ahead of processing.
      overlap_duration_s: Seconds of overlap between sliding windows.
      speed: Resampling multiplier to accommodate different recording speeds.
      bandpass_fmin: Lower bound for the bandpass filter in Hz.
      bandpass_fmax: Upper bound for the bandpass filter in Hz.
      half_precision: Use float16 where supported for inference.
      max_audio_duration_min: Maximum total duration per call.
      show_stats: Level of statistics logging to emit.
      progress_callback: Optional callback to report progress.
      device: Target device(s) for running the backend.
      max_n_files: Upper bound on files to limit resource consumption.

    Returns:
      AcousticEncodingSession: Session capable of running encodings.
    """
    return AcousticEncodingSession(
      species_list=self.species_list,
      model_path=self.model_path,
      model_segment_size_s=self.get_segment_size_s(),
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
    apply_sigmoid: bool = False,
    sigmoid_sensitivity: float | None = None,
    default_confidence_threshold: float | None = 0.1,
    custom_confidence_thresholds: dict[str, float] | None = None,
    custom_species_list: str | Path | Collection[str] | None = None,
    half_precision: bool = False,
    max_audio_duration_min: float | None = None,
    show_stats: Literal["minimal", "progress", "benchmark"] | None = None,
    progress_callback: Callable[[AcousticProgressStats], None] | None = None,
    device: str | list[str] = "CPU",
    max_n_files: int = 65_536,  # Limit to avoid excessive memory usage
  ) -> AcousticPredictionSession:
    """Create a prediction session allowing manual control over the inference lifecycle.

    Args:
      species_list: Ordered species collection used during the session.
      model_path: Path to the acoustic model binary.
      top_k: Number of highest-confidence results to return per segment.
      n_producers: Threads tasked with producing audio batches.
      n_workers: Optional worker count for backend processing.
      batch_size: Number of records evaluated per inference call.
      prefetch_ratio: How many batches to decode ahead of processing.
      overlap_duration_s: Seconds of overlap between sliding windows.
      bandpass_fmin: Lower bound for the bandpass filter in Hz.
      bandpass_fmax: Upper bound for the bandpass filter in Hz.
      speed: Resampling multiplier to accommodate different recording speeds.
      apply_sigmoid: Whether to transform logits with a sigmoid.
        When False, output scores are raw logits and thresholds are interpreted in
        logit space rather than as probabilities.
      sigmoid_sensitivity: Optional scale for the sigmoid function.
      default_confidence_threshold: Base threshold to emit a detection.
        When apply_sigmoid=True this is a probability (typical range 0 to 1);
        when apply_sigmoid=False it is a logit value.
      custom_confidence_thresholds: Species-specific override thresholds.
      custom_species_list: Path or iterable defining a subset of species.
      half_precision: Use float16 where supported for inference.
      max_audio_duration_min: Maximum total duration per call.
      show_stats: Level of statistics logging to emit.
      progress_callback: Optional callback to report progress.
      device: Target device(s) for running the backend.
      max_n_files: Upper bound on files to limit resource consumption.

    Returns:
      AcousticPredictionSession: Session capable of running predictions.
    """
    return AcousticPredictionSession(
      species_list=self.species_list,
      model_path=self.model_path,
      model_segment_size_s=self.get_segment_size_s(),
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
  ) -> AcousticEncodingResultBase:
    """Run encoding with the Perch V2 model on files or paths to obtain embeddings.

    Args:
      inp: Path(s) or string(s) pointing to audio files to encode.
      n_producers: Threads tasked with producing audio batches.
      n_workers: Optional worker count for backend processing.
      batch_size: Number of records evaluated per inference call.
      prefetch_ratio: How many batches to decode ahead of processing.
      overlap_duration_s: Seconds of overlap between sliding windows.
      speed: Resampling multiplier to accommodate different recording speeds.
      bandpass_fmin: Lower bound for the bandpass filter in Hz.
      bandpass_fmax: Upper bound for the bandpass filter in Hz.
      half_precision: Use float16 where supported for inference.
      max_audio_duration_min: Maximum total duration per call.
      show_stats: Level of statistics logging to emit.
      progress_callback: Optional callback to report progress.
      device: Target device(s) for running the backend.

    Returns:
      AcousticEncodingResultBase: Object containing embeddings for each file.
    """
    input_files = InferenceConfig.validate_input_files(inp)
    max_n_files = len(input_files)

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
      max_n_files=max_n_files,
      device=device,
      progress_callback=progress_callback,
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
  ) -> AcousticEncodingResultBase:
    """Run encoding with the Perch V2 model directly on in-memory audio arrays.

    Args:
      inp: Tuple(s) of (audio ndarray, sampling rate).
      n_producers: Threads generating batches from the arrays.
      n_workers: Optional worker count for backend processing.
      batch_size: Number of records evaluated per inference call.
      prefetch_ratio: How many batches to decode ahead of processing.
      overlap_duration_s: Seconds of overlap between sliding windows.
      speed: Resampling multiplier to accommodate different recording speeds.
      bandpass_fmin: Lower bound for the bandpass filter in Hz.
      bandpass_fmax: Upper bound for the bandpass filter in Hz.
      half_precision: Use float16 where supported for inference.
      max_audio_duration_min: Maximum total duration per call.
      show_stats: Level of statistics logging to emit.
      progress_callback: Optional callback to report progress.
      device: Target device(s) for running the backend.

    Returns:
      AcousticEncodingResultBase: Object containing embeddings for each input array.
    """
    input_arrays = InferenceConfig.validate_input_audio(inp)
    max_n_files = len(input_arrays)

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
      max_n_files=max_n_files,
      device=device,
      progress_callback=progress_callback,
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
    apply_sigmoid: bool = False,
    sigmoid_sensitivity: float | None = None,
    default_confidence_threshold: float | None = 0.1,
    custom_confidence_thresholds: dict[str, float] | None = None,
    custom_species_list: str | Path | Collection[str] | None = None,
    half_precision: bool = False,
    max_audio_duration_min: float | None = None,
    device: str | list[str] = "CPU",
    show_stats: Literal["minimal", "progress", "benchmark"] | None = None,
    progress_callback: Callable[[AcousticProgressStats], None] | None = None,
  ) -> AcousticPredictionResultBase:
    """Run prediction with the Perch V2 model on files or paths with configurable
    inference options.

    Args:
      inp: Path(s) or string(s) pointing to audio files to analyze.
      top_k: Number of highest-confidence results to return per segment.
      n_producers: Threads tasked with producing audio batches.
      n_workers: Optional worker count for backend processing.
      batch_size: Number of records evaluated per inference call.
      prefetch_ratio: How many batches to decode ahead of processing.
      overlap_duration_s: Seconds of overlap between sliding windows.
      bandpass_fmin: Lower bound for the bandpass filter in Hz.
      bandpass_fmax: Upper bound for the bandpass filter in Hz.
      speed: Resampling multiplier to accommodate different recording speeds.
      apply_sigmoid: Whether to transform logits with a sigmoid.
        When False, output scores are raw logits and thresholds are interpreted in
        logit space rather than as probabilities.
      sigmoid_sensitivity: Optional scale for the sigmoid function.
      default_confidence_threshold: Base threshold to emit a detection.
        When apply_sigmoid=True this is a probability (typical range 0 to 1);
        when apply_sigmoid=False it is a logit value.
      custom_confidence_thresholds: Species-specific override thresholds.
      custom_species_list: Path or iterable defining a subset of species.
      half_precision: Use float16 where supported for inference.
      max_audio_duration_min: Maximum total duration per call.
      device: Target device(s) for running the backend.
      show_stats: Level of statistics logging to emit.
      progress_callback: Optional callback to report progress.

    Returns:
      AcousticPredictionResultBase: Object containing detected species and confidence
        scores.
    """
    input_files = InferenceConfig.validate_input_files(inp)
    max_n_files = len(input_files)

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
      show_stats=show_stats,
      progress_callback=progress_callback,
      max_n_files=max_n_files,
      device=device,
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
    apply_sigmoid: bool = False,
    sigmoid_sensitivity: float | None = None,
    default_confidence_threshold: float | None = 0.1,
    custom_confidence_thresholds: dict[str, float] | None = None,
    custom_species_list: str | Path | Collection[str] | None = None,
    half_precision: bool = False,
    max_audio_duration_min: float | None = None,
    device: str | list[str] = "CPU",
    show_stats: Literal["minimal", "progress", "benchmark"] | None = None,
    progress_callback: Callable[[AcousticProgressStats], None] | None = None,
  ) -> AcousticPredictionResultBase:
    """Run prediction with the Perch V2 model directly on in-memory audio arrays.

    Args:
      inp: Tuple(s) of (audio ndarray, sampling rate).
      top_k: Number of highest-confidence results to return per segment.
      n_producers: Threads generating batches from the arrays.
      n_workers: Optional worker count for backend processing.
      batch_size: Number of records evaluated per inference call.
      prefetch_ratio: How many batches to decode ahead of processing.
      overlap_duration_s: Seconds of overlap between sliding windows.
      bandpass_fmin: Lower bound for the bandpass filter in Hz.
      bandpass_fmax: Upper bound for the bandpass filter in Hz.
      speed: Resampling multiplier to accommodate different recording speeds.
      apply_sigmoid: Whether to transform logits with a sigmoid.
        When False, output scores are raw logits and thresholds are interpreted in
        logit space rather than as probabilities.
      sigmoid_sensitivity: Optional scale for the sigmoid function.
      default_confidence_threshold: Base threshold to emit a detection.
        When apply_sigmoid=True this is a probability (typical range 0 to 1);
        when apply_sigmoid=False it is a logit value.
      custom_confidence_thresholds: Species-specific override thresholds.
      custom_species_list: Path or iterable defining a subset of species.
      half_precision: Use float16 where supported for inference.
      max_audio_duration_min: Maximum total duration per call.
      device: Target device(s) for running the backend.
      show_stats: Level of statistics logging to emit.
      progress_callback: Optional callback to report progress.

    Returns:
      AcousticPredictionResultBase: Object containing detected species and confidence
        scores.
    """
    input_arrays = InferenceConfig.validate_input_audio(inp)
    max_n_files = len(input_arrays)

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
      show_stats=show_stats,
      progress_callback=progress_callback,
      max_n_files=max_n_files,
      device=device,
    ) as session:
      return session.run_arrays(input_arrays)
