"""A backend that returns fixed output instantly, for testing the pipeline itself.

The acoustic pipeline is a parent process, N producers, N workers and a
performance tracker moving audio through shared memory. Almost none of what can
go wrong in it has anything to do with the model, but until now the only way to
start it was to download and run a real one — so every test of a killed child,
a leaked lock or a teardown path paid tens of seconds and a 50 MB model for the
privilege of exercising a few hundred microseconds of coordination.

That cost is why concurrency bugs here have been found by CI rather than by
tests: a kill point that is expensive to add does not get added. This backend
makes the pipeline itself cheap to start, so a test can afford to kill a child
at a specific moment and assert what happens.

`AcousticPredictionSession` and `AcousticEncodingSession` already take a
`model_backend_type`, so nothing in the library needs a hook for this. The class
lives at module level and is referenced by import path, which is what lets it
survive being pickled into a `spawn` child.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from ordered_set import OrderedSet

from birdnet.acoustic.inference.session import (
  AcousticEncodingSession,
  AcousticPredictionSession,
)

# Small enough that a result block is a few hundred bytes, which keeps the
# fake pipeline off the queue-tearing path unless a test asks for it.
DEFAULT_N_SPECIES = 32
DEFAULT_EMB_DIM = 16


class FakeAcousticBackend:
  """Returns deterministic scores without loading anything.

  Configured through class attributes rather than the constructor: the pipeline
  builds this in the child process, so anything a test wants to vary has to
  survive pickling with the class rather than travel with an instance.
  """

  # Fixed for the whole suite. These are read in the parent to size the species
  # list, and the class travels to the child by import path, so a test that
  # mutated them would change one side only.
  n_species_out: ClassVar[int] = DEFAULT_N_SPECIES
  emb_dim_out: ClassVar[int] = DEFAULT_EMB_DIM

  def __init__(
    self,
    model_path: Path,
    device_name: str,
    half_precision: bool = False,
    seconds_per_batch: float = 0.0,
  ) -> None:
    self._model_path = model_path
    self._device_name = device_name
    self._half_precision = half_precision
    # Wall time to burn per batch, so a test can be certain a worker is inside
    # inference when it kills it. Travels through `model_backend_custom_kwargs`
    # because that is pickled with the loader; a class attribute would not
    # reach the child.
    self._seconds_per_batch = seconds_per_batch

  def load(self) -> None:
    pass

  def unload(self) -> None:
    pass

  def _stall(self) -> None:
    if self._seconds_per_batch > 0:
      time.sleep(self._seconds_per_batch)

  def predict(self, batch: np.ndarray) -> np.ndarray:
    self._stall()
    # Each segment's scores are the same ramp, shifted by that segment's
    # marker (write_marked_audio makes sample 0 identify the segment). Top-k
    # permutes the species axis, so per-species values cannot be asserted on
    # at the far end -- but the *mean* over species survives any permutation
    # and is strictly monotone in the marker, which is what lets a healthy-run
    # test detect a dropped or reordered block by content.
    out = np.empty((batch.shape[0], self.n_species_out), dtype=np.float32)
    out[:] = np.linspace(0.0, 1.0, self.n_species_out, dtype=np.float32)
    out += batch[:, :1]
    return out

  def encode(self, batch: np.ndarray) -> np.ndarray:
    self._stall()
    out = np.empty((batch.shape[0], self.emb_dim_out), dtype=np.float32)
    out[:] = np.linspace(0.0, 1.0, self.emb_dim_out, dtype=np.float32)
    # Embeddings reach the result tensor untouched, so the segment marker that
    # `write_marked_audio` put into the audio can be carried straight through.
    # That is what lets a test assert a segment landed in the right row rather
    # than only that the right number of rows arrived -- the failure mode of a
    # result path that reorders or drops work is silent otherwise.
    out[:, 0] = batch[:, 0]
    return out

  @classmethod
  def supports_encoding(cls) -> bool:
    return True

  @classmethod
  def supports_cow(cls) -> bool:
    return False

  @classmethod
  def applies_sigmoid(cls) -> bool:
    return False

  @classmethod
  def max_batch_size(cls) -> int | None:
    return None

  @classmethod
  def fixed_input_samples(cls) -> int | None:
    return None

  @property
  def n_species(self) -> int:
    return self.n_species_out

  @classmethod
  def precision(cls) -> str:
    return "fp32"

  @classmethod
  def name(cls) -> str:
    return "fake-acoustic"

  def copy_to_device(self, batch: np.ndarray) -> np.ndarray:
    return batch

  def copy_from_device(self, inference_result: np.ndarray) -> np.ndarray:
    return inference_result

  def half_precision(self, inference_result: np.ndarray) -> np.ndarray:
    # As the real backends do, so a worker's precision shows in its output.
    if self._half_precision:
      return inference_result.astype(np.float16, copy=False)
    return inference_result


def loading_marker(model_path: Path) -> Path:
  """Written by the stuck backends right before they hang, for a test to wait on."""
  return model_path.with_name(model_path.name + ".loading")


class NeverLoadingBackend(FakeAcousticBackend):
  """Hangs in `load`, as a model whose interpreter never returns would."""

  def load(self) -> None:
    loading_marker(self._model_path).touch()
    time.sleep(60)


class IgnoringTerminationBackend(FakeAcousticBackend):
  """Hangs in `load` and ignores SIGTERM, as native code with its own handler can."""

  def load(self) -> None:
    import signal

    if hasattr(signal, "SIGTERM"):
      signal.signal(signal.SIGTERM, signal.SIG_IGN)
    loading_marker(self._model_path).touch()
    time.sleep(60)


class FailingToLoadBackend(FakeAcousticBackend):
  """Raises in `load`, as a model with corrupt weights would."""

  def load(self) -> None:
    raise RuntimeError("bad weights")


class UnpicklableErrorBackend(FakeAcousticBackend):
  """Raises a builtin exception that cannot be pickled back to the parent."""

  def load(self) -> None:
    raise RuntimeError("bad weights", lambda: None)


class TimeoutErrorBackend(FakeAcousticBackend):
  """Raises a `TimeoutError` of its own in `load`, as a slow network share can."""

  def load(self) -> None:
    raise TimeoutError("network share read timed out")


def raise_foreign_load_error() -> None:
  """Raises an exception whose class lives in a module only this call imports."""
  from birdnet_tests.foreign_load_error import ForeignLoadError

  raise ForeignLoadError("bad weights")


class ForeignErrorBackend(FakeAcousticBackend):
  """Raises a non-builtin exception, as TensorFlow does for a corrupt model."""

  def load(self) -> None:
    raise_foreign_load_error()


class RaisingBackend(FakeAcousticBackend):
  """Raises inside inference on its ``raise_at_batch``-th batch, as a model would."""

  def __init__(
    self,
    model_path: Path,
    device_name: str,
    half_precision: bool = False,
    seconds_per_batch: float = 0.0,
    raise_at_batch: int = 2,
  ) -> None:
    super().__init__(model_path, device_name, half_precision, seconds_per_batch)
    self._raise_at_batch = raise_at_batch
    self._n_batches = 0

  def predict(self, batch: np.ndarray) -> np.ndarray:
    self._n_batches += 1
    if self._n_batches == self._raise_at_batch:
      raise RuntimeError(f"synthetic model failure in batch {self._n_batches}")
    return super().predict(batch)


class WideFakeAcousticBackend(FakeAcousticBackend):
  """The same stub at realistic output widths, for the queue-tearing surface.

  The default stub keeps blocks at a few hundred bytes, which is the safe side
  of every framing threshold. These widths put one prediction block with
  ``top_k=None`` at ~45 KB and one encoding block at ``batch_size=4`` past
  16 KB -- the sizes at which POSIX splits a queue message into separate
  header and body writes, which is where a killed writer can leave a message
  half-written. Tests about that surface must use this class; the default
  stub cannot reach it.
  """

  n_species_out = 6_000
  emb_dim_out = 1_024


class InGraphSigmoidFakeBackend(FakeAcousticBackend):
  """The stub as a model with the sigmoid in its graph: scores, not logits.

  Every segment gets the same scores: a ramp of float32 sigmoids over the
  logits [-15, 15] with a saturated 0.0 and 1.0 at the ends, which is the
  range the prediction worker's logit recovery has to handle. The segment
  marker is not applied, so ``scores()`` is what a result compares against.
  """

  @classmethod
  def applies_sigmoid(cls) -> bool:
    return True

  @classmethod
  def scores(cls) -> np.ndarray:
    logits = np.linspace(-15.0, 15.0, cls.n_species_out - 2, dtype=np.float32)
    inside = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    return np.concatenate([[0.0], inside, [1.0]]).astype(np.float32)

  def predict(self, batch: np.ndarray) -> np.ndarray:
    self._stall()
    out = np.empty((batch.shape[0], self.n_species_out), dtype=np.float32)
    out[:] = self.scores()
    return out


def fake_model_path(tmp_path: Path) -> Path:
  """A file that exists, which is all the session asserts about the model."""
  path = tmp_path / "fake-model.tflite"
  if not path.exists():
    path.write_bytes(b"")
  return path


def _species(n: int) -> OrderedSet[str]:
  return OrderedSet([f"Fake species {i}_Fake {i}" for i in range(n)])


def fake_predict_session(
  tmp_path: Path,
  *,
  n_workers: int | None = 2,
  n_producers: int = 1,
  batch_size: int = 1,
  top_k: int | None = 5,
  seconds_per_batch: float = 0.0,
  backend: type[FakeAcousticBackend] = FakeAcousticBackend,
  show_stats: Literal["minimal", "progress", "benchmark"] | None = None,
  speed: float = 1.0,
  progress_callback: Callable[..., None] | None = None,
  default_confidence_threshold: float | None = 0.1,
  prefetch_ratio: int = 1,
  overlap_duration_s: float = 0,
  bandpass_fmin: int = 0,
  bandpass_fmax: int = 15_000,
  half_precision: bool = False,
  max_audio_duration_min: float | None = None,
  max_n_files: int = 1024,
  **kwargs: object,
) -> AcousticPredictionSession:
  return AcousticPredictionSession(
    species_list=_species(backend.n_species_out),
    model_path=fake_model_path(tmp_path),
    model_segment_size_s=3.0,
    model_sample_rate=48_000,
    model_is_custom=False,
    model_sig_fmin=0,
    model_sig_fmax=15_000,
    model_version="2.4",
    model_backend_type=backend,  # type: ignore[arg-type]
    model_backend_custom_kwargs={"seconds_per_batch": seconds_per_batch},
    top_k=top_k,
    n_producers=n_producers,
    n_workers=n_workers,
    batch_size=batch_size,
    prefetch_ratio=prefetch_ratio,
    overlap_duration_s=overlap_duration_s,
    speed=speed,
    bandpass_fmin=bandpass_fmin,
    bandpass_fmax=bandpass_fmax,
    apply_sigmoid=True,
    apply_softmax=False,
    sigmoid_sensitivity=1.0,
    default_confidence_threshold=default_confidence_threshold,
    custom_confidence_thresholds=None,
    custom_species_list=None,
    half_precision=half_precision,
    max_audio_duration_min=max_audio_duration_min,
    show_stats=show_stats,
    progress_callback=progress_callback,
    device="CPU",
    max_n_files=max_n_files,
    **kwargs,  # type: ignore[arg-type]
  )


def fake_encode_session(
  tmp_path: Path,
  *,
  n_workers: int | None = 2,
  n_producers: int = 1,
  batch_size: int = 1,
  seconds_per_batch: float = 0.0,
  backend: type[FakeAcousticBackend] = FakeAcousticBackend,
  show_stats: Literal["minimal", "progress", "benchmark"] | None = None,
  speed: float = 1.0,
  progress_callback: Callable[..., None] | None = None,
  prefetch_ratio: int = 1,
  overlap_duration_s: float = 0,
  bandpass_fmin: int = 0,
  bandpass_fmax: int = 15_000,
  half_precision: bool = False,
  max_audio_duration_min: float | None = None,
  max_n_files: int = 1024,
  **kwargs: object,
) -> AcousticEncodingSession:
  return AcousticEncodingSession(
    species_list=_species(backend.n_species_out),
    model_path=fake_model_path(tmp_path),
    model_segment_size_s=3.0,
    model_sample_rate=48_000,
    model_is_custom=False,
    model_sig_fmin=0,
    model_sig_fmax=15_000,
    model_version="2.4",
    model_backend_type=backend,  # type: ignore[arg-type]
    model_backend_custom_kwargs={"seconds_per_batch": seconds_per_batch},
    model_emb_dim=backend.emb_dim_out,
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
    device="CPU",
    max_n_files=max_n_files,
    **kwargs,  # type: ignore[arg-type]
  )


def segment_marker(segment_index: int) -> float:
  """The constant sample value carried by one segment of marked audio."""
  return (segment_index + 1) / 1000.0


def write_marked_audio(
  path: Path,
  n_segments: int,
  segment_size_s: float = 3.0,
  sample_rate: int = 48_000,
) -> Path:
  """A wav file whose every segment is a constant identifying that segment.

  The pipeline is configured here so nothing alters the samples on the way in:
  the file's rate matches the model's, speed is 1.0, and the bandpass bounds
  equal the model's own, which is what switches filtering off. So the value the
  backend sees is the value written here, and `encode` can hand it back as the
  first element of the embedding.
  """
  import soundfile as sf

  per_segment = int(segment_size_s * sample_rate)
  samples = np.empty(n_segments * per_segment, dtype=np.float32)
  for i in range(n_segments):
    samples[i * per_segment : (i + 1) * per_segment] = segment_marker(i)
  # Float samples, not the wav default of PCM-16, so the marker survives the
  # file exactly: at 16 bits a marker of 0.001 comes back as 32/32768.
  # Quantisation would only force the assertions to guess at a tolerance.
  sf.write(path, samples, sample_rate, subtype="FLOAT")
  return path


def worker_pid_file(tmp_path: Path) -> Path:
  return tmp_path / f"worker-{os.getpid()}.pid"
