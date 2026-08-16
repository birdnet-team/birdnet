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
from pathlib import Path
from typing import ClassVar

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
    # Deterministic and per-segment distinct, so a test can tell which segments
    # actually made it into the result rather than only how many.
    out = np.empty((batch.shape[0], self.n_species_out), dtype=np.float32)
    out[:] = np.linspace(0.0, 1.0, self.n_species_out, dtype=np.float32)
    return out

  def encode(self, batch: np.ndarray) -> np.ndarray:
    self._stall()
    out = np.empty((batch.shape[0], self.emb_dim_out), dtype=np.float32)
    out[:] = np.linspace(0.0, 1.0, self.emb_dim_out, dtype=np.float32)
    return out

  @classmethod
  def supports_encoding(cls) -> bool:
    return True

  @classmethod
  def supports_cow(cls) -> bool:
    return False

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
    return inference_result


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
  n_workers: int = 2,
  n_producers: int = 1,
  batch_size: int = 1,
  top_k: int | None = 5,
  seconds_per_batch: float = 0.0,
  **kwargs: object,
) -> AcousticPredictionSession:
  return AcousticPredictionSession(
    species_list=_species(FakeAcousticBackend.n_species_out),
    model_path=fake_model_path(tmp_path),
    model_segment_size_s=3.0,
    model_sample_rate=48_000,
    model_is_custom=False,
    model_sig_fmin=0,
    model_sig_fmax=15_000,
    model_version="2.4",
    model_backend_type=FakeAcousticBackend,  # type: ignore[arg-type]
    model_backend_custom_kwargs={"seconds_per_batch": seconds_per_batch},
    top_k=top_k,
    n_producers=n_producers,
    n_workers=n_workers,
    batch_size=batch_size,
    prefetch_ratio=1,
    overlap_duration_s=0,
    speed=1.0,
    bandpass_fmin=0,
    bandpass_fmax=15_000,
    apply_sigmoid=True,
    apply_softmax=False,
    sigmoid_sensitivity=1.0,
    default_confidence_threshold=0.1,
    custom_confidence_thresholds=None,
    custom_species_list=None,
    half_precision=False,
    max_audio_duration_min=None,
    show_stats=None,
    progress_callback=None,
    device="CPU",
    max_n_files=1024,
    **kwargs,  # type: ignore[arg-type]
  )


def fake_encode_session(
  tmp_path: Path,
  *,
  n_workers: int = 2,
  n_producers: int = 1,
  batch_size: int = 1,
  seconds_per_batch: float = 0.0,
  **kwargs: object,
) -> AcousticEncodingSession:
  return AcousticEncodingSession(
    species_list=_species(FakeAcousticBackend.n_species_out),
    model_path=fake_model_path(tmp_path),
    model_segment_size_s=3.0,
    model_sample_rate=48_000,
    model_is_custom=False,
    model_sig_fmin=0,
    model_sig_fmax=15_000,
    model_version="2.4",
    model_backend_type=FakeAcousticBackend,  # type: ignore[arg-type]
    model_backend_custom_kwargs={"seconds_per_batch": seconds_per_batch},
    model_emb_dim=FakeAcousticBackend.emb_dim_out,
    n_producers=n_producers,
    n_workers=n_workers,
    batch_size=batch_size,
    prefetch_ratio=1,
    overlap_duration_s=0,
    speed=1.0,
    bandpass_fmin=0,
    bandpass_fmax=15_000,
    half_precision=False,
    max_audio_duration_min=None,
    show_stats=None,
    progress_callback=None,
    device="CPU",
    max_n_files=1024,
    **kwargs,  # type: ignore[arg-type]
  )


def write_silence(path: Path, seconds: float, sample_rate: int = 48_000) -> Path:
  """A wav file of a known length, so segment counts are predictable."""
  import soundfile as sf

  sf.write(path, np.zeros(int(seconds * sample_rate), dtype=np.float32), sample_rate)
  return path


def worker_pid_file(tmp_path: Path) -> Path:
  return tmp_path / f"worker-{os.getpid()}.pid"
