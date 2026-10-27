"""The settings both strategies share reach every result and benchmark meta.

Each family runs on the fake backend away from every default, so a field that
falls back to its default or swaps with another setting shows here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from birdnet.acoustic.inference.core.benchmarking import (
  FullBenchmarkMetaBase,
  MinimalBenchmarkMetaBase,
)
from birdnet.acoustic.inference.core.encoding.encoding_benchmarking import (
  FullBenchmarkEmbMeta,
)
from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticFileEncodingResult,
)
from birdnet.acoustic.inference.core.prediction.prediction_benchmarking import (
  FullBenchmarkMeta,
)
from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticFilePredictionResult,
)
from birdnet.acoustic.inference.core.result_base import AcousticResultBase
from birdnet.acoustic.inference.session import (
  AcousticEncodingSession,
  AcousticPredictionSession,
)
from birdnet.globals import MODEL_TYPE_ACOUSTIC, NA
from birdnet.utils.segments import get_n_segments_speed
from birdnet_tests.fake_acoustic_backend import (
  DEFAULT_EMB_DIM,
  DEFAULT_N_SPECIES,
  fake_encode_session,
  fake_model_path,
  fake_predict_session,
  write_marked_audio,
)

pytestmark = pytest.mark.no_tf

_N_WORKERS = 2
_PREFETCH_RATIO = 4
_SPEED = 2.0
_OVERLAP_S = 1.0
_SETTINGS: dict[str, object] = {
  "n_workers": _N_WORKERS,
  "batch_size": 3,
  "prefetch_ratio": _PREFETCH_RATIO,
  "speed": _SPEED,
  "overlap_duration_s": _OVERLAP_S,
  "bandpass_fmin": 100,
  "bandpass_fmax": 14_000,
  "half_precision": True,
  # Tracks performance, which the full meta needs, without writing a
  # benchmark directory.
  "show_stats": "progress",
}
_SEGMENTS_PER_FILE = (4, 7)
_SEGMENT_S = 3.0

_SHM_FIELDS = {
  "mem_shm_size_file_indices_MiB": "rf_file_indices",
  "mem_shm_size_segment_indices_MiB": "rf_segment_indices",
  "mem_shm_size_audio_samples_MiB": "rf_audio_samples",
  "mem_shm_size_batch_sizes_MiB": "rf_batch_sizes",
  "mem_shm_size_flags_MiB": "rf_flags",
}
_MINIMAL_FIELDS = [
  "file_formats",
  "file_segments_total",
  "model_segment_duration_seconds",
  "mem_result_total_memory_usage_MiB",
  *_SHM_FIELDS,
]


@dataclass
class _Run:
  result: AcousticResultBase
  per_file: list[AcousticResultBase]
  minimal: MinimalBenchmarkMetaBase
  full: FullBenchmarkMetaBase
  shm_MiB: dict[str, float]
  start_method: str


def _typed(values: dict[str, object]) -> dict[str, tuple[str, object]]:
  # True == 1 == 1.0, so the type is compared too.
  return {k: (type(v).__name__, v) for k, v in values.items()}


def _run(
  factory: Callable[..., AcousticPredictionSession | AcousticEncodingSession],
  tmp_path: Path,
) -> _Run:
  files = [
    write_marked_audio(tmp_path / f"{i}.wav", n)
    for i, n in enumerate(_SEGMENTS_PER_FILE)
  ]
  per_file: list[AcousticResultBase] = []
  with factory(tmp_path, **_SETTINGS, on_file_complete=per_file.append) as session:
    result = session.run(files)
    args = (session._conf, session._specific_config, session._resources, result)
    ring = session._resources.ring_buffer_resources
    return _Run(
      result=result,
      per_file=sorted(per_file, key=lambda r: str(r.inputs[0])),
      minimal=session._strategy.create_minimal_benchmark_meta(*args),
      full=session._strategy.create_full_benchmark_meta(*args),
      shm_MiB={
        field: getattr(ring, name).nbytes / 1024**2
        for field, name in _SHM_FIELDS.items()
      },
      start_method=session._conf.start_method,
    )


def _assert_shared(run: _Run, tmp_path: Path) -> None:
  for result in (run.result, *run.per_file):
    assert _typed(
      {
        "segment_duration_s": result.segment_duration_s,
        "overlap_duration_s": result.overlap_duration_s,
        "speed": result.speed,
        "model_fmin": result.model_fmin,
        "model_fmax": result.model_fmax,
        "model_sr": result.model_sr,
        "model_precision": result.model_precision,
        "model_version": result.model_version,
        "model_path": result.model_path,
      }
    ) == _typed(
      {
        "segment_duration_s": _SEGMENT_S,
        "overlap_duration_s": _OVERLAP_S,
        "speed": _SPEED,
        "model_fmin": 0,
        "model_fmax": 15_000,
        "model_sr": 48_000,
        "model_precision": "fp32",
        "model_version": "2.4",
        "model_path": fake_model_path(tmp_path),
      }
    )
  assert len(run.per_file) == len(_SEGMENTS_PER_FILE)

  durations = [n * _SEGMENT_S for n in _SEGMENTS_PER_FILE]
  n_segments = [
    get_n_segments_speed(d, _SEGMENT_S, _OVERLAP_S, _SPEED) for d in durations
  ]
  np.testing.assert_array_equal(run.result.input_durations, durations)
  np.testing.assert_array_equal(run.full._file_durations, durations)

  expected_full = {
    "file_formats": "WAV",
    "file_segments_total": sum(n_segments),
    "file_segments_maximum": max(n_segments),
    "model_segment_duration_seconds": _SEGMENT_S,
    "param_producers": 1,
    "param_workers": _N_WORKERS,
    "param_batch_size": 3,
    "param_prefetch_ratio": _PREFETCH_RATIO,
    "param_overlap_seconds": _OVERLAP_S,
    "param_bandpass_fmin": 100,
    "param_bandpass_fmax": 14_000,
    "param_half_precision": True,
    "param_devices": "CPU",
    "param_inference_library": NA,
    "mem_shm_ringsize": _N_WORKERS + _N_WORKERS * _PREFETCH_RATIO,
    "model_type": MODEL_TYPE_ACOUSTIC,
    "model_backend": "fake-acoustic",
    "model_version": "2.4",
    "model_is_custom": False,
    "model_path": str(fake_model_path(tmp_path).absolute()),
    "model_species": DEFAULT_N_SPECIES,
    "model_precision": "fp32",
    "model_sample_rate": 48_000,
    "model_sig_fmin": 0,
    "model_sig_fmax": 15_000,
    "sw_start_method": run.start_method,
    **run.shm_MiB,
  }
  assert _typed({k: getattr(run.full, k) for k in expected_full}) == _typed(
    expected_full
  )
  assert _typed({k: getattr(run.minimal, k) for k in _MINIMAL_FIELDS}) == _typed(
    {k: getattr(run.full, k) for k in _MINIMAL_FIELDS}
  )
  np.testing.assert_array_equal(run.minimal._file_durations, durations)


def test_prediction_results_and_meta_carry_the_settings(tmp_path: Path) -> None:
  run = _run(fake_predict_session, tmp_path)

  _assert_shared(run, tmp_path)
  for result in (run.result, *run.per_file):
    assert isinstance(result, AcousticFilePredictionResult)
    assert result.species_probs.dtype == np.float16
  assert isinstance(run.full, FullBenchmarkMeta)
  expected = {
    "param_top_k": 5,
    "param_sigmoid_apply": True,
    "param_softmax_apply": False,
    "param_sigmoid_sensitivity": 1.0,
    "param_confidence_threshold_default": 0.1,
    "param_confidence_threshold_custom": 0,
    "param_custom_species": 0,
  }
  assert _typed({k: getattr(run.full, k) for k in expected}) == _typed(expected)


def test_encoding_results_and_meta_carry_the_settings(tmp_path: Path) -> None:
  run = _run(fake_encode_session, tmp_path)

  _assert_shared(run, tmp_path)
  for result in (run.result, *run.per_file):
    assert isinstance(result, AcousticFileEncodingResult)
    assert result.embeddings.dtype == np.float16
  assert isinstance(run.full, FullBenchmarkEmbMeta)
  assert run.full.model_emb_dim == DEFAULT_EMB_DIM
