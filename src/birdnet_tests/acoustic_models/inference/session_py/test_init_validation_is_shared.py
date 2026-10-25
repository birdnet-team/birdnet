"""Both sessions validate and assemble the arguments they share identically."""

from __future__ import annotations

from pathlib import Path

import pytest

from birdnet.acoustic.inference.configs import (
  FilteringConfig,
  InferenceConfig,
  ModelConfig,
  OutputConfig,
  ProcessingConfig,
)
from birdnet_tests.fake_acoustic_backend import (
  FakeAcousticBackend,
  fake_encode_session,
  fake_model_path,
  fake_predict_session,
)

pytestmark = pytest.mark.no_tf


def _progress(stats: object) -> None:
  pass


def _on_file(result: object) -> None:
  pass


@pytest.mark.parametrize(
  ("name", "value", "error", "match"),
  [
    pytest.param(*row, id=row[0])
    for row in [
      ("n_producers", 0, ValueError, r"n_producers must be >= 1"),
      ("n_workers", 0, ValueError, r"n_workers must be >= 1"),
      ("batch_size", 0, ValueError, r"batch size must be >= 1"),
      ("prefetch_ratio", -1, ValueError, r"prefetch_ratio must be >= 0"),
      ("overlap_duration_s", 3.0, ValueError, r"overlap duration must be in"),
      ("speed", 0, ValueError, r"speed must be in the range"),
      ("bandpass_fmin", 20_000, ValueError, r"bandpass frequencies must be in"),
      ("half_precision", "yes", TypeError, r"half_precision must be a boolean"),
      ("max_audio_duration_min", 0, ValueError, r"max_audio_duration_min must be"),
      ("show_stats", "verbose", ValueError, r"show stats must be one of"),
      # The fake sessions default to show_stats=None.
      ("progress_callback", _progress, ValueError, r"Progress callback can only"),
      ("max_n_files", 0, ValueError, r"max_n_files must be >= 1"),
      ("on_file_complete", 42, TypeError, r"on_file_complete must be callable"),
    ]
  ],
)
def test_an_invalid_shared_argument_is_rejected_alike(
  tmp_path: Path, name: str, value: object, error: type[Exception], match: str
) -> None:
  with pytest.raises(error, match=match) as predict_error:
    fake_predict_session(tmp_path, **{name: value})
  with pytest.raises(error, match=match) as encode_error:
    fake_encode_session(tmp_path, **{name: value})

  assert type(predict_error.value) is type(encode_error.value)
  assert str(predict_error.value) == str(encode_error.value)


def test_shared_arguments_land_where_they_belong(tmp_path: Path) -> None:
  # Distinct values, so a swapped field shows; only a bool can equal 0 or 1.
  shared: dict[str, object] = {
    "n_producers": 1,
    "n_workers": 2,
    "batch_size": 4,
    "prefetch_ratio": 3,
    "max_n_files": 7,
    "overlap_duration_s": 1.5,
    "speed": 2.0,
    "max_audio_duration_min": 10.0,
    "bandpass_fmin": 100,
    "bandpass_fmax": 14_000,
    "half_precision": True,
    "show_stats": "benchmark",
    "progress_callback": _progress,
    "on_file_complete": _on_file,
  }
  predict = fake_predict_session(tmp_path, **shared)  # type: ignore[arg-type]
  encode = fake_encode_session(tmp_path, **shared)  # type: ignore[arg-type]

  expected = InferenceConfig(
    model_conf=ModelConfig(
      # The only list argument, passed through as is.
      species_list=predict._conf.model_conf.species_list,
      path=fake_model_path(tmp_path),
      version="2.4",
      segment_size_s=3.0,
      sample_rate=48_000,
      sig_fmin=0,
      sig_fmax=15_000,
      is_custom=False,
      backend_type=FakeAcousticBackend,
      backend_kwargs={"seconds_per_batch": 0.0},
    ),
    processing_conf=ProcessingConfig(
      producers=1,
      workers=2,
      batch_size=4,
      prefetch_ratio=3,
      overlap_duration_s=1.5,
      speed=2.0,
      half_precision=True,
      max_audio_duration_min=10.0,
      device="CPU",
      max_n_files=7,
    ),
    filtering_conf=FilteringConfig(bandpass_fmin=100, bandpass_fmax=14_000),
    output_conf=OutputConfig(
      show_stats="benchmark",
      progress_callback=_progress,
      file_completion_callback=_on_file,
    ),
  )
  assert predict._conf == expected
  assert encode._conf == expected
