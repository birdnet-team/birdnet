from pathlib import Path

import pytest
from ordered_set import OrderedSet

from birdnet.acoustic.inference.configs import ModelConfig
from birdnet_tests.fake_acoustic_backend import FakeAcousticBackend

pytestmark = pytest.mark.no_tf


def _config(segment_size_s: float, sample_rate: int) -> ModelConfig:
  return ModelConfig(
    species_list=OrderedSet(["species_a"]),
    path=Path("model"),
    version="3.0",
    segment_size_s=segment_size_s,
    sample_rate=sample_rate,
    sig_fmin=0,
    sig_fmax=15_000,
    is_custom=False,
    backend_type=FakeAcousticBackend,  # type: ignore[arg-type]
    backend_kwargs={},
  )


@pytest.mark.parametrize(
  ("segment_size_s", "sample_rate", "samples"),
  [
    (3.0, 48_000, 144_000),
    (3.0, 32_000, 96_000),
    (2.5, 32_000, 80_000),
    # 2.01 * 32000 is 64319.99999999999 in floating point.
    (2.01, 32_000, 64_320),
  ],
)
def test_segment_size_samples_rounds(
  segment_size_s: float, sample_rate: int, samples: int
) -> None:
  assert _config(segment_size_s, sample_rate).segment_size_samples == samples
