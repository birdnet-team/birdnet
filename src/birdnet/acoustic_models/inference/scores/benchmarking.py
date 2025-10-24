from __future__ import annotations

from dataclasses import dataclass

from birdnet.acoustic_models.inference.benchmarking import (
  FullBenchmarkMetaBase,
  MinimalBenchmarkMetaBase,
)


@dataclass
class MinimalBenchmarkMeta(MinimalBenchmarkMetaBase):
  pass


@dataclass
class FullBenchmarkMeta(FullBenchmarkMetaBase):
  param_top_k: int | None
  param_sigmoid_apply: bool
  param_sigmoid_sensitivity: float | None
  param_confidence_threshold_default: float | None
  param_confidence_threshold_custom: int
  param_custom_species: int
